import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from flask import Flask, Response, jsonify, request, send_from_directory  # type: ignore
from flask_cors import CORS
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
import requests
from dotenv import load_dotenv

# ---------- Load .env from backend folder ----------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP = Flask(
    __name__, static_folder="../frontend", static_url_path="/"
)  # แก้ static_folder ให้ชี้ไปที่ /frontend
CORS(APP, resources={r"/*": {"origins": "*"}})  # เปิด CORS

# ---------- ENV ----------
API_BASE = os.getenv("TESA_API_BASE", "https://tesa-api.crma.dev/api").rstrip("/")
SOCKET_URL = os.getenv(
    "TESA_SOCKET_URL",
    "https://tesa-api.crma.dev/api/object-detection/550e8400-e29b-41d4-a716-446655440000",
).rstrip("/")

DEF_ID = os.getenv("DEFENCE_CAMERA_ID", "").strip()

DEF_TK = os.getenv("DEFENCE_CAMERA_TOKEN", "").strip()
OFF_ID = os.getenv("OFFENCE_CAMERA_ID", "").strip()
OFF_TK = os.getenv("OFFENCE_CAMERA_TOKEN", "").strip()


PORT = int(os.getenv("PORT", "5001"))

# ---------- Database Configuration ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tesa_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Connection pool for database
_db_pool = None


def init_db_pool():
    """Initialize database connection pool"""
    global _db_pool
    # Skip initialization if DB credentials are not set
    if not all([DB_HOST, DB_NAME, DB_USER]):
        logging.warning("Database credentials not fully configured. Database features will be disabled.")
        _db_pool = None
        return False
    
    try:
        _db_pool = psycopg2.pool.SimpleConnectionPool(
            1,  # min connections
            10,  # max connections
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logging.info("Database connection pool initialized successfully")
        return True
    except psycopg2.OperationalError as e:
        logging.warning(f"Database connection failed (database may not be set up): {e}")
        logging.info("Tip: Set DB_HOST, DB_NAME, DB_USER, DB_PASSWORD in .env file to enable database features")
        _db_pool = None
        return False
    except Exception as e:
        logging.warning(f"Failed to initialize database pool: {e}")
        _db_pool = None
        return False


def ensure_db_pool():
    """Ensure database pool is initialized (returns True if successful)"""
    global _db_pool
    if _db_pool is None:
        return init_db_pool()
    return True


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    if not ensure_db_pool() or _db_pool is None:
        raise Exception("Database not available. Please configure database connection in .env file")
    
    conn = None
    try:
        conn = _db_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            _db_pool.putconn(conn)


def get_db_cursor():
    """Get a database cursor with RealDictCursor for dict-like results"""
    if not ensure_db_pool() or _db_pool is None:
        raise Exception("Database not available. Please configure database connection in .env file")
    conn = _db_pool.getconn()
    return conn, conn.cursor(cursor_factory=RealDictCursor)


# ---------- Helpers ----------
def _h(token: str):
    return {"x-camera-token": token}


def _u(*parts: str) -> str:
    """Build URL from API_BASE and parts"""
    url = API_BASE
    for s in parts:
        url = url.rstrip("/") + "/" + s.lstrip("/")
    return url


def _proxy(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)

        logging.info(
            "PROXY %s %s %s -> %s", request.path, url, params or {}, r.status_code
        )
        # ถ่ายโอน Headers ทั้งหมด ยกเว้น Content-Encoding ที่อาจทำให้ Flask มีปัญหา
        excluded_headers = ["content-encoding"]
        response_headers = [
            (name, value)
            for name, value in r.headers.items()
            if name.lower() not in excluded_headers
        ]

        return Response(r.content, r.status_code, response_headers)
    except requests.RequestException as e:
        logging.error("PROXY ERROR %s %s %s -> %s", request.path, url, params or {}, e)
        return jsonify({"ok": False, "error": str(e)}), 500  



@APP.after_request
def _hdr(resp):
    # Security Headers

    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Cache Control

    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# ---------- Route สำหรับเสิร์ฟไฟล์ Frontend (แก้ไข) ----------
@APP.route("/")
def root():
    return send_from_directory(APP.static_folder, "monitor.html")



@APP.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(APP.static_folder + "/js", filename)


@APP.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(APP.static_folder + "/css", filename)


# ---------- Health / Host / Metrics ----------
@APP.get("/health")
def health():
    db_ok = False
    db_error = None
    try:
        if _db_pool:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            db_ok = True
    except Exception as e:
        db_ok = False
        db_error = str(e)

    # App is OK if cameras are configured, database is optional
    app_ok = all([DEF_ID, DEF_TK, OFF_ID, OFF_TK])

    return jsonify(
        {
            "ok": app_ok,
            "api": API_BASE,
            "socket": SOCKET_URL,
            "def_set": bool(DEF_ID and DEF_TK),
            "off_set": bool(OFF_ID and OFF_TK),
            "db_connected": db_ok,
            "db_error": db_error,
            "routes": [str(r) for r in APP.url_map.iter_rules()],
        }
    )



@APP.get("/host/metrics")
def host_metrics():
    # Get CPU Temperature (Linux specific)
    cpu = None
    try:
        for p in (
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
        ):
            try:
                with open(p) as f:
                    cpu = int(f.read().strip()) / 1000.0
                    break
            except Exception:
                continue
    except Exception:
        cpu = None

    # Get Uptime (Linux specific)
    uptime_sec = None
    try:
        with open("/proc/uptime", "r") as f:
            uptime_sec = float(f.readline().split()[0])
    except Exception:
        uptime_sec = None

    return jsonify(
        {"time": int(time.time()), "cpu_temp_c": cpu, "uptime_sec": uptime_sec}
    )



# ---------- Config for frontend ----------
@APP.get("/config")
def config():
    return jsonify(
        {
            "socket_url": SOCKET_URL,
            "defence": {"camera_id": DEF_ID, "set": bool(DEF_ID and DEF_TK)},
            "offence": {"camera_id": OFF_ID, "set": bool(OFF_ID and OFF_TK)},
            "rest": {
                "defence": {
                    "info": "/api/defence/info",
                    "recent": "/api/defence/recent",
                },
                "offence": {
                    "info": "/api/offence/info",
                    "recent": "/api/offence/recent",
                },
                # Image base URL จะอยู่ที่ root ของ API URL (ตัด '/api')
                "image_base": API_BASE.replace("/api", ""),
            },
        }
    )


# ---------- API proxies (ประกาศก่อน static) ----------
@APP.get("/api/defence/info")
def def_info():
    if not (DEF_ID and DEF_TK):
        return jsonify({"ok": False, "error": "DEF camera not set"}), 400
    # เรียก TESA_API_BASE/object-detection/info/{DEF_ID}
    return _proxy(_u("object-detection", "info", DEF_ID), headers=_h(DEF_TK))


@APP.get("/api/defence/recent")
def def_recent():
    if not (DEF_ID and DEF_TK):
        return jsonify({"ok": False, "error": "DEF camera not set"}), 400
    limit = request.args.get("limit", "10")
    # เรียก TESA_API_BASE/object-detection/{DEF_ID}?limit={limit}
    return _proxy(
        _u("object-detection", DEF_ID), headers=_h(DEF_TK), params={"limit": limit}
    )


# *** โค้ดที่เพิ่มเข้ามา (Offence Side) ***
@APP.get("/api/offence/info")
def off_info():
    if not (OFF_ID and OFF_TK):
        return jsonify({"ok": False, "error": "OFF camera not set"}), 400
    # เรียก TESA_API_BASE/object-detection/info/{OFF_ID}
    return _proxy(_u("object-detection", "info", OFF_ID), headers=_h(OFF_TK))


@APP.get("/api/offence/recent")
def off_recent():
    if not (OFF_ID and OFF_TK):
        return jsonify({"ok": False, "error": "OFF camera not set"}), 400
    limit = request.args.get("limit", "10")
    # เรียก TESA_API_BASE/object-detection/{OFF_ID}?limit={limit}
    return _proxy(
        _u("object-detection", OFF_ID), headers=_h(OFF_TK), params={"limit": limit}
    )


if __name__ == "__main__":
    # Initialize database connection pool
    init_db_pool()
    APP.run(host="0.0.0.0", port=PORT, debug=True)


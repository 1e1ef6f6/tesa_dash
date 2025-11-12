import os
import time
import logging
from urllib.parse import urljoin
from flask import Flask, jsonify, send_from_directory, request, Response # type: ignore
from flask_cors import CORS # type: ignore
from dotenv import load_dotenv # type: ignore
import requests # type: ignore
from pathlib import Path

# ---------- Load .env from backend folder ----------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP = Flask(__name__, static_folder="../frontend", static_url_path="/") # แก้ static_folder ให้ชี้ไปที่ /frontend
CORS(APP, resources={r"/*": {"origins": "*"}}) # เปิด CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- ENV ----------
API_BASE    = os.getenv("TESA_API_BASE", "https://tesa-api.crma.dev/api").rstrip("/")
SOCKET_URL = os.getenv("TESA_SOCKET_URL", "https://tesa-api.crma.dev/api/object-detection/550e8400-e29b-41d4-a716-446655440000").rstrip("/")

DEF_ID = os.getenv("DEFENCE_CAMERA_ID", "").strip()
DEF_TK = os.getenv("DEFENCE_CAMERA_TOKEN", "").strip()
OFF_ID = os.getenv("OFFENCE_CAMERA_ID", "").strip()
OFF_TK = os.getenv("OFFENCE_CAMERA_TOKEN", "").strip()

PORT = int(os.getenv("PORT", "5000"))

# ---------- Helpers ----------
def _h(token: str):
    return {"x-camera-token": token}

def _u(*parts: str) -> str:
    base = API_BASE
    for s in parts:
        base = urljoin(base + "/", s.strip("/"))
    return base

def _proxy(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        logging.info("PROXY %s %s %s -> %s", request.path, url, params or {}, r.status_code)
        # ถ่ายโอน Headers ทั้งหมด ยกเว้น Content-Encoding ที่อาจทำให้ Flask มีปัญหา
        excluded_headers = ['content-encoding'] 
        response_headers = [(name, value) for name, value in r.headers.items() if name.lower() not in excluded_headers]
        return Response(r.content, r.status_code, response_headers)
    except requests.RequestException as e:
        logging.error("PROXY ERROR %s %s %s -> %s", request.path, url, params or {}, e)
        return jsonify({"ok": False, "error": str(e), "url": url}), 502

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
    return send_from_directory(APP.static_folder, "overview.html") # เปลี่ยนเป็น overview.html

@APP.route("/monitor")
def monitor():
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
    return jsonify({
        "ok": all([DEF_ID, DEF_TK, OFF_ID, OFF_TK]),
        "api": API_BASE, "socket": SOCKET_URL,
        "def_set": bool(DEF_ID and DEF_TK), "off_set": bool(OFF_ID and OFF_TK),
        "routes": [str(r) for r in APP.url_map.iter_rules()]
    })

@APP.get("/host/metrics")
def host_metrics():
    cpu = None
    uptime_sec = None
    
    # Get CPU Temp (Linux/Pi specific)
    try:
        for p in ("/sys/class/thermal/thermal_zone0/temp",
                  "/sys/devices/virtual/thermal/thermal_zone0/temp"):
            if os.path.exists(p):
                with open(p) as f:
                    cpu = int(f.read().strip())/1000.0
                    break
    except Exception:
        cpu = None

    # Get Uptime (Linux specific)
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_sec = float(f.readline().split()[0])
    except Exception:
        uptime_sec = None

    return jsonify({"time": int(time.time()), "cpu_temp_c": cpu, "uptime_sec": uptime_sec})

# ---------- Config for frontend ----------
@APP.get("/config")
def config():
    return jsonify({
        "socket_url": SOCKET_URL,
        "defence": {"camera_id": DEF_ID, "set": bool(DEF_ID and DEF_TK)},
        "offence": {"camera_id": OFF_ID, "set": bool(OFF_ID and OFF_TK)},
        "rest": {
            "defence": {"info": "/api/defence/info", "recent": "/api/defence/recent"},
            "offence": {"info": "/api/offence/info", "recent": "/api/offence/recent"},
            # Image base URL จะอยู่ที่ root ของ API URL (ตัด '/api')
            "image_base": API_BASE.replace("/api", "") 
        }
    })

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
    return _proxy(_u("object-detection", DEF_ID), headers=_h(DEF_TK), params={"limit": limit})

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
    return _proxy(_u("object-detection", OFF_ID), headers=_h(OFF_TK), params={"limit": limit})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=PORT, debug=True)
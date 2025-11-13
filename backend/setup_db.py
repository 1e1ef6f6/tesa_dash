#!/usr/bin/env python3
"""
Database setup script for TESA Dashboard
Creates tables and indexes from schema.sql
"""
import os
import sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tesa_db")
DB_USER = os.getenv("DB_USER", "tesa_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def setup_database():
    """Create database schema from schema.sql"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        conn.autocommit = False
        
        # Read schema file
        schema_file = BASE_DIR / "schema.sql"
        if not schema_file.exists():
            print(f"Error: schema.sql not found at {schema_file}")
            sys.exit(1)
        
        with open(schema_file, "r") as f:
            schema_sql = f.read()
        
        # Execute schema
        cur = conn.cursor()
        cur.execute(schema_sql)
        conn.commit()
        cur.close()
        
        print("✓ Database schema created successfully!")
        print(f"  Database: {DB_NAME}")
        print(f"  User: {DB_USER}")
        print(f"  Host: {DB_HOST}:{DB_PORT}")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Setting up TESA Dashboard database...")
    setup_database()


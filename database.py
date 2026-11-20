import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "prediction_history.db"

def get_connection():
    """Get a connection to the SQLite database, ensuring directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database with the predictions table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            battery_id TEXT NOT NULL,
            soh REAL NOT NULL,
            condition TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            v_mean REAL NOT NULL,
            v_min REAL NOT NULL,
            t_mean REAL NOT NULL,
            t_max REAL NOT NULL,
            remaining_life TEXT
        )
    """)
    # Create indexes for fast querying by battery and date
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_battery ON predictions (battery_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions (timestamp)")
    conn.commit()
    conn.close()

def save_prediction(battery_id, soh, condition, cycle, v_mean, v_min, t_mean, t_max, remaining_life=None):
    """
    Save a validated prediction into the SQLite database.
    Returns the new record's ID.
    """
    clean_battery_id = str(battery_id).strip()
    if not clean_battery_id:
        raise ValueError("Battery ID cannot be empty.")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, battery_id, soh, condition, cycle,
            v_mean, v_min, t_mean, t_max, remaining_life
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        clean_battery_id,
        round(float(soh), 2),
        str(condition),
        int(cycle),
        round(float(v_mean), 3),
        round(float(v_min), 3),
        round(float(t_mean), 2),
        round(float(t_max), 2),
        str(remaining_life) if remaining_life else "N/A"
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_predictions(battery_id=None, start_date=None, end_date=None):
    """
    Fetch prediction records as a pandas DataFrame with optional filters.
    """
    conn = get_connection()
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if battery_id and battery_id != "All Batteries":
        query += " AND battery_id = ?"
        params.append(battery_id)

    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date} 00:00:00")

    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date} 23:59:59")

    query += " ORDER BY id DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_unique_batteries():
    """Return a list of unique battery IDs stored in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT battery_id FROM predictions ORDER BY battery_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [row["battery_id"] for row in rows]

def delete_prediction(record_id):
    """Delete a prediction record by its ID. Returns True if deleted."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (int(record_id),))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def get_battery_summary(battery_id=None):
    """
    Compute summary metrics for dashboard cards.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total counts
    cursor.execute("SELECT COUNT(*) as total_preds, COUNT(DISTINCT battery_id) as total_batteries FROM predictions")
    overall = cursor.fetchone()
    total_preds = overall["total_preds"]
    total_batteries = overall["total_batteries"]

    latest_soh = None
    soh_change = None
    earliest_date = None
    latest_date = None

    if battery_id and battery_id != "All Batteries":
        cursor.execute("""
            SELECT soh, timestamp FROM predictions 
            WHERE battery_id = ? 
            ORDER BY id DESC LIMIT 2
        """, (battery_id,))
        records = cursor.fetchall()
        if records:
            latest_soh = records[0]["soh"]
            latest_date = records[0]["timestamp"]
            if len(records) > 1:
                prev_soh = records[1]["soh"]
                soh_change = round(latest_soh - prev_soh, 2)
        
        cursor.execute("SELECT MIN(timestamp) as min_ts FROM predictions WHERE battery_id = ?", (battery_id,))
        min_rec = cursor.fetchone()
        if min_rec:
            earliest_date = min_rec["min_ts"]
    elif total_preds > 0:
        cursor.execute("SELECT soh, timestamp FROM predictions ORDER BY id DESC LIMIT 1")
        latest = cursor.fetchone()
        if latest:
            latest_soh = latest["soh"]
            latest_date = latest["timestamp"]

    conn.close()
    return {
        "total_preds": total_preds,
        "total_batteries": total_batteries,
        "latest_soh": latest_soh,
        "soh_change": soh_change,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
    }

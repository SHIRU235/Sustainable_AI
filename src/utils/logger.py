# src/utils/logger.py
import logging
import os
import sqlite3
from datetime import datetime

# ====== File Logging Setup ======

# Make logs folder
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with date
log_filename = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")

# Setup logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Create main logger
logger = logging.getLogger("SustainableAI")

# ====== Database Logging Setup ======

DB_PATH = os.path.join(LOG_DIR, "logs.db")

# Create table for structured logs
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prompt_text TEXT,
            input_params TEXT,
            energy_estimate REAL,
            recommendation TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_to_db(prompt_text: str, input_params: dict, energy_estimate: float, recommendation: str):
    """Log structured data to SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO logs (timestamp, prompt_text, input_params, energy_estimate, recommendation)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            prompt_text,
            str(input_params),  # store as string, or use JSON if needed
            energy_estimate,
            recommendation
        ))
        conn.commit()
        conn.close()
        logger.info("Logged data to DB successfully.")
    except Exception as e:
        logger.error(f"Failed to log data to DB: {e}")

# Example usage
if __name__ == "__main__":
    logger.info("Application started.")
    log_to_db(
        prompt_text="Summarize climate change report",
        input_params={"model": "T5", "tokens": 120},
        energy_estimate=2.34,
        recommendation="Use simplified summary to save energy."
    )
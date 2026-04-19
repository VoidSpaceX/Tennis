import sqlite3
from datetime import datetime

class Database:
    def __init__(self):
        self.conn = sqlite3.connect("calibration.db")
        self.create_table()
        
    def create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                point1_x REAL NOT NULL,
                point1_y REAL NOT NULL,
                point2_x REAL NOT NULL,
                point2_y REAL NOT NULL,
                real_size REAL NOT NULL,
                scale REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
        
    def save_calibration(self, name, point1_x, point1_y, point2_x, point2_y, real_size, scale):
        cursor = self.conn.execute(
            "INSERT INTO calibrations (name, point1_x, point1_y, point2_x, point2_y, real_size, scale) VALUES (?,?,?,?,?,?,?)",
            (name, point1_x, point1_y, point2_x, point2_y, real_size, scale)
        )
        self.conn.commit()
        return cursor.lastrowid
        
    def get_all_calibrations(self):
        cursor = self.conn.execute(
            """SELECT id, name, point1_x, point1_y, point2_x, point2_y, real_size, scale 
               FROM calibrations ORDER BY created_at DESC"""
        )
        return cursor.fetchall()

    def get_calibration(self, cal_id):
        cursor = self.conn.execute(
            "SELECT id, name, point1_x, point1_y, point2_x, point2_y, real_size, scale FROM calibrations WHERE id = ?",
            (cal_id,)
        )
        return cursor.fetchone()
    
    def close(self):
        self.conn.close()
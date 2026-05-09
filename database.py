import sqlite3
from datetime import datetime

class Database:
    def __init__(self):
        self.conn = sqlite3.connect("calibration.db")
        self.create_table()
        
    def create_table(self):
        """Создаёт таблицу 'calibrations', если она не существует."""
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
        """Сохраняет новую калибровку в базу данных."""
        cursor = self.conn.execute(
            "INSERT INTO calibrations (name, point1_x, point1_y, point2_x, point2_y, real_size, scale) VALUES (?,?,?,?,?,?,?)",
            (name, point1_x, point1_y, point2_x, point2_y, real_size, scale)
        )
        self.conn.commit()
        return cursor.lastrowid
        
    def get_all_calibrations(self):
        """Возвращает все калибровки из базы, отсортированные по времени создания (сначала новые)."""
        cursor = self.conn.execute(
            """SELECT * FROM calibrations ORDER BY created_at DESC"""
        )
        return cursor.fetchall()

    def get_calibration(self, cal_id):
        """Возвращает одну калибровку по её ID."""
        cursor = self.conn.execute(
            "SELECT id, name, point1_x, point1_y, point2_x, point2_y, real_size, scale FROM calibrations WHERE id = ?",
            (cal_id,)
        )
        return cursor.fetchone()
    
    def delete_calibration(self, cal_id):
        """Удаляет одну калибровку по ID."""
        self.conn.execute("DELETE FROM calibrations WHERE id = ?", (cal_id,))
        self.conn.commit()

    def delete_all_calibrations(self):
        """Удаляет ВСЕ калибровки из таблицы."""
        self.conn.execute("DELETE FROM calibrations")
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name='calibrations'")
        self.conn.commit()
    
    def close(self):
        """Закрывает соединение с базой данных."""
        self.conn.close()
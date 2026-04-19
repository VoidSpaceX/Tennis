import tkinter as tk
from tkinter import ttk, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2
import time

from calculation_speed import CalculationSpeed

class VideoApp(tk.Toplevel):
    def __init__(self, parent, video_source, scale):
        super().__init__(parent)
        self.parent = parent
        self.video_source = video_source
        self.scale = scale
        
        self.title("Анализ видео")
        self.geometry("1000x800")
        
        self.running = False
        self.paused = False
        self.canvas_image = None
        
        self.speed_tracer = CalculationSpeed(scale)
        self.SPEED_MIN_KMH = 20.0
        self.SPEED_MAX_KMH = 130.0
        
        self.model = None
        self.load_model()
        
        self.setup_ui()
        self.start_video()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def load_model(self):
        try:
            self.model = YOLO("tennis_model.pt")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
            self.destroy()
            
    def setup_ui(self):
        self.canvas = tk.Canvas(self,width=800, height=600, bg='black')
        self.canvas.pack(pady=10)
        
        speed_frame = ttk.LabelFrame(self, text="Текущая скорость", padding=10)
        speed_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.speed_label = ttk.Label(speed_frame, text="0.00 м/с (0.00 км/ч)", font=("Arial", 20, "bold"))
        self.speed_label.pack()
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)
        
        self.btn_pause = ttk.Button(btn_frame, text="Пауза", command=self.pause_video)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_resume = ttk.Button(btn_frame, text="Продолжить", command=self.resume_video, state=tk.DISABLED)
        self.btn_resume.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(btn_frame, text="Стоп", command=self.stop_video)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
    
    def start_video(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео/камеру")
            self.destroy()
            return
        
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()
        
    def process_video(self):
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                
                ball_center = self.detect_ball(frame)
                
                current_speed = 0.0
                if ball_center:
                    cx, cy = ball_center
                    self.speed_tracer.add_position(cx, cy)
                    current_speed = self.speed_tracer.get_speed()
                
                speed_kmh = current_speed * 3.6
                
                if ball_center and speed_kmh > self.SPEED_MIN_KMH and speed_kmh < self.SPEED_MAX_KMH:
                    cx, cy = ball_center
                    cv2.rectangle(frame, (cx-20, cy-20), (cx+20, cy+20), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Ball", (cx-20, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                self.update_display(frame, current_speed)
                
                time.sleep(0.033)
            else:
                time.sleep(0.05)
        
        self.cap.release()
        self.after(0, self.on_video_end)
    
    def detect_ball(self, frame):
        if self.model is None:
            return frame, None
        
        results = self.model(frame, verbose = False)
        ball_center = None
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])            
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                ball_center = (cx, cy)
                break
            if ball_center:
                break
        return ball_center
    
    def update_display(self, frame, speed):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((800, 600), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(img)
        
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        else:
            self.canvas.itemconfig(self.canvas_image, image=imgtk)
        self.canvas.image = imgtk
        
        speed_kmh = speed * 3.6
        if speed_kmh > self.SPEED_MIN_KMH and speed_kmh < self.SPEED_MAX_KMH:
            self.speed_label.config(text=f"{speed:.2f} м/с ({speed*3.6:.2f} км/ч)")
        
    def pause_video(self):
        self.paused = True
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL)
    
    def resume_video(self):
        self.paused = False
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
    
    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()
        
    def on_video_end(self):
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
    
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()
        self.parent.deiconify()
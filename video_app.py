import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2
import time
from collections import deque

from calculation_speed import CalculationSpeed
from speed_display import SpeedDisplayDialog

class VideoApp(tk.Toplevel):
    def __init__(self, parent, video_source, scale):
        super().__init__(parent)
        self.parent = parent
        self.video_source = video_source
        self.scale = scale
        
        self.title("Анализ видео")
        self.geometry("1000x800")
        
        self.current_frame_pos = 0
        self.total_frames = 0
        self.slider = None
                
        self.running = False
        self.paused = False
        self.canvas_image = None
        
        self.speed_tracer = CalculationSpeed(scale)
        self.SPEED_MIN_KMH = 30.0
        self.SPEED_MAX_KMH = 250.0
        self.DIRECTION_THRESHOLD = 2
        
        self.speed_display = None
        self.prev_ball_center = None 
        self.last_speed_player1 = 0.0 
        self.last_speed_player2 = 0.0 
        self.display_frame_counter = 0
        
        self.last_sent_speed = {1: -1.0, 2: -1.0}
        self.stroke_measure_active = {1: False, 2: False}
        self.stroke_measure_start = {1: 0.0, 2: 0.0}
        self.stroke_measure_max = {1: 0.0, 2: 0.0}
        self.measure_window = 0.3
        self.video_time = 0.0 
        self.stroke_detect_frames = 3
        self.stroke_detect_buffer = deque(maxlen=self.stroke_detect_frames)
        
        self.model = None
        self.load_model()
        
        self.setup_ui()
        self.start_video()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def load_model(self):
        """Загружает модель YOLO из файла 'tennis_model.pt'.
        В случае ошибки показывает сообщение и закрывает окно."""
        try:
            self.model = YOLO("tennis_model.pt")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
            self.destroy()
            
    def setup_ui(self):
        """Создаёт GUI"""
        self.canvas = tk.Canvas(self,width=800, height=600, bg='black')
        self.canvas.pack(pady=10)
        
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.time_label = ttk.Label(control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=(0,10))
        
        self.video_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                             length=400, command=self.on_slider_move)
        self.video_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.video_slider.config(state=tk.DISABLED)
        
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
        
        self.btn_show_score = ttk.Button(btn_frame, text="📺 Отобразить скорость", command=self.open_display_settings)
        self.btn_show_score.pack(side=tk.LEFT, padx=5)
    
    def start_video(self):
        """Открывает видео/камеру и запускает поток обработки кадров (process_video).
        Если источник не открывается, показывает ошибку и закрывает окно."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео/камеру")
            self.destroy()
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames > 0:
            self.video_slider.config(state=tk.NORMAL, to=self.total_frames - 1)
            total_seconds = self.total_frames / self.fps
            self.time_label.config(text=f"00:00 / {self.format_time(total_seconds)}")
        else:
            self.video_slider.config(state=tk.DISABLED)
            self.time_label.config(text="Время недоступно для камеры")
            
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()
        
    def process_video(self):
        """Основной цикл обработки видео (выполняется в отдельном потоке)."""
        fps = self.fps if self.fps > 0 else 30
        frame_time = 1.0 / fps
        last_time = time.time()
        
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                
                self.current_frame_pos += 1
                self.video_time += frame_time
                
                if self.current_frame_pos % 3 == 0:
                    self.after(0, self.update_slider_and_time, self.current_frame_pos)
                
                ball_center = self.detect_ball(frame)
                
                speed_kmh = 0.0
                current_speed = 0.0
                if ball_center:
                    cx, cy = ball_center
                    if self.prev_ball_center is not None:
                        px, py = self.prev_ball_center
                        dx = cx - px
                        
                        self.stroke_detect_buffer.append(dx)

                        if len(self.stroke_detect_buffer) == self.stroke_detect_frames:
                            signs = [1 if d > self.DIRECTION_THRESHOLD else
                                    -1 if d < -self.DIRECTION_THRESHOLD else 0
                                    for d in self.stroke_detect_buffer]
                            
                            if (signs[0] != 0 and signs[0] != signs[-1] and signs[-1] != 0) or \
                            (all(s == 0 for s in signs[:-1]) and signs[-1] != 0):
                                if signs[-1] > 0:
                                    player = 1
                                else:
                                    player = 2

                                self.speed_tracer.reset()
                                
                                self.stroke_measure_active[player] = True
                                self.stroke_measure_start[player] = self.video_time
                                self.stroke_measure_max[player] = 0.0

                                other = 3 - player
                                self.stroke_measure_active[other] = False

                        if abs(dx) > self.DIRECTION_THRESHOLD:
                            self.speed_tracer.add_position(cx, cy)
                            current_speed = self.speed_tracer.get_speed()
                            speed_kmh = current_speed * 3.6

                    self.prev_ball_center = (cx, cy)
                else:
                    for p in [1, 2]:
                        if self.stroke_measure_active[p]:
                            if self.stroke_measure_max[p] > 0:
                                if p == 1:
                                    self.last_speed_player1 = self.stroke_measure_max[p]
                                else:
                                    self.last_speed_player2 = self.stroke_measure_max[p]
                            self.stroke_measure_active[p] = False
                    self.prev_ball_center = None
                    self.stroke_detect_buffer.clear()
                    self.speed_tracer.reset()

                for p in [1, 2]:
                    if self.stroke_measure_active[p] and current_speed > 0:
                        if current_speed > self.stroke_measure_max[p]:
                            self.stroke_measure_max[p] = current_speed
                        if self.video_time - self.stroke_measure_start[p] >= self.measure_window:
                            if self.stroke_measure_max[p] > 0:
                                if p == 1:
                                    self.last_speed_player1 = self.stroke_measure_max[p]
                                else:
                                    self.last_speed_player2 = self.stroke_measure_max[p]
                            self.stroke_measure_active[p] = False

                self.display_frame_counter += 1
                if self.display_frame_counter % 10 == 0:
                    self.after(0, self.send_speed_to_display)
                   
                if ball_center and self.SPEED_MIN_KMH <= speed_kmh <= self.SPEED_MAX_KMH:
                    cx, cy = ball_center
                    cv2.rectangle(frame, (cx-20, cy-20), (cx+20, cy+20), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Ball", (cx-20, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                self.update_display(frame, current_speed)
                
                elapsed = time.time() - last_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
                
            else:
                time.sleep(0.05)
        
        self.cap.release()
        self.after(0, self.on_video_end)
    
    def detect_ball(self, frame):
        """Детектирует мяч на кадре с помощью модели YOLO."""
        if self.model is None:
            return None
        
        results = self.model(frame, verbose=False, device="cuda")
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
        """Обновляет canvas (отображает кадр) и метку текущей скорости в GUI"""
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
    
    def send_speed_to_display(self):
        """Передаёт скорости обоих игроков на табло."""
        if not (self.speed_display and self.speed_display.winfo_exists()):
            return
        # Игрок 1
        if self.last_speed_player1 != self.last_sent_speed[1]:
            self.speed_display.update_speed(0, self.last_speed_player1)
            self.last_sent_speed[1] = self.last_speed_player1
        # Игрок 2
        if self.last_speed_player2 != self.last_sent_speed[2]:
            self.speed_display.update_speed(1, self.last_speed_player2)
            self.last_sent_speed[2] = self.last_speed_player2
      
    def pause_video(self):
        """Приостанавливает воспроизведение видео."""
        self.paused = True
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL)
    
    def resume_video(self):
        """Возобновляет воспроизведение видео."""
        self.paused = False
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
    
    def stop_video(self):
        """Останавливает обработку видео и закрывает окно."""
        self.running = False
        if self.speed_display:
            self.speed_display.close()
        if self.cap:
            self.cap.release()
        self.destroy()
        
    def on_video_end(self):
        """Вызывается при завершении видео (не при ручной остановке) — отключает кнопки."""
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
    
    def on_close(self):
        """Обработчик закрытия окна (крестик или системный вызов).
        Останавливает поток, освобождает камеру и возвращает родительское окно."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()
        self.parent.deiconify()
    
    def format_time(self, seconds):
        """Преобразует секунды в строку MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def update_slider_and_time(self, frame_number):
        """Обновляет положение ползунка и метку времени в интерфейсе."""
        if not getattr(self, 'slider_is_moving', False):
            self.video_slider.set(frame_number)
        
        if self.total_frames > 0:
            current_seconds = frame_number / self.fps
            total_seconds = self.total_frames / self.fps
            self.time_label.config(text=f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}")
    
    def on_slider_move(self, value):
        """Обрабатывает ручную перемотку видео ползунком."""
        if self.cap is None:
            return
        self.slider_is_moving = True
        
        target_frame = int(float(value))
        if self.total_frames > 0 and 0 <= target_frame < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame_pos = target_frame

            current_seconds = target_frame / self.fps
            total_seconds = self.total_frames / self.fps
            self.time_label.config(text=f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}")
        
        self.after(100, lambda: setattr(self, 'slider_is_moving', False))
     
    def open_display_settings(self):
        """Диалог выбора количества игроков и типа отображения."""
        dialog = tk.Toplevel(self)
        dialog.title("Настройка отображения скорости")
        dialog.geometry("400x350")
        dialog.resizable(False, False)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Количество игроков (1 или 2):", font=("Arial", 10, "bold")).pack(pady=(10,5))
        num_players_entry = ttk.Entry(dialog, width=5)
        num_players_entry.pack(pady=(0,10))
        
        ttk.Label(dialog, text="Куда выводить скорость:", font=("Arial", 10, "bold")).pack(pady=(10,5))
        display_target_var = tk.StringVar()
        display_combo = ttk.Combobox(dialog, textvariable=display_target_var, values=[
            "На весь экран",
            "На внешнее табло"
        ], state="readonly", width=40)
        display_combo.current(0)
        display_combo.pack(pady=(0,10))
        
        # Кнопка подтверждения
        def on_confirm():
            try:
                num_players = int(num_players_entry.get().strip())
                if num_players not in (1, 2):
                    raise ValueError
            except:
                messagebox.showerror("Ошибка", "Введите 1 или 2")
                return
            
            target = display_target_var.get()
            if target.startswith("На весь экран"):
                self.create_fullscreen_display(num_players)
            else:
                messagebox.showinfo("Предупреждение", "Не реализована")
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=on_confirm).pack(pady=20)

    def create_fullscreen_display(self, num_players):
        """Создаёт полноэкранное окно отображения скорости."""
        if self.speed_display:
            self.speed_display.close()
        self.speed_display = SpeedDisplayDialog(self,num_players)
        self.speed_display.attributes('-fullscreen', True)
        self.speed_display.lift()
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class PointSelectorDialog(tk.Toplevel):
    def __init__(self, parent, frame):
        super().__init__(parent)
        self.parent = parent
        self.original_frame = frame.copy()
        self.title("Выбор точек для калибровки")
        self.geometry("900x700")
        
        self.points = [] 
        self.pixel_distance = None
        self.real_distance = None
        self.scale = None
        
        self.setup_ui()
        self.display_frame()
        
    def setup_ui(self):
        self.canvas = tk.Canvas(self, bg='gray', cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.label_info = ttk.Label(info_frame, text="🔴 Кликните на ПЕРВУЮ точку", font=("Arial", 11))
        self.label_info.pack()
        
        self.points_label = ttk.Label(info_frame, text="Точек выбрано: 0/2", font=("Arial", 10))
        self.points_label.pack(pady=5)
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        self.btn_ok = ttk.Button(btn_frame, text="✅ Рассчитать масштаб", command=self.calculate_scale, state=tk.DISABLED)
        self.btn_ok.pack(side=tk.LEFT, padx=5, ipady=5)
        
        ttk.Button(btn_frame, text="❌ Отмена", command=self.destroy).pack(side=tk.LEFT, padx=5, ipady=5)
        ttk.Button(btn_frame, text="🔄 Сбросить точки", command=self.reset_points).pack(side=tk.LEFT, padx=5, ipady=5)
    
    def display_frame(self):
        frame_rgb = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 100 else 800
        canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 100 else 600
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def on_click(self, event):
        if len(self.points) < 2:
            x, y = event.x, event.y
            self.points.append((x, y))
        
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="white", width=2)
            self.canvas.create_text(x+15, y-15, text=str(len(self.points)), fill="red", font=("Arial", 12, "bold"))
            
            if len(self.points) == 2:
                p1, p2 = self.points
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="yellow", width=3, dash=(5,5))
                self.pixel_distance = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                self.label_info.config(text=f"✅ Расстояние между точками: {self.pixel_distance:.1f} пикселей")
                self.points_label.config(text=f"Точек выбрано: 2/2 (расстояние: {self.pixel_distance:.1f} px)")
                self.btn_ok.config(state=tk.NORMAL)
            else:
                self.label_info.config(text="🔵 Кликните на ВТОРУЮ точку")
                self.points_label.config(text="Точек выбрано: 1/2")
    
    def reset_points(self):
        self.points = []
        self.pixel_distance = None
        self.btn_ok.config(state=tk.DISABLED)
        self.label_info.config(text="🔴 Кликните на ПЕРВУЮ точку")
        self.points_label.config(text="Точек выбрано: 0/2")
        self.display_frame()
    
    def calculate_scale(self):
        if not self.pixel_distance:
            messagebox.showerror("Ошибка", "Сначала выберите две точки")
            return
        
        real_dist = simpledialog.askfloat(
            "Реальное расстояние",
            "Введите реальное расстояние между точками (в метрах):",
            parent=self
        )
        
        if real_dist and real_dist > 0:
            self.real_distance = real_dist
            self.scale = real_dist / self.pixel_distance
            messagebox.showinfo("Масштаб", f"Масштаб рассчитан:\n{self.scale:.6f} м/пиксель")
            self.destroy()
        else:
            messagebox.showerror("Ошибка", "Некорректное расстояние")
    
    def get_calibration_data(self):
        return {
            'points': self.points,
            'pixel_distance': self.pixel_distance,
            'real_distance': self.real_distance,
            'scale': self.scale
        }
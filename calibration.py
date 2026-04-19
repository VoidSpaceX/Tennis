import tkinter as tk
from tkinter import ttk, messagebox
import cv2

from database import Database
from point_selector import PointSelectorDialog
from video_app import VideoApp

class CalibrationWindow(tk.Toplevel):
    def __init__(self, parent, first_frame, video_source):
        super().__init__(parent)
        self.parent = parent
        self.first_frame = first_frame
        self.video_source = video_source
        
        
        self.title("Калибровка")
        self.geometry("800x700")
        
        self.db = Database()
        
        self.edit_mode = False
        self.selected_calib_id = None
        self.scale = None
        self.pixel_distance = None
        self.real_distance = None
        self.point1 = None
        self.point2 = None
        
        self.name_var = tk.StringVar()
        self.real_size_var = tk.StringVar()
        self.coefficient_var = tk.StringVar()
        self.point1_var = tk.StringVar()
        self.point2_var = tk.StringVar()
        
        self.setup_ui()
        self.load_calibration()
        
        self.combobox.bind("<<ComboboxSelected>>", self.on_calibration_selected)
        
    def setup_ui(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        calib_name = ttk.Labelframe(
            main_frame,
            text="Сохранённые калибровки"
        )
        calib_name.pack(fill=tk.X, pady=(0, 20))
        
        self.combobox = ttk.Combobox(calib_name, width=30, state="readonly")
        self.combobox.pack(side=tk.LEFT)
        
        frame_vidget = ttk.Labelframe(
            main_frame,
            text="Параметры калибровки",
            padding="15"
        )
        frame_vidget.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        frame_vidget.columnconfigure(1, weight=1)
        
        ttk.Label(frame_vidget,text="Название:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=10, padx=(0, 10))
        self.name_entry = ttk.Entry(frame_vidget, width=40, textvariable=self.name_var, state="readonly")
        self.name_entry.grid(row=0, column=1, sticky="ew", pady=10)
        
        ttk.Label(frame_vidget, text="Реальный размер (м):", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=10, padx=(0, 10))
        self.real_size_entry = ttk.Entry(frame_vidget, width=20, textvariable=self.real_size_var, state="readonly")
        self.real_size_entry.grid(row=1, column=1, sticky="w", pady=10)
        
        ttk.Label(frame_vidget, text="Коэффициент (м/пикс):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=10, padx=(0, 10))
        self.coefficient_entry = ttk.Entry(frame_vidget, width=20, textvariable=self.coefficient_var, state="readonly")
        self.coefficient_entry.grid(row=2, column=1, sticky="w", pady=10)
        
        # Точка 1
        ttk.Label(frame_vidget,text="Точка 1 (x,y)", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=10, padx=(0, 10))
        self.point1_entry = ttk.Entry(frame_vidget, width=20, textvariable=self.point1_var, state="readonly")
        self.point1_entry.grid(row=3, column=1, sticky="w", pady=10)
        
        # Точка 2
        ttk.Label(frame_vidget,text="Точка 2 (x,y)", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w", pady=10, padx=(0, 10))
        self.point2_entry = ttk.Entry(frame_vidget, width=20, textvariable=self.point2_var, state="readonly")
        self.point2_entry.grid(row=4, column=1, sticky="w", pady=10)
        
        self.btn_select_points = ttk.Button(frame_vidget, text="Выбрать две точки на кадре", command=self.select_points, state=tk.DISABLED)
        self.btn_select_points.grid(row=5, column=0, columnspan=2, pady=15)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.BOTH, pady=(10, 0))
        
        self.btn_use = ttk.Button(button_frame, text="Использовать выбранную", command=self.use_selected)
        self.btn_use.pack(side=tk.LEFT, padx=5, ipady=15,pady=30)
        
        self.btn_new = ttk.Button(button_frame, text="Создать новую", command=self.create_new)
        self.btn_new.pack(side=tk.LEFT, padx=5, ipady=15, pady=30)
        
        self.btn_save = ttk.Button(button_frame, text="Сохранить", command=self.save_calibration, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5, ipady=8)

        self.btn_cancel = ttk.Button(button_frame, text="Отмена", command=self.cancel_edit, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=5, ipady=8)
            
    def load_calibration(self):
        calibrations = self.db.get_all_calibrations()
        if calibrations:
            calib_list = [f"ID {cal[0]}: {cal[1]}" for cal in calibrations]
            self.combobox['values'] = calib_list
            self.calibrations_data = {f"ID {cal[0]}: {cal[1]}": cal for cal in calibrations}
        else:
            self.combobox['values'] = ["Нет сохранённых калибровок"]
            self.calibrations_data = {}
            
    def on_calibration_selected(self, event):
        selection = self.combobox.get()
        if selection in self.calibrations_data:
            cal = self.calibrations_data[selection]
            self.selected_calib_id = cal[0]
            self.name_var.set(cal[1])
            self.real_size_var.set(str(cal[6]) if cal[6] else "")
            self.coefficient_var.set(f"{cal[7]:.6f}")
            self.point1_var.set(f"({cal[2]}, {cal[3]})" if cal[2] else "")
            self.point2_var.set(f"({cal[4]}, {cal[5]})" if cal[4] else "")
            self.scale = cal[7]
            self.btn_use.config(state=tk.NORMAL)
            
    def create_new(self):
        self.edit_mode = True
        
        self.name_var.set("")
        self.real_size_var.set("")
        self.coefficient_var.set("")
        self.point1_var.set("")
        self.point2_var.set("")
        self.pixel_distance = None
        self.real_distance = None
        self.point1 = None
        self.point2 = None
        self.scale = None
        
        self.name_entry.config(state="normal")
        
        self.btn_select_points.config(state=tk.NORMAL)
        self.btn_new.config(state=tk.DISABLED)
        self.btn_use.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.NORMAL)
    
        self.combobox.config(state=tk.DISABLED)
        
        self.name_entry.focus()

    def select_points(self):
        if not self.edit_mode:
            messagebox.showwarning("Предупреждение", "Сначала нажмите 'Создать новую'")
            return
        selector = PointSelectorDialog(self, self.first_frame)
        self.wait_window(selector)
        
        if selector.scale is not None:
            self.point1 = selector.points[0] if len(selector.points) > 0 else None
            self.point2 = selector.points[1] if len(selector.points) > 1 else None
            self.pixel_distance = selector.pixel_distance
            self.real_distance = selector.real_distance
            self.scale = selector.scale
            
            self.point1_var.set(f"({self.point1[0]}, {self.point1[1]})" if self.point1 else "")
            self.point2_var.set(f"({self.point2[0]}, {self.point2[1]})" if self.point2 else "")
            self.coefficient_var.set(f"{self.scale:.6f}")
            self.real_size_var.set(str(self.real_distance) if self.real_distance else "")
        else:
            messagebox.showwarning("Предупреждение", "Калибровка не выполнена")   
    
    def save_calibration(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Введите название калибровки")
            return
        if not self.pixel_distance or not self.real_distance:
            messagebox.showerror("Ошибка", "Сначала выберите две точки и укажите реальное расстояние")
            return
        if not self.point1 or not self.point2:
            messagebox.showerror("Ошибка", "Не выбраны точки")
            return
        
        try:
            cal_id = self.db.save_calibration(
                name=name,
                point1_x=self.point1[0], point1_y=self.point1[1],
                point2_x=self.point2[0], point2_y=self.point2[1],
                real_size=self.real_distance,
                scale=self.scale
            )
            if cal_id:
                messagebox.showinfo("Успех", f"Калибровка '{name}' сохранена!")
                self.load_calibration()
                self.cancel_edit()
                self.combobox.set(f"ID {cal_id}: {name}")
                self.scale = self.scale
                self.go_to_video()
            else:
                messagebox.showerror("Ошибка", "Не удалось сохранить")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {e}")
            
    def cancel_edit(self):
        self.edit_mode = False
        
        self.name_entry.config(state="readonly")
        self.real_size_entry.config(state="readonly")
        
        self.btn_select_points.config(state=tk.DISABLED)
        self.btn_new.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)
        self.btn_use.config(state=tk.NORMAL)
        
        self.combobox.config(state="readonly")
        
        if not self.combobox.get():
            self.name_var.set("")
            self.real_size_var.set("")
            self.coefficient_var.set("")
            self.point1_var.set("")
            self.point2_var.set("")
            self.btn_use.config(state=tk.DISABLED)
        else:
            self.on_calibration_selected(None)
        
    def use_selected(self):
        if self.scale:
            self.go_to_video()
        else:
            messagebox.showerror("Ошибка", "Калибровка не выбрана")

    def go_to_video(self):
        self.destroy()
        self.parent.deiconify() 
        VideoApp(self.parent, self.video_source, self.scale)
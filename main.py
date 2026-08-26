import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os

from calibration import CalibrationWindow

class TennisApp:
    def __init__(self,root):
        self.root = root
        self.root.title("TennisBall Speed Tracker")
        self.root.geometry("500x300")
        
        self.video_source = None
        
        self.source()
        self.dialog_info()

    def source(self):
        """Создаёт интерфейс выбора источника видео."""
        ins_frame = ttk.Frame(self.root)
        ins_frame.pack(anchor="w")
        ins_label = ttk.Label(ins_frame, text="Инструкция", cursor="hand2", foreground="#0645AD", font=("Arial", 10, "underline"), padding=5)
        ins_label.bind("<Button-1>", self.label_clicked)
        ins_label.pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(anchor="center",expand=True)
        
        self.status = ttk.Label(
            btn_frame,
            text="Выберите источник",
            font=25
        )
        self.status.pack(pady=10)
        
        btn_video = ttk.Button(
            btn_frame,
            text="📁 Загрузить видео", 
            command=self.load_video,
            width=25
        )
        btn_video.pack(side=tk.LEFT, ipady=20, padx=10,expand=True)

        btn_camera = ttk.Button(
            btn_frame,
            text="📷 Веб-камера", 
            command=self.start_camera,
            width=25
        )
        btn_camera.pack(side=tk.LEFT, ipady=20, padx=10,expand=True)
        
    
    def load_video(self):
        """Открывает диалог выбора видеофайла и запускает калибровку."""
        try:
            path = filedialog.askopenfilename(filetypes=[("Video","*.mp4 *.avi *.mov")])
            if path:
                valid_extensions = ('.mp4', '.avi', '.mov')
                file_ext = os.path.splitext(path)[1].lower()
                
                if file_ext not in valid_extensions:
                    messagebox.showerror("Ошибка", 
                        f"Формат файла '{file_ext}' не поддерживается.\n"
                        f"Поддерживаемые форматы: .mp4, .avi, .mov")
                    return
                
                self.video_source = path
                self.open_calibration_window()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить видео: {e}")
            
    def start_camera(self):
        """Пытается открыть веб-камеру (индекс 0). При успехе запускает калибровку."""
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self.video_source = 0
            cap.release()
            self.open_calibration_window()
        else:
            messagebox.showerror("Ошибка", f"Не удалось открыть камеру")
    
    def open_calibration_window(self):
        """Читает первый кадр из источника и открывает окно калибровки."""
        cap = cv2.VideoCapture(self.video_source)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.root.withdraw()
            CalibrationWindow(self.root, frame, self.video_source)
        else:
            messagebox.showerror("Ошибка", "Не удалось получить кадр для калибровки")
    
    def label_clicked(self, event):
        """Открывает файл инструкции (pdf)."""
        try:
            os.startfile("Инструкция.pdf")
        except FileNotFoundError:
            messagebox.showerror("Ошибка", "Файл 'Инструкция.pdf' не найден в папке программы.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл инструкции:\n{e}")
    
    def dialog_info(self):
        messagebox.showinfo("Предупреждение","Внимание! При запуске программы камера активируется исключительно для измерения скорости теннисного мяча. В поле зрения камеры может оказаться лицо игрока. Важно: видео не записывается, не сохраняется и не передаётся никуда — данные используются только для мгновенного расчёта скорости и сразу удаляются.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TennisApp(root)
    root.mainloop()

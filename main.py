import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

from calibration import CalibrationWindow

class TennisApp:
    def __init__(self,root):
        self.root = root
        self.root.title("Tennis Speed Track")
        self.root.geometry("500x300")
        
        self.video_source = None
        self.source()
        
    def source(self):
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
        path = filedialog.askopenfilename(filetypes=[("Video","*.mp4 *.avi *.mov")])
        if path:
            self.video_source = path
            self.open_calibration_window()
            
    def start_camera(self):
        pass
    
    def open_calibration_window(self):
        cap = cv2.VideoCapture(self.video_source)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.root.withdraw()
            calib_window = CalibrationWindow(self.root, frame, self.video_source)
        else:
            messagebox.showerror("Ошибка", "Не удалось получить кадр для калибровки")

if __name__ == "__main__":
    root = tk.Tk()
    app = TennisApp(root)
    root.mainloop()
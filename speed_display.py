import tkinter as tk
from tkinter import ttk, messagebox

class SpeedDisplayDialog(tk.Toplevel):
    def __init__(self, parent, num_players=1):
        super().__init__(parent)
        self.num_players = num_players
        self.title("Табло скорости")
        self.configure(bg='black')
        self.parent_app = parent
    
        main_frame = tk.Frame(self, bg='black')
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.speed_kmh_vars = [tk.StringVar(value="0.0") for _ in range(2)]

        if num_players == 1:
            main_frame.grid_rowconfigure(0, weight=1)
            main_frame.grid_columnconfigure(0, weight=1)
            lbl = tk.Label(main_frame, textvariable=self.speed_kmh_vars[0],
                           font=("Arial", 120, "bold"), fg="#00FF00", bg="black")
            lbl.grid(row=0, column=0)
            unit = tk.Label(main_frame, text="км/ч", font=("Arial", 40, "bold"),
                            fg="white", bg="black")
            unit.grid(row=1, column=0, pady=(0, 60))
        else:
            for i in range(2):
                main_frame.grid_rowconfigure(0, weight=1)
                main_frame.grid_rowconfigure(1, weight=1)
                main_frame.grid_columnconfigure(i, weight=1, uniform="col")

                header = tk.Label(main_frame, text=f"Игрок {i+1}",
                                  font=("Arial", 40, "bold"), fg="white", bg="black")
                header.grid(row=0, column=i, pady=(60, 20))
                lbl = tk.Label(main_frame, textvariable=self.speed_kmh_vars[i],
                               font=("Arial", 100, "bold"), fg="#00FF00", bg="black")
                lbl.grid(row=1, column=i)
                unit = tk.Label(main_frame, text="км/ч", font=("Arial", 30, "bold"),
                                fg="white", bg="black")
                unit.grid(row=2, column=i, pady=(0, 60))

        self.bind('<Escape>', lambda e: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)

    def update_speed(self, player_index, speed_ms):
        """Обновляет скорость конкретного игрока (м/с) — отображается в км/ч."""
        if 0 <= player_index < 2:
            kmh = speed_ms * 3.6
            self.speed_kmh_vars[player_index].set(f"{kmh:.1f}")

    def close(self):
        if self.parent_app and hasattr(self.parent_app, 'speed_display'):
            self.parent_app.speed_display = None
        self.destroy()
        
        
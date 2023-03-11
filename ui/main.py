import tkinter as tk
from tkinter import ttk, filedialog
from ui.video import VideoUI
import os
import json

class MainUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Intelligent Port Monitoring")

        # add label for file selection
        self.label = tk.Label(self.master, text="Analyze a video file (mp4)", font=("Arial", 12))
        self.label.grid(row=0, column=0, padx=10, pady=10)

        # add button for file selection
        self.button = tk.Button(self.master, text="Select File", command=self.analyze_video)
        self.button.grid(row=1, column=0, padx=10, pady=5)


        # add label for stream webcam
        self.label = tk.Label(self.master, text="Stream IP Camera", font=("Arial", 12))
        self.label.grid(row=0, column=1, padx=10, pady=10)


        self.button = tk.Button(self.master, text="Stream IP Camera", command=self.stream_camera)
        self.button.grid(row=0, column=3, padx=10, pady=5)

        # add separator
        ttk.Separator(self.master, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=3, sticky="we", padx=20, pady=10)

        # add label for line input
        self.line_label = tk.Label(self.master, text="Enter line start and end point:", font=("Arial", 12))
        self.line_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        # add entry fields for line dimensions

        # add labels and entry fields for line dimensions
        self.x_start_label = tk.Label(self.master, text="X start:")
        self.x_start_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.x_start_entry = tk.Entry(self.master, width=10)
        self.x_start_entry.grid(row=5, column=1, padx=10, pady=5,sticky="w")
        self.x_start_entry.insert(0,500)

        self.y_start_label = tk.Label(self.master, text="Y start:")
        self.y_start_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.y_start_entry = tk.Entry(self.master, width=10)
        self.y_start_entry.grid(row=6, column=1, padx=10, pady=5,sticky="w")
        self.y_start_entry.insert(0,0)

        self.x_end_label = tk.Label(self.master, text="X end:")
        self.x_end_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.x_end_entry = tk.Entry(self.master, width=10)
        self.x_end_entry.grid(row=7, column=1, padx=10, pady=5,sticky="w")
        self.x_end_entry.insert(0,500)

        self.y_end_label = tk.Label(self.master, text="Y end:")
        self.y_end_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.y_end_entry = tk.Entry(self.master, width=10)
        self.y_end_entry.grid(row=8, column=1, padx=10, pady=5,sticky="w")
        self.y_end_entry.insert(0,1000)

        # add label for orientation selection
        self.orientation_label = tk.Label(self.master, text="Select line orientation:", font=("Arial", 12))
        self.orientation_label.grid(row=4, column=2, padx=10, pady=10, sticky="w")

        # add radio button for orientation selection
        self.orientation_var = tk.StringVar()
        self.orientation_var.set("left") # set default value to left
        self.left_radio = tk.Radiobutton(self.master, text="Left", variable=self.orientation_var, value="left")
        self.left_radio.grid(row=5, column=2, padx=10, pady=5, sticky="w")
        self.right_radio = tk.Radiobutton(self.master, text="Right", variable=self.orientation_var, value="right")
        self.right_radio.grid(row=5, column=2, padx=10, pady=5, sticky="e")
        self.bottom_radio = tk.Radiobutton(self.master, text="Bottom", variable=self.orientation_var, value="bottom")
        self.bottom_radio.grid(row=6, column=2, padx=10, pady=5, sticky="w")
        self.above_radio = tk.Radiobutton(self.master, text="Above", variable=self.orientation_var, value="above")
        self.above_radio.grid(row=6, column=2, padx=10, pady=5, sticky="e")

        login_json = 'login.json'
        upass = ''
        ip_address = ''
        if os.path.exists(login_json):
            with open('login.json', 'r') as f:
                data = json.load(f)
                upass = data['upass']
                ip_address = data['ip_address']

        # add labels and entry fields for username, password, and IP address
        self.password_label = tk.Label(self.master, text="Username Password:")
        self.password_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.password_entry = tk.Entry(self.master, width=20, show="*")
        self.password_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.password_entry.insert(0, upass)

        self.ipaddress_label = tk.Label(self.master, text="IP Address:")
        self.ipaddress_label.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.ipaddress_entry = tk.Entry(self.master, width=20)
        self.ipaddress_entry.grid(row=1, column=4, padx=10, pady=5, sticky="w")
        self.ipaddress_entry.insert(0, ip_address)


        print("UI Ready")

    def analyze_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.mp4")])
        print(f"Selected file: {file_path}")

        line_roi = (int(self.x_start_entry.get()), int(self.y_start_entry.get()), int(self.x_end_entry.get()), int(self.y_end_entry.get()))

        # hide the main window
        self.master.withdraw()

        # pass the file path, line dimensions, and orientation to VideoUI
        VideoUI(self.master, source=file_path, in_orientation=self.orientation_var.get(), line_roi=line_roi, is_video_player=True)

    def stream_camera(self):
    

        webcam = 'rtsp://' + self.password_entry.get() + '@' + self.ipaddress_entry.get()
        line_roi = (int(self.x_start_entry.get()), int(self.y_start_entry.get()), int(self.x_end_entry.get()), int(self.y_end_entry.get()))

        # hide the main window
        self.master.withdraw()

        # pass the file path, line dimensions, and orientation to VideoUI
        VideoUI(self.master, source=webcam, in_orientation=self.orientation_var.get(), line_roi=line_roi, is_video_player=False)

        
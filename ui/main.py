import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from ui import VideoUI
import cv2

class MainUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Main Window")

        self.label = tk.Label(self.master, text="Please select a video file (mp4)", font=("Arial", 12))
        self.label.pack(pady=10)

        self.button = tk.Button(self.master, text="Select File", command=self.select_file)
        self.button.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.mp4")])
        print(f"Selected file: {file_path}")
        self.master.withdraw()  # hide the main window
        VideoUI(self.master, file_path)


import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2

class MainWindow:
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
        SecondWindow(self.master, file_path)



class SecondWindow:
    def __init__(self, master, file_path):
        self.master = master
        self.second_window = tk.Toplevel(master)
        self.master.title('Video Player')
        self.processor = VideoProcessor(file_path)
        self.second_window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.delay = int(1000 / self.processor.fps)
        self.paused = False
        self.seek_timestamp = None
        self.width = self.processor.width
        self.canvas = tk.Canvas(self.second_window, width=self.processor.width, height=self.processor.height)
        self.canvas.pack()

        self.canvas.bind("<Configure>", self.resize)
          
        # Add buttons for pause, seek forward, and seek backward

        self.seek_slider = tk.Scale(self.second_window, from_=0, to=self.processor.get_duration(), orient=tk.HORIZONTAL, length=self.width, command=self.seek_to_timestamp)
        self.seek_slider.pack()

        self.pause_button = tk.Button(self.second_window, text="Pause/Play", command=self.toggle_pause)
        self.pause_button.pack(side="left")

        self.seek_forward_button = tk.Button(self.second_window, text="Seek Forward", command=self.seek_forward)
        self.seek_forward_button.pack(side="left")

        self.seek_backward_button = tk.Button(self.second_window, text="Seek Backward", command=self.seek_backward)
        self.seek_backward_button.pack(side="left")

        self.sample_output = tk.Text(self.second_window, width=40, height=10)
        self.sample_output.insert(tk.END, "lorem ipsum dolor sit amet")
        self.sample_output.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.update_canvas()

    def resize(self, event):
        # Resize image to fit the window and update the canvas
        self.update_canvas()

    def update_canvas(self):
        if self.seek_timestamp is not None:
            self.processor.seek_to_timestamp(self.seek_timestamp)
            self.seek_timestamp = None
            if self.paused:
                self.current_frame = self.processor.process_frame()
                if self.current_frame is not None:
                    img = Image.fromarray(self.current_frame)
                    img = ImageTk.PhotoImage(image=img)
                    self.canvas.img = img
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                    
                    # Update slider bar position
                    #current_timestamp = self.processor.get_current_timestamp()
                    #self.seek_slider.set(current_timestamp)

        if not self.paused:
            self.current_frame = self.processor.process_frame()
            if self.current_frame is not None:
                img = Image.fromarray(self.current_frame)
                img = ImageTk.PhotoImage(image=img)
                self.canvas.img = img
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

        self.master.after(self.delay, self.update_canvas)
    
    def toggle_pause(self):
        self.paused = not self.paused
    
    def seek_forward(self):
        current_timestamp = self.processor.get_current_timestamp()
        new_timestamp = current_timestamp + 3
        self.seek_timestamp = new_timestamp
        self.seek_slider.set(current_timestamp)

    def seek_backward(self):
        current_timestamp = self.processor.get_current_timestamp()
        new_timestamp = current_timestamp - 3
        self.seek_timestamp = new_timestamp
        self.seek_slider.set(current_timestamp)

    def seek_to_timestamp(self, timestamp):
        self.processor.seek_to_timestamp(float(timestamp))
        

    def on_close(self):
        # Close the window and destroy the image window
        cv2.destroyAllWindows()
        self.master.deiconify()
        self.second_window.destroy()

class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.start_point = (int(self.width/2), 0)
        self.end_point = (int(self.width/2), int(self.height))
        self.color = (0, 255, 0)
        self.thickness = 2

    def process_frame(self):
        # refactor ke sini
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.line(frame, self.start_point, self.end_point, self.color, self.thickness)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self.cap.release()
            return None
    
    def get_current_timestamp(self):
        """Returns the current frame timestamp in seconds."""
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current_frame / self.fps

    def seek_to_timestamp(self, timestamp):
        """Seeks to the specified timestamp in seconds."""
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_duration(self):
        """Returns the duration of the video in seconds."""
        num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return num_frames / self.fps
        
def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
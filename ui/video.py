import tkinter as tk
from PIL import ImageTk, Image
from processor.video import VideoProcessor
import cv2

class VideoUI:
    def __init__(self, master, source, in_orientation='right',line_roi=(500,0,500,800), is_video_player=True):
        self.master = master
        self.second_window = tk.Toplevel(master)
        self.master.title('Intelligent Port Monitoring Video Analysis')
        self.processor = VideoProcessor(is_download=False,
                                        is_agnostic_nms=True,
                                        is_save_bbox_dim=True,
                                        is_video_player=is_video_player,
                                        master=self,
                                        in_orientation=in_orientation,
                                        source=source,
                                        name='refactor',
                                        conf_thres=0.5,
                                        line_roi=line_roi,
                                        )
    
        self.second_window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.delay = int(1000 / self.processor.dataset.get_fps())
        self.paused = False
        self.seek_timestamp = None
        self.width = self.processor.width
        self.height = self.processor.height
        self.is_counting = False
        self.is_video_player = is_video_player
    

        # Create a frame to hold the canvas and the controls
        self.main_frame = tk.Frame(self.second_window)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.second_window.grid_rowconfigure(0, weight=1)
        self.second_window.grid_columnconfigure(0, weight=1)

        # Add the canvas
        self.canvas = tk.Canvas(self.main_frame, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Add the controls frame
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, sticky="ew") 
        self.main_frame.grid_rowconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
       
        # Add the pause/play button
        self.pause_button = tk.Button(self.controls_frame, text="Pause/Play", command=self.toggle_pause)
        self.pause_button.grid(row=1, column=0, padx=5, pady=5)
        if not self.is_video_player:
            self.pause_button.grid_forget()
       
        # Add the seek forward button
        self.seek_forward_button = tk.Button(self.controls_frame, text="Seek Forward", command=self.seek_forward)
        self.seek_forward_button.grid(row=1, column=1, padx=5, pady=5)        
        if not self.is_video_player:
            self.seek_forward_button.grid_forget()

        # Add the seek backward button
        self.seek_backward_button = tk.Button(self.controls_frame, text="Seek Backward", command=self.seek_backward)
        self.seek_backward_button.grid(row=1, column=2, padx=5, pady=5)
        if not self.is_video_player:
            self.seek_backward_button.grid_forget()

        # Add the toggle counting button
        self.toggle_counting_button = tk.Button(self.controls_frame, text="Toggle Counting", command=self.toggle_counting)
        self.toggle_counting_button.grid(row=1, column=3, padx=5, pady=5)

        # Create the output text box
        self.sample_output = tk.Text(self.main_frame, width=80, height=10, state="disabled")
        self.sample_output.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create the vertical scrollbar
        scrollbar = tk.Scrollbar(self.main_frame)
        scrollbar.grid(row=0, column=2, padx=0, pady=5, sticky="ns")

        # Associate the scrollbar with the text box
        self.sample_output.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.sample_output.yview)

        self.main_frame.grid_rowconfigure(2, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.update_canvas()

    def update_message(self, message=""):
        self.sample_output.configure(state="normal")  # make the textbox editable
        self.sample_output.insert("end", message + "\n")  # insert the new message
        self.sample_output.configure(state="disabled")  # make the textbox

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
    
    def toggle_counting(self):
        if not self.is_counting:
            self.is_counting = True
            self.processor.start_state_tracker()
        else:
            self.is_counting = False
            self.processor.stop_state_tracking()

    def seek_forward(self):
        current_timestamp = self.processor.get_current_timestamp()
        new_timestamp = current_timestamp + 3
        self.seek_timestamp = new_timestamp

    def seek_backward(self):
        current_timestamp = self.processor.get_current_timestamp()
        new_timestamp = current_timestamp - 3
        self.seek_timestamp = new_timestamp

    def seek_to_timestamp(self, timestamp):
        self.processor.seek_to_timestamp(float(timestamp))
        
    def on_close(self):
        # Close the window and destroy the image window
        cv2.destroyAllWindows()
        self.master.deiconify()
        self.second_window.destroy()

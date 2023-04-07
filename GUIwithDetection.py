import os
import pathlib
import time
import PIL.Image, PIL.ImageTk

import customtkinter as tk
import cv2
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import numpy as np
import vlc
from customtkinter import filedialog
from feat import Detector
from feat.plotting import imshow
#from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#detector = Detector()

class MyGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Lie Detector")

        tk.set_appearance_mode("dark")
        tk.set_default_color_theme("dark-blue")

        #set the size of the window
        self.set_window_size(window, 0.8, 0.8)

        # Create Live Feed button
        self.live_feed_button = tk.CTkButton(window, text="Live Feed", command=self.live_feed)
        self.live_feed_button.pack(pady=10)
        
        # Create Import Video button
        self.import_video_button = tk.CTkButton(window, text="Import Video", command=self.import_video)
        self.import_video_button.pack(pady=10)

        self.paused = False

    def live_feed(self):

        #Remove previous buttons with pack_forget
        self.live_feed_button.pack_forget()
        self.import_video_button.pack_forget()

        #create a new Frame for live video detection
        self.select_live_feed = tk.CTkFrame(self.window)
        self.select_live_feed.pack(expand=True, fill="both")

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.zoomed_fig = Figure(figsize=(5, 4), dpi=100)

        # Create a OpenCV capture object
        self.cap = cv2.VideoCapture(0)

        # Create a label for live feed
        self.live_feed_label = tk.CTkLabel(self.select_live_feed,text="Live Feed")
        self.live_feed_label.pack(side="top", anchor="center", padx=10, pady=10)

        #create a frame for the canvas to anchor to center
        self.canvas_frame = tk.CTkFrame(self.select_live_feed)
        self.canvas_frame.pack(side=tk.TOP, anchor=tk.CENTER)

        #create canvas to display live feed
        self.canvas_live = tk.CTkCanvas(self.canvas_frame, width=640, height=480)
        self.canvas_live.pack(expand=True, fill="both", side="top")

        self.btn_live_feed_pause = tk.CTkButton(self.canvas_frame, text="Play/Pause", command=self.toggle_pause)
        self.btn_live_feed_pause.pack(side=tk.TOP, anchor=tk.CENTER)

        arr_emotion=["happy", "happy", "happy", "angry", "angry", "happy", "happy", "calm", "calm", "calm", "calm", "calm"]
        x_array = [1,2,3,4,5,6,7,8,9,10,11,12]

        # Create figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Plot the initial data
        self.ax.plot(x_array, arr_emotion)
        self.ax.set_xlim([0, 15])
        #self.ax.set_ylim([-5, 5])
        self.ax.set_xlabel('Time (frame)')
        self.ax.set_ylabel('Emotion')
        self.ax.set_facecolor('#212121')
        self.fig.set_facecolor('#212121')
        self.ax.tick_params(colors='white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')

        # Create the zoomed-in subplot
        self.zoomed_fig, self.zoomed_ax = plt.subplots()
        self.zoomed_canvas = FigureCanvasTkAgg(self.zoomed_fig)
        self.zoomed_canvas.draw()
        self.zoomed_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.zoomed_ax.set_facecolor('#212121')
        self.zoomed_fig.set_facecolor('#212121')
        self.zoomed_ax.tick_params(colors='white')
        self.zoomed_ax.spines['top'].set_visible(False)
        self.zoomed_ax.spines['right'].set_visible(False)
        self.zoomed_ax.spines['bottom'].set_color('white')
        self.zoomed_ax.spines['left'].set_color('white')

        # Create the span selector widget
        self.span = SpanSelector(self.ax, self.zoom, 'horizontal', useblit=True,
                                 props=dict(alpha=0.5, facecolor='grey'))
        self.span.visible = False

        # Bind the hover event to the canvas
        self.canvas.mpl_connect('motion_notify_event', self.hover)

        #create a back button to return to previous screen
        self.back_button = tk.CTkButton(self.canvas_frame, text="Back", command=self.back_to_main_live_feed)
        self.back_button.pack(padx=(10,10), pady=(10,10))
        #call the update function
        self.update_frame()



    def zoom(self, xmin, xmax):
            x, y = self.ax.lines[0].get_data()
            mask = (x > xmin) & (x < xmax)
            x_zoomed, y_zoomed = x[mask], y[mask]
            self.zoomed_ax.clear()
            self.zoomed_ax.plot(x_zoomed, y_zoomed)
            self.zoomed_ax.set_xlim([xmin, xmax])
            self.zoomed_ax.set_xlabel('Time (frame)')
            self.zoomed_ax.set_ylabel('Emotion')
            self.zoomed_ax.set_title('Zoomed In')
            self.zoomed_canvas.draw()

    def hover(self, event):
            if event.inaxes == self.ax:
                x, y = event.xdata, event.ydata
                self.ax.format_coord = lambda x, y: f'Time={x:.2f}, Amplitude={y:.2f}'
                self.canvas.draw_idle()
                if self.span.active:
                    self.zoom(x)


    def update_frame(self):
        # Capture video frame
        ret, frame = self.cap.read()
        
        if not self.paused:
            # Convert video frame to tkinter compatible format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            img_tk = PIL.ImageTk.PhotoImage(image=img)
            
            # Update canvas with video frame
            self.canvas_live.img_tk = img_tk
            self.canvas_live.create_image(0, 0, anchor=tk.NW, image=img_tk)

            #TODO include code that can take the images and store them for analysis
            
        # Repeat video loop after 15 milliseconds
        self.window.after(15, self.update_frame)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_live_feed_pause.configure(text="Resume")
        else:
            self.btn_live_feed_pause.configure(text="Pause")

    def import_video(self):

        #Remove previous buttons with pack_forget
        self.import_video_button.pack_forget()
        self.live_feed_button.pack_forget()

        #Create new Frame for the import video window
        self.select_import_video = tk.CTkFrame(self.window)
        self.select_import_video.pack(expand=True, fill="both")
        
        #create a label for the button
        self.file_label = tk.CTkLabel(self.select_import_video, text="Select a video file")
        self.file_label.pack()

        #Create a button to allow selection of video
        self.select_button = tk.CTkButton(self.select_import_video, text="Select File", command=self.select_file)
        self.select_button.pack()

        #create a frame for two canvases to display the video and result side by side
        self.display_output = tk.CTkFrame(self.select_import_video)
        self.display_output.pack(expand=True, fill="both")

        # Create canvas for video player
        self.canvas_video = tk.CTkCanvas(self.display_output, bg="black")
        self.canvas_video.pack(fill=tk.BOTH, expand=True,side=tk.LEFT, padx=(10,0), pady=10)

        #create canvas for video results
        self.canvas_video_result = tk.CTkCanvas(self.display_output, bg="black")
        self.canvas_video_result.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=(10,10), pady=10)

        # Create frame for play/pause/skipback/skipforward buttons
        self.buttons_frame = tk.CTkFrame(self.select_import_video)
        self.buttons_frame.pack()

        self.skip_back_button = tk.CTkButton(self.buttons_frame, text="<<", command=self.skip_back)
        self.skip_back_button.pack(side=tk.LEFT)

        #create a current time label to show the current timestamp the video is on
        #WIP
        self.label_text = tk.StringVar()
        self.label_text.set("0:00")
        self.text_box_current_timestamp = tk.CTkLabel(self.buttons_frame, textvariable=self.label_text)
        self.text_box_current_timestamp.pack(side=tk.LEFT)

        self.play_button = tk.CTkButton(self.buttons_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.CTkButton(self.buttons_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        self.restart_button = tk.CTkButton(self.buttons_frame, text="Restart", command=self.restart)
        self.restart_button.pack(side=tk.LEFT)

        self.analyze_button = tk.CTkButton(self.buttons_frame, text="Analyze", command=self.analyze)
        self.analyze_button.pack(side=tk.LEFT)

        self.text_box_final_timestamp = tk.CTkLabel(self.buttons_frame)
        self.text_box_final_timestamp.pack(side=tk.LEFT)

        self.skip_forward_button = tk.CTkButton(self.buttons_frame, text=">>", command=self.skip_forward)
        self.skip_forward_button.pack(side=tk.LEFT)

        # Create media player object
        self.media_player = None
        

        #self.update_time()
        # Add a button for going back to the original screen
        self.back_button_import = tk.CTkButton(self.select_import_video, text="Back", command=self.back_to_main)
        self.back_button_import.pack(pady=10)
    
    def select_file(self):
        #Allow selection of video through File Explorer
        #BUG if you select a second video, it opens a new window
        file_path = tk.filedialog.askopenfilename()
        #DEV places the file path
        self.file_label.configure(text=file_path)

        # Create new media player object with selected file
        #BUG located here with creating a new vlc object
        self.media_player = vlc.MediaPlayer(file_path)

        # Set the media player to display in the canvas
        self.media_player.set_hwnd(self.canvas_video.winfo_id())

        #WIP set the framerate of the video, needed to advance video frame by frame
        #TODO Figure out how to get this data through parsing
        self.frame_rate = 34

    def play(self):
        #play button functionality
        #TODO reset video upon ending
        if self.media_player is not None:
            self.media_player.play()
        
    def restart(self):
        #restart video
        if self.media_player is not None:
            self.media_player.set_time(0)
            time.sleep(0.1)
            self.media_player.set_hwnd(self.canvas_video.winfo_id())
            self.media_player.set_pause(True)

    def pause(self):
        #pause button functionality
        if self.media_player is not None:
            self.media_player.set_pause(True)

    def skip_back(self):
        #Skip back functionality, works in ms
        if self.media_player is not None:
            current_time = self.media_player.get_time()
            self.media_player.set_time(current_time - self.frame_rate) # skip one frame back in ms

    def skip_forward(self):
        #Skip forword functionality, works in ms
        if self.media_player is not None:
            current_time = self.media_player.get_time()
            self.media_player.set_time(current_time + self.frame_rate) # skip one frame forward in ms

    def set_window_size(self, window, x, y):

        # Get the screen dimensions
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # Set the size and position of the main window
        window_width = int(screen_width * x)
        window_height = int(screen_height * y)
        window_x = int((screen_width - window_width) / 2)
        window_y = int((screen_height - window_height) / 2)
        window.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")

    def analyze(self):
        print("Hello World")

    def pause_play_CV(self):
        self.pause = not self.pause
    def update_time(self):
        state = self.media_player.get_state()
        if state == vlc.State.Playing:
            self.label_text.set(str(media_player.get_time()))
        self.text_box_current_timestamp.after(10, self.update_time())

    def back_to_main_live_feed(self):
        # Stop the video and hide the live feed and subwindows, and show the original screen
        self.cap.release()
        self.select_live_feed.destroy()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)

    def back_to_main(self):
        self.select_import_video.destroy()
        self.media_player.stop()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)

if __name__ == "__main__":
    window = tk.CTk()
    gui = MyGUI(window)
    window.mainloop()
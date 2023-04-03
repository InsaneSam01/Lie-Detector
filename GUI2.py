import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image
import vlc
import time

class MyGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("My GUI")

        # Get the screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Set the size and position of the main window
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        window_x = int((screen_width - window_width) / 2)
        window_y = int((screen_height - window_height) / 2)
        self.window.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
        
        # Create Live Feed button
        self.live_feed_button = tk.Button(window, text="Live Feed", command=self.live_feed)
        self.live_feed_button.pack(pady=10)
        
        # Create Import Video button
        self.import_video_button = tk.Button(window, text="Import Video", command=self.import_video)
        self.import_video_button.pack(pady=10)
        
    def live_feed(self):

        #Remove previous buttons with pack_forget
        self.live_feed_button.pack_forget()
        self.import_video_button.pack_forget()

        #create a new Frame for live video detection
        self.select_live_feed = tk.Frame(self.window)
        self.select_live_feed.pack(expand=True, fill="both")

        # Create a OpenCV capture object
        self.cap = cv2.VideoCapture(0)

        # Create a label for live feed
        self.live_feed_label = tk.Label(self.select_live_feed,text="Live Feed")
        self.live_feed_label.pack(expand=True, side="top", anchor="center")

        #create a frame for the canvas to anchor to center
        self.canvas_frame = tk.Frame(self.select_live_feed)
        self.canvas_frame.pack(side=tk.TOP, anchor=tk.CENTER)

        #create canvas to display live feed
        self.canvas = tk.Canvas(self.canvas_frame, width=640, height=480)
        self.canvas.pack(expand=True, fill="both", side="top")

        #Create a new frame for subwindows
        self.subwindows = tk.Frame(self.select_live_feed)

        self.subwindows.pack(side="bottom",fill="both", expand=True, padx=10, pady=10)

        #Create subwindows to fit in the main Subwindow Frame
        #Subwindow1 creation
        
        self.subwindow1 = tk.Frame(self.subwindows, relief="solid", borderwidth=2,)
        self.subwindow1.grid(row=0, column=0, padx=10, pady=10,sticky="nsew")
        self.subwindow1_label = tk.Label(self.subwindow1, text="Subwindow 1")
        self.subwindow1_label.pack(side="top", padx=10, pady=10)
        self.subwindow1_textbox = tk.Text(self.subwindow1)
        self.subwindow1_textbox.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        #Subwindow2 creation
        self.subwindow2 = tk.Frame(self.subwindows, relief="solid", borderwidth=2,)
        self.subwindow2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.subwindow2_label = tk.Label(self.subwindow2, text="Subwindow 2")
        self.subwindow2_label.pack(side="top", padx=10, pady=10)
        self.subwindow2_textbox = tk.Text(self.subwindow2)
        self.subwindow2_textbox.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        #Subwindow3 creation
        self.subwindow3 = tk.Frame(self.subwindows, relief="solid", borderwidth=2,)
        self.subwindow3.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.subwindow3_label = tk.Label(self.subwindow3, text="Subwindow 3")
        self.subwindow3_label.pack(side="top", padx=10, pady=10)
        self.subwindow3_textbox = tk.Text(self.subwindow3)
        self.subwindow3_textbox.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        #place subwindows in correct positions
        self.subwindows.rowconfigure(0, weight=1)
        self.subwindows.columnconfigure(0, weight=1)
        self.subwindows.columnconfigure(1, weight=1)
        self.subwindows.columnconfigure(2, weight=1)

        #create a back button to return to previous screen
        self.back_button = tk.Button(self.canvas_frame, text="Back", command=self.back_to_main_live_feed)
        self.back_button.pack()
        
        #call the update function
        self.update_frame()

    def update_frame(self):
        # Get a frame from the video capture
        ret, frame = self.cap.read()
        
        # Display the frame on the canvas
        if ret:
            self.photo = cv2.cvtColor(frame, 0)
            self.photo = cv2.resize(self.photo, (640, 480))
            self.photo = tk.PhotoImage( data=cv2.imencode('.png', self.photo)[1].tobytes())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Call update_frame again after 10 milliseconds
        self.window.after(10, self.update_frame)


    def import_video(self):

        #Remove previous buttons with pack_forget
        self.import_video_button.pack_forget()
        self.live_feed_button.pack_forget()

        #Create new Frame for the import video window
        self.select_import_video = tk.Frame(self.window)
        self.select_import_video.pack(expand=True, fill="both")
        
        #create a label for the button
        self.file_label = tk.Label(self.select_import_video, text="Select a video file")
        self.file_label.pack()

        #Create a button to allow selection of video
        self.select_button = tk.Button(self.select_import_video, text="Select File", command=self.select_file)
        self.select_button.pack()

        #create a frame for two canvases to display the video and result side by side
        self.display_output = tk.Frame(self.select_import_video)
        self.display_output.pack(expand=True, fill="both")

        # Create canvas for video player
        self.canvas_video = tk.Canvas(self.display_output, bg="black")
        self.canvas_video.pack(fill=tk.BOTH, expand=True,side=tk.LEFT, padx=(10,0), pady=10)

        #create canvas for video results
        self.canvas_video_result = tk.Canvas(self.display_output, bg="black")
        self.canvas_video_result.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=(10,10), pady=10)

        # Create frame for play/pause/skipback/skipforward buttons
        self.buttons_frame = tk.Frame(self.select_import_video)
        self.buttons_frame.pack()

        self.skip_back_button = tk.Button(self.buttons_frame, text="<<", command=self.skip_back)
        self.skip_back_button.pack(side=tk.LEFT)

        self.play_button = tk.Button(self.buttons_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.buttons_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        self.restart_button = tk.Button(self.buttons_frame, text="Restart", command=self.restart)
        self.restart_button.pack(side=tk.LEFT)

        self.analyze_button = tk.Button(self.buttons_frame, text="Analyze Video", command=self.analyze_video)
        self.analyze_button.pack(side=tk.LEFT)

        self.skip_forward_button = tk.Button(self.buttons_frame, text=">>", command=self.skip_forward)
        self.skip_forward_button.pack(side=tk.LEFT)

        # Create media player object
        self.media_player = None
        
        # Add a button for going back to the original screen
        self.back_button_import = tk.Button(self.select_import_video, text="Back", command=self.back_to_main)
        self.back_button_import.pack(pady=10)
    
    def select_file(self):
        #Allow selection of video through File Explorer
        #BUG if you select a second video, it opens a new window
        file_path = tk.filedialog.askopenfilename()
        #DEV places the file path
        self.file_label.config(text=file_path)

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

    def analyze_video(self):
        new_window = tk.Toplevel(self.window)
        new_window.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
        
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
    window = tk.Tk()
    gui = MyGUI(window)
    window.mainloop()

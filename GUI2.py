import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image

class MyGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("My GUI")
        
        # Create Live Feed button
        self.live_feed_button = tk.Button(window, text="Live Feed", command=self.live_feed)
        self.live_feed_button.pack(pady=10)
        
        # Create Import Video button
        self.import_video_button = tk.Button(window, text="Import Video", command=self.import_video)
        self.import_video_button.pack(pady=10)
        
    def live_feed(self):
        # TODO: Implement live feed functionality
        
        #create a new Frame for live video detection
        self.live_feed_button.pack_forget()
        self.import_video_button.pack_forget()
        self.select_live_feed = tk.Frame(self.window)

        # Create a OpenCV capture object
        self.cap = cv2.VideoCapture(0)

        # Create canvas for video display
        self.canvas = tk.Canvas(self.select_live_feed, width=640, height=480)
        self.canvas.pack(expand=True, fill='both')

        self.back_button = tk.Button(self.select_live_feed, text="Back", command=self.back_to_main_live_feed)
        self.back_button.pack()
        
        self.update_frame()

        self.select_live_feed.pack()

    def update_frame(self):
        # Get a frame from the video capture
        ret, frame = self.cap.read()
        
        # Display the frame on the canvas
        if ret:
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = cv2.resize(self.photo, (640, 480))
            self.photo = tk.PhotoImage( data=cv2.imencode('.png', self.photo)[1].tobytes())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Call update_frame again after 10 milliseconds
        self.window.after(10, self.update_frame)


    def import_video(self):
        # Create a new frame for selecting video directory
        self.select_video_frame = tk.Frame(self.window)
        
        # Add a label and a button for selecting the video directory
        self.select_video_label = tk.Label(self.select_video_frame, text="Select a video file:")
        self.select_video_label.pack(pady=10)
        
        self.select_video_button = tk.Button(self.select_video_frame, text="Browse", command=self.browse_video)
        self.select_video_button.pack(pady=10)
        
        # Add a button for going back to the original screen
        self.back_button = tk.Button(self.select_video_frame, text="Back", command=self.back_to_main)
        self.back_button.pack(pady=10)
        
        # Show the new frame for selecting video directory
        self.import_video_button.pack_forget()
        self.live_feed_button.pack_forget()
        self.select_video_frame.pack()
    
    def browse_video(self):
        # Allow user to select a video file and save its directory
        self.video_directory = filedialog.askopenfilename()
        
        # Load and display the selected video on the frame
        self.video_capture = cv2.VideoCapture(self.video_directory)
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_canvas = tk.Canvas(self.select_video_frame, width=self.video_width, height=self.video_height)
        self.video_canvas.pack(pady=10)
        
        self.video_frame = tk.Frame(self.video_canvas)
        self.video_frame.pack()
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(pady=10)
        

        
    def back_to_main_live_feed(self):
        # Stop the video and hide the frame for selecting video directory, and show the original screen
        self.cap.release()
        self.select_live_feed.destroy()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)

    def back_to_main(self):
        self.live_feed_button.pack()
        self.import_video_button.pack()

if __name__ == "__main__":
    window = tk.Tk()
    gui = MyGUI(window)
    window.mainloop()

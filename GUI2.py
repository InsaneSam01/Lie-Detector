import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image

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
        # TODO: Implement live feed functionality
        
        #create a new Frame for live video detection
        self.live_feed_button.pack_forget()
        self.import_video_button.pack_forget()
        self.select_live_feed = tk.Frame(self.window)

        # Create a OpenCV capture object
        self.cap = cv2.VideoCapture(0)

        # Create a label and a canvas for video display
        self.live_feed_label = tk.Label(self.select_live_feed,text="Live Feed")
        self.live_feed_label.pack(expand=True, side="top", anchor="center")

        self.canvas = tk.Canvas(self.select_live_feed, width=640, height=480)
        self.canvas.pack(expand=True, fill="both", side="top")

        #Create a new frame for subwindows
        self.subwindows = tk.Frame(self.window)

        self.subwindows.pack(side="bottom",fill="both", expand=True, padx=10, pady=10)

        #Create subwindows to fit in the main Subwindow Frame
        self.subwindow1 = tk.Frame(self.window)
        
        self.subwindow1 = tk.Frame(self.subwindows, relief="solid", borderwidth=2,)
        self.subwindow1.grid(row=0, column=0, padx=10, pady=10,sticky="nsew")
        self.subwindow1_label = tk.Label(self.subwindow1, text="Subwindow 1")
        self.subwindow1_label.pack(side="top", padx=10, pady=10)
        self.subwindow1_textbox = tk.Text(self.subwindow1)
        self.subwindow1_textbox.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.subwindow2 = tk.Frame(self.subwindows, relief="solid", borderwidth=2,)
        self.subwindow2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.subwindow2_label = tk.Label(self.subwindow2, text="Subwindow 2")
        self.subwindow2_label.pack(side="top", padx=10, pady=10)
        self.subwindow2_textbox = tk.Text(self.subwindow2)
        self.subwindow2_textbox.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.subwindow3 = tk.Frame(self.subwindows, relief="solid", borderwidth=2,)
        self.subwindow3.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.subwindow3_label = tk.Label(self.subwindow3, text="Subwindow 3")
        self.subwindow3_label.pack(side="top", padx=10, pady=10)
        self.subwindow3_textbox = tk.Text(self.subwindow3)
        self.subwindow3_textbox.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.subwindows.rowconfigure(0, weight=1)
        self.subwindows.columnconfigure(0, weight=1)
        self.subwindows.columnconfigure(1, weight=1)
        self.subwindows.columnconfigure(2, weight=1)

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
        self.back_button_import = tk.Button(self.select_video_frame, text="Back", command=self.back_to_main)
        self.back_button_import.pack(pady=10)
        
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
        # Stop the video and hide the live feed and subwindows, and show the original screen
        self.cap.release()
        self.select_live_feed.destroy()
        self.subwindows.destroy()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)

    def back_to_main(self):
        self.select_video_frame.destroy()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)

if __name__ == "__main__":
    window = tk.Tk()
    gui = MyGUI(window)
    window.mainloop()

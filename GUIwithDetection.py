from time import sleep
import PIL.Image, PIL.ImageTk
import customtkinter as tk
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
import tensorflow as tf
import vlc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

arr_emotion=[]
x_array = []

class MyGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Lie Detector")

        #set dark mode and color theme
        tk.set_appearance_mode("dark")
        tk.set_default_color_theme("dark-blue")

        #set the size of the window
        self.set_window_size(window, 0.8, 0.8)

        img = PIL.Image.open("CxBEoQi.jpg")

        self.img = tk.CTkImage(img, img, (448.4,208.1))

        # Create Live Feed button
        self.live_feed_button = tk.CTkButton(window, text="", command=self.live_feed, image=self.img)
        self.live_feed_button.pack(pady=10)

        self.labelbtn = tk.CTkLabel(window, text="Live Feed", compound=tk.CENTER, font=("Arial", 24))
        self.labelbtn.place(in_=self.live_feed_button, relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Create Import Video button
        self.import_video_button = tk.CTkButton(window, text="Import Video", command=self.import_video)
        self.import_video_button.pack(pady=10)

        #create quit button
        self.app_quit = tk.CTkButton(window, text="Quit", command=self.window.quit)
        self.app_quit.pack(pady=10)

        self.paused = False

    def live_feed(self):

        #Remove previous buttons with pack_forget
        self.live_feed_button.pack_forget()
        self.import_video_button.pack_forget()
        self.app_quit.pack_forget()

        #create a new Frame for live video detection
        self.select_live_feed = tk.CTkFrame(self.window)
        self.select_live_feed.pack(expand=True, fill="both")
    
        # Create a OpenCV capture object
        self.cap = cv2.VideoCapture(0)

        #return width and height
        self.width = 720
        self.height = 720

        #create a frame for the canvas to anchor to center
        #BUG doesnt scale correctly
        self.canvas_frame = tk.CTkFrame(self.select_live_feed)
        self.canvas_frame.pack(side=tk.LEFT, anchor=tk.CENTER, padx=(50,0))

        # Create a label for live feed
        self.live_feed_label = tk.CTkLabel(self.canvas_frame, text="Live Feed")
        self.live_feed_label.pack(side="top", anchor="center")

        #create canvas to display live feed
        self.canvas_live = tk.CTkCanvas(self.canvas_frame, width=self.width, height=self.height)
        self.canvas_live.pack(side="top")

        #create button to pause/play live feed video
        self.btn_live_feed_pause = tk.CTkButton(self.canvas_frame, text="Play/Pause", command=self.toggle_pause)
        self.btn_live_feed_pause.pack(side=tk.TOP, anchor=tk.CENTER, pady=(10,0))

        #create a back button to return to previous screen
        self.back_button = tk.CTkButton(self.canvas_frame, text="Back", command=self.back_to_main_live_feed)
        self.back_button.pack(padx=(10,10), pady=(10,10))

        #create frame for figures
        self.figures_frame = tk.CTkFrame(self.select_live_feed)
        self.figures_frame.pack(side=tk.LEFT, fill="both", expand=True)

        # Load the pre-trained emotion detection model
        self.model = tf.keras.models.load_model('model_weights.h5')

        # Compile the model with categorical cross-entropy loss, adam optimizer, and accuracy metric
        self.model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

        self.create_plots(self.figures_frame)

        #call the update function
        self.update_frame()


    def update_frame(self):
        # Capture video frame
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.width, self.height))
        # Define the emotion labels
        EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        if not self.paused:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            label = None
            # Loop over the detected faces
            for (x, y, w, h) in faces:
                # Extract the face ROI
                roi = gray[y:y + h, x:x + w]
    
                # Resize the face ROI to match the input size of the model
                roi = cv2.resize(roi, (48, 48))

                # Preprocess the face ROI
                roi = roi.astype("float") / 255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction on the face ROI using the emotion detection model
                preds = self.model.predict(roi)[0]

                # Determine the dominant emotion label
                label = EMOTIONS[preds.argmax()]
                
                # Draw the bounding box around the face and label the detected emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            if label is not None:
                arr_emotion.append(label)
                latest_index = len(arr_emotion)
                x_array.append(latest_index)
                
            self.ax.plot(x_array, arr_emotion)
            self.canvas.draw()


            #testing
            #print(arr_emotion)
            #print(x_array)

            ## Convert the updated frame to the format compatible with Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(rgb_frame)
            img_tk = PIL.ImageTk.PhotoImage(image=img)

            # Get the dimensions of the canvas
            canvas_width = self.canvas_live.winfo_width()
            canvas_height = self.canvas_live.winfo_height()

            # Calculate the center point of the canvas
            center_x = canvas_width / 2
            center_y = canvas_height / 2

            ## Calculate the top-left corner coordinates of the frame
            frame_width = img_tk.width()
            frame_height = img_tk.height()
            x = center_x - (frame_width / 2)
            y = center_y - (frame_height / 2)

            # Update the canvas with the emotion detection results
            self.canvas_live.create_image(x, y, anchor=tk.NW, image=img_tk)
            self.canvas_live.image = img_tk # update reference to the image to prevent garbage collection
            
            
        # Repeat video loop after 15 milliseconds
        self.window.after(30, self.update_frame)

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
        self.app_quit.pack_forget()

        #Create new Frame for the import video window
        self.select_import_video = tk.CTkFrame(self.window)
        self.select_import_video.pack(expand=True, fill="both")
        
        #create label to show user if media is selected
        self.media_detected = tk.CTkLabel(self.select_import_video, text="No Media Detected")
        self.media_detected.pack()

        #create a label for the button
        self.file_label = tk.CTkLabel(self.select_import_video, text="Select a video file")
        self.file_label.pack()

        #Create a button to allow selection of video
        self.select_button = tk.CTkButton(self.select_import_video, text="Select File", command=self.select_file)
        self.select_button.pack()

        #create a frame as a container for the other frames
        self.frames_container = tk.CTkFrame(self.select_import_video)
        self.frames_container.pack(side=tk.LEFT, anchor=tk.CENTER, padx=(50,0))
        

        #create a frame for the import video canvas
        #self.display_output = tk.CTkCanvas(self.frames_container, width=self.width, height=self.height)
        #self.display_output.pack(side="top")

        # Create canvas for video player
        self.canvas_video = tk.CTkCanvas(self.frames_container, width=700, height=700)
        self.canvas_video.pack(side="top")

        #create a Frame to place the plots in
        self.plots = tk.CTkFrame(self.select_import_video, width=700, height=700)
        self.plots.pack(side=tk.LEFT, fill="both", expand=True)

        self.create_plots(self.plots)

        # Create frame for play/pause/skipback/skipforward buttons
        self.buttons_frame = tk.CTkFrame(self.frames_container)
        self.buttons_frame.pack(side=tk.TOP, anchor=tk.CENTER, pady=(10,0))

        #create a skip back button to go back a frame
        self.skip_back_button = tk.CTkButton(self.buttons_frame, text="<<", command=self.skip_back)
        self.skip_back_button.pack(side=tk.LEFT)

        #create a current time label to show the current timestamp the video is on
        #WIP
        self.label_text_currect = tk.StringVar()
        self.text_box_current_timestamp = tk.CTkLabel(self.buttons_frame, textvariable=self.label_text_currect)
        self.text_box_current_timestamp.pack(side=tk.LEFT)

        self.play_button = tk.CTkButton(self.buttons_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.CTkButton(self.buttons_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        self.restart_button = tk.CTkButton(self.buttons_frame, text="Restart", command=self.restart)
        self.restart_button.pack(side=tk.LEFT)

        self.analyze_button = tk.CTkButton(self.buttons_frame, text="Analyze", command=self.analyze)
        self.analyze_button.pack(side=tk.LEFT)

        self.label_text_final = tk.StringVar()
        self.text_box_final_timestamp = tk.CTkLabel(self.buttons_frame, textvariable=self.label_text_final)
        self.text_box_final_timestamp.pack(side=tk.LEFT)

        self.skip_forward_button = tk.CTkButton(self.buttons_frame, text=">>", command=self.skip_forward)
        self.skip_forward_button.pack(side=tk.LEFT)

        # Create media player object
        self.media_player = None
        
        # Add a button for going back to the original screen
        self.back_button_import = tk.CTkButton(self.frames_container, text="Back", command=self.back_to_main)
        self.back_button_import.pack(pady=10)
    
    def select_file(self):

        if self.media_player:
            self.media_player.stop()
        #Allow selection of video through File Explorer
        file_path = tk.filedialog.askopenfilename()
        #update media_detected label
        self.media_detected.configure(text="Media Found")
        #DEV places the file path
        self.file_label.configure(text=file_path)
        #Stop and delete old media player object if it exists
        if self.media_player is not None:
            self.media_player.stop()
            del self.media_player

        # Create new media player object with selected file
        self.media_player = vlc.MediaPlayer(file_path)

        # Set the media player to display in the canvas
        self.media_player.set_hwnd(self.canvas_video.winfo_id())

        #WIP set the framerate of the video, needed to advance video frame by frame
        #TODO Figure out how to get this data through parsing
        self.frame_rate = 34

    def create_plots(self, frame):

        arr_emotion = []
        x_array = []

        # Create figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Plot the initial data
        self.ax.plot(x_array, arr_emotion)
        self.ax.set_xlabel('Time (frame)', color="white", fontsize=16)
        self.ax.set_ylabel('Emotion', color="white", fontsize=16)
        self.ax.set_facecolor('#212121')
        self.fig.set_facecolor('#212121')
        self.ax.tick_params(colors='white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')

        # Create the zoomed-in subplot
        self.zoomed_fig, self.zoomed_ax = plt.subplots()
        self.zoomed_canvas = FigureCanvasTkAgg(self.zoomed_fig, frame)
        self.zoomed_canvas.draw()
        self.zoomed_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1,)

        self.zoomed_ax.set_facecolor('#212121')
        self.zoomed_fig.set_facecolor('#212121')
        self.zoomed_ax.tick_params(colors='white')
        self.zoomed_ax.spines['top'].set_visible(False)
        self.zoomed_ax.spines['right'].set_visible(False)
        self.zoomed_ax.spines['bottom'].set_color('white')
        self.zoomed_ax.spines['left'].set_color('white')

        # Create the span selector widget
        self.span = SpanSelector(self.ax, self.zoom, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor='grey'))
        self.span.set_visible(False)

        # Bind the hover event to the canvas
        self.canvas.mpl_connect('motion_notify_event', self.hover)

    def zoom(self, xmin, xmax):
            x, y = self.ax.lines[0].get_data()
            mask = (x > xmin) & (x < xmax)
            x_zoomed, y_zoomed = x[mask], y[mask]
            self.zoomed_ax.clear()
            self.zoomed_ax.plot(x_zoomed, y_zoomed)
            self.zoomed_ax.set_xlim([xmin, xmax])
            self.zoomed_ax.set_xlabel('Time (frame)',color="white", fontsize=16)
            self.zoomed_ax.set_ylabel('Emotion', color="white", fontsize=16)
            self.zoomed_ax.set_title('Zoomed In',color="white")
            self.zoomed_canvas.draw()

    def hover(self, event):
            if event.inaxes == self.ax:
                x, y = event.xdata, event.ydata
                self.ax.format_coord = lambda x, y: f'Time={x:.2f}, Amplitude={y:.2f}'
                self.canvas.draw_idle()

    def update_time(self):
        current_time = self.media_player.get_time()

        self.label_text_currect.set(str(current_time/1000))

        self.window.after(20, self.update_time)

    def play(self):
        #play button functionality
        #check if media has ended
        if self.media_player is not None:
            if self.media_player.get_state() == vlc.State.Ended:
             self.media_player.stop()
             self.media_player.set_time(0)
            self.media_player.play()
            sleep(0.1)
            self.update_time()
            self.label_text_final.set(str(self.media_player.get_length()/1000))
           
        
    def restart(self):
        #restart video
        if self.media_player is not None:
            self.media_player.stop()
            self.media_player.set_time(0)
            sleep(0.1)
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
        pass

    def pause_play_CV(self):
        self.pause = not self.pause

    def back_to_main_live_feed(self):
        # Stop the video and hide the live feed and subwindows, and show the original screen
        self.cap.release()
        del self.cap
        self.select_live_feed.destroy()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)
        self.app_quit.pack(pady=10)

    def back_to_main(self):
        #destroy all widgets and return to main screen
        self.select_import_video.destroy()
        self.live_feed_button.pack(pady=10)
        self.import_video_button.pack(pady=10)
        self.app_quit.pack(pady=10)

    #def quit_app(self):
        #self.window.quit()

if __name__ == "__main__":
    window = tk.CTk()
    gui = MyGUI(window)
    window.mainloop()
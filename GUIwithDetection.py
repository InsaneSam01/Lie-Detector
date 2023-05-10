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
from feat import Detector


class MyGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Lie Detector")

        #set dark mode and color theme
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

        #create quit button
        self.app_quit = tk.CTkButton(window, text="Quit", command=self.window.quit)
        self.app_quit.pack(pady=10)

        self.paused = False

        self.arr_emotion = []
        self.x_array = []

        # Define the emotion labels
        self.EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        # Load the pre-trained emotion detection model
        self.model = tf.keras.models.load_model('model_weights.h5')

        # Compile the model with categorical cross-entropy loss, adam optimizer, and accuracy metric
        self.model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

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

        #define width and height
        self.width = 720
        self.height = 480

        #create a frame as a container for the other frames
        self.frames_container_live = tk.CTkFrame(self.select_live_feed)
        self.frames_container_live.pack(side=tk.TOP, padx=(50,50), expand=True, fill=tk.BOTH)

        self.micro_expression_frame = tk.CTkFrame(self.frames_container_live)
        self.micro_expression_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        self.output_box(self.micro_expression_frame)
        #create a frame for the canvas to anchor to center
        #BUG doesnt scale correctly
        self.canvas_frame = tk.CTkFrame(self.frames_container_live)
        self.canvas_frame.pack(side=tk.LEFT, anchor=tk.CENTER, padx=20)

        # Create a label for live feed
        self.live_feed_label = tk.CTkLabel(self.canvas_frame, text="Live Feed")
        self.live_feed_label.pack(side="top", anchor="center")

        #create canvas to display live feed
        self.canvas_live = tk.CTkCanvas(self.canvas_frame, width=self.width, height=self.height)
        self.canvas_live.pack(side="top",fill="both", expand=True)

        self.radar_frame = tk.CTkFrame(self.frames_container_live)
        self.radar_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.create_radar_graph(self.radar_frame)

        #create button to pause/play live feed video
        self.btn_live_feed_pause = tk.CTkButton(self.canvas_frame, text="Play/Pause", command=self.toggle_pause)
        self.btn_live_feed_pause.pack(side=tk.TOP, anchor=tk.CENTER, pady=(10,0))

        #create a back button to return to previous screen
        self.back_button = tk.CTkButton(self.canvas_frame, text="Back", command=self.back_to_main_live_feed)
        self.back_button.pack(padx=(10,10), pady=(10,10))

        #create frame for figures
        self.figures_frame = tk.CTkFrame(self.select_live_feed)
        self.figures_frame.pack(side=tk.BOTTOM, fill="both", expand=True)

        self.create_graphs(self.figures_frame)
        
        #call the update functions
        self.update_frame()

        self.update_text()
        
    def output_box(self, parent_frame):
        title_label = tk.CTkLabel(parent_frame, text="Micro-Expressions Detected", font=("Helvetica", 16))
        title_label.pack(padx=5, pady=5)
        # create the output box
        output_var = tk.StringVar()
        output_var.set("-Detection recorded from frame 10 to 16 ")
        self.output_box_widget = tk.CTkTextbox(parent_frame, height=400, width=300, state="disabled")
        self.output_box_widget.pack(side=tk.LEFT)
        self.output_box_widget.configure(state="normal")
        self.output_box_widget.insert(tk.END, output_var.get() + "\n")
        self.output_box_widget.configure(state="disabled")

        # set flag to True when the widget is created
        self.output_box_visible = True

    def update_text(self):
        if self.output_box_visible:
            # Generate a random text
            random_text = "Random text " + str(np.random.randint(1, 100))

            # Clear the output box
            self.output_box_widget.configure(state="normal")
            #self.output_box.delete('1.0', tk.END)

            # Update the output box with the random text
            self.output_box_widget.insert(tk.END, random_text + "\n")

            # Disable the output box
            self.output_box_widget.configure(state="disabled")

        # Call this function again after 1 second
        self.window.after(1000, self.update_text)
        
    def micro_expressions(self, emotions, start_from=0):
        subarrays = []
        current_emotion = None
        current_subarray = []
        start_index = 0

        for i, emotion in enumerate(emotions[start_from:], start_from):
            if current_emotion == emotion:
                current_subarray.append(emotion)
            else:
                if current_emotion is not None and 5 <= len(current_subarray) <= 12:
                    subarrays.append((start_index, i-1))
                current_emotion = emotion
                current_subarray = [emotion]
                start_index = i

        # Add last subarray if it exists
        if current_emotion is not None and 5 <= len(current_subarray) <= 12:
            subarrays.append((start_index, len(emotions)-1))

        result = [(start, end) for start, end in subarrays if emotions[start:end+1].count(emotions[start]) == len(emotions[start:end+1]) and len(emotions[start:end+1]) >= 5]

        for start, end in result:
            duration = end - start + 1
            emotion = emotions[start]
            print(f"Detected micro-expression of {emotion} for {duration} frames.")
        #return result

    def update_frame(self):

        # Capture video frame
        ret, frame = self.cap.read()

        if ret:
            #resize the image to fit the tk canvas
            frame = cv2.resize(frame, (self.width, self.height))

            label = None
            self.num = [0,0,0,0,0,0,0]
            
            if not self.paused:
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                
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
                    self.preds = self.model.predict(roi)[0]
                    self.num = np.round(self.preds * 100, decimals = 2)
                    #print([rd])

                    # Determine the dominant emotion label
                    label = self.EMOTIONS[self.preds.argmax()]
                    
                    # Draw the bounding box around the face and label the detected emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
                if label is not None:
                    self.arr_emotion.append(label)
                    latest_index = len(self.arr_emotion)
                    self.x_array.append(latest_index)

                
                self.ax.plot(self.x_array, self.arr_emotion)
                self.update_radar_chart(self.num)
                self.canvas.draw()
            
            #self.update_radar_chart(self.num)
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

        # Repeat video loop after 30 milliseconds
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

        #wip pyfeat detector
        #self.detector = Detector()

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
        self.select_button.pack(pady=(0,10))

        #create a frame as a container for the other frames
        self.frames_container_import = tk.CTkFrame(self.select_import_video)
        self.frames_container_import.pack(side=tk.TOP, padx=(50,50), expand=True, fill=tk.BOTH)

        #frame for video player
        self.video_container = tk.CTkFrame(self.frames_container_import)
        self.video_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Create canvas for video player
        self.canvas_video = tk.CTkCanvas(self.video_container)
        self.canvas_video.pack(side="top", expand=True, fill=tk.Y)

        # Create frame for play/pause/skipback/skipforward buttons
        self.buttons_frame = tk.CTkFrame(self.video_container)
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

        # Add a button for going back to the original screen
        self.back_button_import = tk.CTkButton(self.video_container, text="Back", command=self.back_to_main)
        self.back_button_import.pack(pady=10)

        #create a Frame to place the plots in
        self.plots = tk.CTkFrame(self.select_import_video)
        self.plots.pack(side=tk.BOTTOM, fill="both", expand=True)

        self.create_graphs(self.plots)

        # Create media player object
        self.media_player = None

        self.arr_emotion = []
        self.x_array = []
        

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

    def create_graphs(self, frame):

        # Create figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Define graph details
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
        self.zoomed_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1,)

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
            
            # Get the index of the line object
            line_index = len(self.ax.lines) - 1

            x, y = self.ax.lines[line_index].get_data()
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

    def create_radar_graph(self, frame):
        title_label = tk.CTkLabel(frame, text="Emotions Detected", font=("Helvetica", 16))
        title_label.pack(padx=5, pady=5)

        #define radar graph basics
        fig = plt.Figure(figsize=(3, 3))
        fig.set_facecolor('#212121')
        self.radarcanvas = FigureCanvasTkAgg(fig, frame)
        self.radarcanvas.get_tk_widget().pack(side=tk.RIGHT, fill='both', expand=1)
        
        #get lenght of emotions
        self.n_emotions = len(self.EMOTIONS)
        #define inner circles
        self.label_loc = np.linspace(start=0, stop=2 * np.pi, num=self.n_emotions, endpoint=False) + (2*np.pi/(2*self.n_emotions))
        #create plot
        self.radar = fig.add_subplot(111, polar=True)
        self.radar.tick_params(axis='both', which='major', pad=7)

        self.radar.set_facecolor('#212121')
        self.radar.tick_params(colors='#1F86CF')

        #change color of outer ring
        self.radar.spines['polar'].set_color('white')

        # Rotate plot by 90 degrees
        self.radar.set_theta_offset(np.pi / 2)

        #create labels to show on the radar graph
        lines, labels = self.radar.set_thetagrids(np.degrees(self.label_loc), labels=self.EMOTIONS)
        value = np.zeros(self.n_emotions)
        self.value_plot, = self.radar.plot(np.append(self.label_loc, self.label_loc[0]), np.append(value, value[0]), marker='o', color='green', linewidth=1)
        self.value_fill, = self.radar.fill(np.append(self.label_loc, self.label_loc[0]), np.append(value, value[0]), color='green', alpha=0.2)


    def update_radar_chart(self, data):
        #value = np.random.randint(1, 10, size=len(self.EMOTIONS))
        value = data
        self.value_plot.set_ydata(np.append(data, data[0]))
        self.value_fill.set_xy(np.column_stack((self.label_loc, data)))

        self.radar.relim()
        self.radar.autoscale_view()

        self.radarcanvas.draw()


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

        # get the FPS of the video
        #is always zero unless we play the video first
        fps = self.media_player.get_fps()

        # convert FPS from frames per second to milliseconds per frame
        #BUG small bug about fps being float exact number 
        #is needed to skip frame exactly, if truncated, not every
        #press will skip frame
        self.frame_rate = int(1000 / fps)
           
        
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
        self.output_box_visible = False
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

if __name__ == "__main__":
    window = tk.CTk()
    gui = MyGUI(window)
    window.mainloop()
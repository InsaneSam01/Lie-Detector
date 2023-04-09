import tkinter as tk
from feat import Detector
import os
import pathlib
import cv2
import numpy as np
from feat.plotting import imshow
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class GraphGUI(tk.Tk):
    def __init__(self, file_path):
        super().__init__()
        self.title("Live Graph from CSV")
        self.geometry("800x600")

        self.detector = Detector()

        # Read CSV file and set up initial graph
        self.df = pd.read_csv('Data.csv')
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.df['Frame'], self.df['Emotion'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Define the resolution and frame rate
        resolution = (640, 480)
        fps = 24
        #micro 40ms and 500ms so between 1 and 12
        # Initialize the camera capture object
        cap = cv2.VideoCapture(0)

        # Set the camera resolution and frame rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)

        # Initialize the array to store the frames
        frames = []
        # Get the desktop path
        desktop_path = str(pathlib.Path.home() / "Desktop")

        # Create the folder to store the images
        self.folder_path = os.path.join(desktop_path, "images")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # Initialize the frame counter
        frame_count = 0

        # Loop over the frames
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            # Check if the frame was successfully captured
            if not ret:
                break
            
            # Save the frame as an image in the folder
            image_path = os.path.join(self.folder_path, f"frame_{frame_count}.jpg")
            display = os.path.join(self.folder_path, f"frame_{0}.jpg")
            cv2.imwrite(image_path, frame)

            # Increment the frame counter
            frame_count += 1

            # Add the frame to the array
            frames.append(frame)

            # Display the frame (optional)
            cv2.imshow('Frame', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

            
        # Release the camera capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        # Convert the array to a numpy array and print its shape
        frames = np.array(frames)
        print(frames.shape)
        print(frame_count)

        
        self.after(1000, self.update_graph)

    def update_graph(self):

        
        # Read CSV file and update the graph if there are new values
        new_df = pd.read_csv('Data.csv')

        i = 1
        while i < 10:
            display = os.path.join(self.folder_path, f"frame_{i}.jpg")
            single_face_prediction = self.detector.detect_image(display)
            max_col= single_face_prediction[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].idxmax(axis=1)
            max_val=single_face_prediction[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].max(axis=1)
        
            new_row = {'Frame': i+1, 'Emotion': max_col.to_string(index=False)}
            new_df.loc[len(new_df)] = new_row
            print(len(new_df))
            i=i+1
            
            
        if not new_df.equals(self.df):
                self.df = new_df
                self.line.set_data(self.df['Frame'], self.df['Emotion'])
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()



if __name__ == '__main__':
    file_path = 'Data.csv'
    app = GraphGUI(file_path)
    app.mainloop()
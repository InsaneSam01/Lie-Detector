import tkinter as tk
import cv2

class VideoRecorderGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Recorder")
        
        # Create OpenCV capture object
        self.cap = cv2.VideoCapture(0)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # Create buttons for saving and deleting video
        self.save_button = tk.Button(window, text="Save Video", command=self.save_video)
        self.save_button.pack(side=tk.LEFT, padx=10)
        
        self.delete_button = tk.Button(window, text="Delete Video", command=self.delete_video)
        self.delete_button.pack(side=tk.RIGHT, padx=10)
        
        self.update_frame()
    
    def update_frame(self):
        # Get a frame from the video capture
        ret, frame = self.cap.read()
        
        # Display the frame on the canvas
        if ret:
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = cv2.resize(self.photo, (640, 480))
            self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.photo)[1].tobytes())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Call update_frame again after 10 milliseconds
        self.window.after(10, self.update_frame)
    
    def save_video(self):
        # Save the video to a file
        filename = "video.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        
        while True:
            # Get a frame from the video capture
            ret, frame = self.cap.read()
            
            # Write the frame to the video file
            if ret:
                out.write(frame)
                
                # Display the frame on the canvas
                self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = cv2.resize(self.photo, (640, 480))
                self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.photo)[1].tobytes())
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Update the window to show the new frame
                self.window.update()
            else:
                break
        
        # Release resources
        out.release()
    
    def delete_video(self):
        # Delete the video file
        import os
        filename = "video.avi"
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")

if __name__ == "__main__":
    window = tk.Tk()
    gui = VideoRecorderGUI(window)
    window.mainloop()
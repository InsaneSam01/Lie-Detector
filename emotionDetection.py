import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('model_weights.h5')

# Compile the model with categorical cross-entropy loss, adam optimizer, and accuracy metric
model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Define the emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create a video capture object
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

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
        preds = model.predict(roi)[0]

        # Determine the dominant emotion label
        label = EMOTIONS[preds.argmax()]

        # Draw the bounding box around the face and label the detected emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Display the frame with emotion detection results
    cv2.imshow("Emotion Detection", frame)

    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

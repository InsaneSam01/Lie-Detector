from feat import FaceLandmarkDetection, MicroExpressionRecognition
import cv2
#text
fld = FaceLandmarkDetection()

mer = MicroExpressionRecognition()

cap = cv2.VideoCapture(0) # Open the camera

while True:
    ret, frame = cap.read() # Read a frame from the camera
    if ret:
        # Detect the landmarks of the face in the frame
        landmarks = fld.detect_landmarks(frame)

        # Recognize micro-expressions from the detected landmarks
        expression = mer.recognize_micro_expression(landmarks)

        # Display the live feed with the recognized expression
        cv2.putText(frame, expression, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Micro Expression Detection', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cap.release() # Release the camera
cv2.destroyAllWindows() # Destroy all windows

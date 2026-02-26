import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import model_from_json

# Load model architecture and weights
with open('sign_language_model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('sign_language_model.weights.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
classes.remove('J')
classes.remove('Z')

# Constants
ROI_TOP_LEFT = (100, 100)
ROI_SIZE = 220
CONFIDENCE_THRESHOLD = 75  # Show prediction only if confidence > threshold
PREDICTION_HISTORY_LENGTH = 5  # Number of frames to average for smoothing

# Queue to store recent predictions
prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Enhance contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 28, 28, 1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI rectangle
    cv2.rectangle(frame, ROI_TOP_LEFT, (ROI_TOP_LEFT[0] + ROI_SIZE, ROI_TOP_LEFT[1] + ROI_SIZE), (255, 0, 0), 2)

    # Crop ROI from frame
    roi = frame[ROI_TOP_LEFT[1]:ROI_TOP_LEFT[1] + ROI_SIZE, ROI_TOP_LEFT[0]:ROI_TOP_LEFT[0] + ROI_SIZE]

    # Preprocess ROI and predict
    input_img = preprocess_frame(roi)
    prediction = model.predict(input_img)[0]

    # Add new prediction to history
    prediction_history.append(prediction)
    avg_prediction = np.mean(prediction_history, axis=0)

    confidence = np.max(avg_prediction) * 100
    pred_class = classes[np.argmax(avg_prediction)] if confidence > CONFIDENCE_THRESHOLD else "..."

    # Overlay prediction and confidence
    cv2.putText(frame, f"{pred_class} ({confidence:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.imshow('Improved Real-time ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

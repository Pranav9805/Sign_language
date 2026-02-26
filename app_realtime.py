import cv2
import numpy as np
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

ROI_TOP_LEFT = (100, 100)
ROI_SIZE = 200
CONFIDENCE_THRESHOLD = 70  # percent

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    # Draw ROI rectangle on frame
    cv2.rectangle(frame, ROI_TOP_LEFT, (ROI_TOP_LEFT[0] + ROI_SIZE, ROI_TOP_LEFT[1] + ROI_SIZE), (255, 0, 0), 2)

    # Crop ROI from frame
    roi = frame[ROI_TOP_LEFT[1]:ROI_TOP_LEFT[1] + ROI_SIZE, ROI_TOP_LEFT[0]:ROI_TOP_LEFT[0] + ROI_SIZE]

    # Preprocess the cropped ROI
    input_img = preprocess_frame(roi)
    prediction = model.predict(input_img)[0]
    confidence = np.max(prediction) * 100

    # Determine predicted class based on confidence threshold
    if confidence > CONFIDENCE_THRESHOLD:
        pred_class = classes[np.argmax(prediction)]
    else:
        pred_class = "..."

    # Overlay prediction text on the frame
    cv2.putText(frame, f"{pred_class} ({confidence:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-time ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

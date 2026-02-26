import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load architecture
with open('sign_language_model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create model from JSON
model = model_from_json(loaded_model_json)

# Load weights
model.load_weights('sign_language_model.weights.h5')

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
classes.remove('J')
classes.remove('Z')

st.title("Sign Language MNIST Gesture Recognition")
st.write("Upload a 28x28 grayscale image of an ASL hand gesture.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    img_array = img_to_array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    pred_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write(f"Predicted Sign: **{pred_class}** with confidence {confidence:.2f}%")

    st.write("Confidence scores for all classes:")
    for cls, score in sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True):
        st.write(f"{cls}: {score*100:.2f}%")



















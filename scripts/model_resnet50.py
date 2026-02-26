import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess
import numpy as np


def preprocess_images(X):
    print("Starting batch image resizing and channel conversion...")
    X_tf = tf.convert_to_tensor(X)  # Convert numpy array to tf tensor
    X_resized = tf.image.resize(X_tf, (224, 224))  # Batch resize all images at once
    X_rgb = tf.image.grayscale_to_rgb(X_resized)  # Convert 1 channel to 3 channels
    X_preprocessed = tf.keras.applications.resnet.preprocess_input(X_rgb)
    print("Finished batch preprocessing images.")
    return X_preprocessed.numpy()  # Convert back to numpy array if needed


def build_resnet50_finetune(num_classes=25):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train():
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess()

    print("Preprocessing images for ResNet50 input (fast batch)...")
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    print("Done preprocessing images.")

    model = build_resnet50_finetune()
    model.summary()

    print("Starting initial training...")
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))
    print("Initial training complete.")

    model.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting fine-tuning...")
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    print("Fine-tuning complete.")

    model.save('sign_language_resnet50_finetuned.keras')
    print("Model saved.")


if __name__ == "__main__":
    train()

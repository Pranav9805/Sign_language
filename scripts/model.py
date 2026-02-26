import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np


# Load and preprocess data
def load_and_preprocess():
    train_df = pd.read_csv('data/sign_mnist_train.csv')
    test_df = pd.read_csv('data/sign_mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values.reshape(-1,28,28,1).astype('float32') / 255.0
    y_train = to_categorical(train_df['label'].values, num_classes=25)

    X_test = test_df.drop('label', axis=1).values.reshape(-1,28,28,1).astype('float32') / 255.0
    y_test = to_categorical(test_df['label'].values, num_classes=25)

    return X_train, y_train, X_test, y_test


def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(25, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_save():
    X_train, y_train, X_test, y_test = load_and_preprocess()
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

    # Evaluate final validation accuracy
    val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final validation accuracy: {val_accuracy * 100:.2f}%")

    model.save('sign_language_model.keras')  # Save in Keras V3 format
    print("Model training complete and saved as sign_language_model.keras")


if __name__ == "__main__":
    train_and_save()

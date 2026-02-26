import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pandas as pd
from data_preprocessing import load_and_preprocess

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)
    cm = cm.numpy()
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_and_preprocess()

    # Load trained model
    model = load_model('../sign_language_model.h5')

    # Predict classes on test set
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Define class names for Sign Language MNIST (labels 0-24)
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # Remove J and Z which are not in dataset
    classes.remove('J')
    classes.remove('Z')

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes)

    # Optional: plot training history if saved
    # For this example, assume history was saved as 'history.npy'
    # Uncomment these lines if you saved training history
    # history = np.load('history.npy', allow_pickle=True).item()
    # plot_training_history(history)

from sklearn.metrics import classification_report
import numpy as np

# Assuming y_true and y_pred are defined as before:
# y_true = np.argmax(y_test, axis=1)
# y_pred = np.argmax(y_pred_prob, axis=1)

# Class names excluding J and Z
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
classes.remove('J')
classes.remove('Z')

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))

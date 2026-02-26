import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess

def preprocess_images_mobilenet(X):
    X_tf = tf.convert_to_tensor(X)
    X_resized = tf.image.resize(X_tf, (224, 224))
    X_rgb = tf.image.grayscale_to_rgb(X_resized)
    X_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(X_rgb)
    return X_preprocessed.numpy()

def build_mobilenetv2_finetune(num_classes=25):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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

    print("Preprocessing images for MobileNetV2 input...")
    X_train = preprocess_images_mobilenet(X_train)
    X_test = preprocess_images_mobilenet(X_test)
    print("Done preprocessing images.")

    model = build_mobilenetv2_finetune()
    model.summary()

    print("Starting initial training...")
    model.fit(X_train, y_train, epochs=3, batch_size=64,
              validation_data=(X_test, y_test))
    print("Initial training complete.")

    model.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting fine-tuning...")
    model.fit(X_train, y_train, epochs=10, batch_size=64,
              validation_data=(X_test, y_test))
    print("Fine-tuning complete.")

    model.save('sign_language_mobilenetv2_finetuned.keras')
    print("Model saved.")

if __name__ == "__main__":
    train()

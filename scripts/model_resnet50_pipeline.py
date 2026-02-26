import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess

def preprocess_single_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return image

def create_dataset(X, y, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(lambda x, y: (preprocess_single_image(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

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
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_and_preprocess()
    
    print("Creating datasets...")
    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_test, y_test, shuffle=False)
    
    model = build_resnet50_finetune()
    model.summary()
    
    print("Training model...")
    model.fit(train_dataset, epochs=3, validation_data=val_dataset)
    
    model.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Fine-tuning model...")
    model.fit(train_dataset, epochs=10, validation_data=val_dataset)
    
    model.save('sign_language_resnet50_finetuned.keras')
    print("Training complete and model saved.")

if __name__ == "__main__":
    train()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class AIImageClassifier:
    def __init__(self, img_height=224, img_width=224, channels=3):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.model = self.build_model()
        self.class_names = None

    def build_model(self):
        base_model = ResNet50V2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.img_height, self.img_width, self.channels)
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy', 
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        return model

    def prepare_data(self, data_dir):
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        self.class_names = list(train_generator.class_indices.keys())

        return train_generator, validation_generator

    def train(self, data_dir="input/AI-face-detection-Dataset", epochs=50):
        train_generator, validation_generator = self.prepare_data(data_dir)

        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        )

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping]
        )

        # Save the trained model
        self.model.save("model/face_classifier.h5")
        print("✅ Model trained and saved to model/face_classifier.h5")
        return history

# Example Usage
if __name__ == '__main__':
    classifier = AIImageClassifier(img_height=150, img_width=150)
    classifier.train("input/AI-face-detection-Dataset")
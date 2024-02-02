# -*- coding: utf-8 -*-
"""Untitled10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OkpIxO3os1Mw6gZhBiahOhIy7oKJG32v
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

class ShipClassifier:
    def __init__(self):
        self.train_data_dir = '/content/ships_dataset/train.csv'
        self.test_data_dir = '/content/ships_dataset/test.csv'

    def prepare_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_data_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )

    def create_model(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_generator, epochs=10, validation_data=self.test_generator)

    def transfer_learning(self):
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

        for layer in pretrained_model.layers:
            layer.trainable = False

        x = Flatten()(pretrained_model.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(10, activation='softmax')(x)

        self.model_transfer = Model(inputs=pretrained_model.input, outputs=predictions)
        self.model_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model_transfer.fit(self.train_generator, epochs=10, validation_data=self.test_generator)

    def evaluate_model(self):
        test_loss, test_accuracy = self.model_transfer.evaluate(self.test_generator)
        print(f"Test Doğruluğu: {test_accuracy}")

if __name__ == "__main__":
    ship_classifier = ShipClassifier()
    ship_classifier.prepare_data()
    ship_classifier.create_model()
    ship_classifier.transfer_learning()
    ship_classifier.evaluate_model()
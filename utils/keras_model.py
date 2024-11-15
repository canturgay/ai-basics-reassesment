import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # Use classification loss

class FaceEmbeddingModel:
    def __init__(self, input_shape=(224, 224, 3), embedding_dim=128):  # Use 224x224 as default input shape
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        base_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze the base model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.embedding_dim, activation='relu'),
            Dense(10, activation='softmax')  # Add a final classification layer with 10 classes
        ])
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])  # Use classification loss
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def create_embedding(self, image):
        return self.model.predict(np.expand_dims(image, axis=0))[0]

    def save_model(self, path):
        # Build the model with a dummy input before saving
        self.model(np.zeros((1, *self.input_shape)))
        self.model.save(path)  # Save in TensorFlow SavedModel format

    @classmethod
    def load_model(cls, path):
        instance = cls()
        instance.model = load_model(path)  # TensorFlow SavedModel format
        return instance

def train_or_load_model(root_folder, model_path):
    if os.path.exists(model_path):
        return FaceEmbeddingModel.load_model(model_path)
    else:
        print("No existing model found. Creating a new model...")
        model = FaceEmbeddingModel()
        model.save_model(model_path)
        return model
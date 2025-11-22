import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(64, 64, 3)):
    """
    Create a CNN model for binary classification (healthy vs infected)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output Layer (binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Test the model architecture
if __name__ == "__main__":
    model = create_cnn_model()
    model.summary()
    
    print("\n✅ Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
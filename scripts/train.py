import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# Import our custom modules
from data_preprocessing import DataPreprocessor
from model import create_cnn_model

class ModelTrainer:
    def __init__(self, dataset_path, img_size=64, epochs=20, batch_size=32):
        """
        Initialize the trainer
        
        Args:
            dataset_path: Path to dataset folder
            img_size: Image size for preprocessing
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None
        self.model = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        print("=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        preprocessor = DataPreprocessor(self.dataset_path, self.img_size)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = preprocessor.get_data()
        
    def build_model(self):
        """Build the CNN model"""
        print("\n" + "=" * 60)
        print("BUILDING MODEL")
        print("=" * 60)
        
        self.model = create_cnn_model(input_shape=(self.img_size, self.img_size, 3))
        self.model.summary()
        
    def train(self):
        """Train the model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        # Callbacks for better training
        callbacks = [
            # Early stopping if validation loss doesn't improve
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when stuck
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
    def save_model(self):
        """Save the trained model"""
        # Create saved_models directory if it doesn't exist
        save_dir = os.path.join('..', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f'malaria_cnn_{timestamp}.h5')
        
        self.model.save(model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        return model_path
    
    def plot_training_history(self):
        """Plot training and validation accuracy/loss"""
        # Create results directory if it doesn't exist
        results_dir = os.path.join('..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(results_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plot saved to: {plot_path}")
        
        plt.show()
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        self.load_data()
        self.build_model()
        self.train()
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"\n📊 Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"📊 Test Loss: {test_loss:.4f}")
        
        # Save model and plots
        self.save_model()
        self.plot_training_history()
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)


if __name__ == "__main__":
    # Configuration
    dataset_path = os.path.join('..', 'dataset')
    
    # Create trainer and run
    trainer = ModelTrainer(
        dataset_path=dataset_path,
        img_size=64,
        epochs=20,
        batch_size=32
    )
    
    trainer.run_complete_training()
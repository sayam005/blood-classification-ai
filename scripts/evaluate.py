import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import cv2
from datetime import datetime

from data_preprocessing import DataPreprocessor

class ModelEvaluator:
    def __init__(self, model_path, dataset_path, img_size=64):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to saved model (.h5 file)
            dataset_path: Path to dataset folder
            img_size: Image size used during training
        """
        self.model = keras.models.load_model(model_path)
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.results_dir = os.path.join('..', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_test_data(self):
        """Load and preprocess test data"""
        print("Loading test data...")
        preprocessor = DataPreprocessor(self.dataset_path, self.img_size)
        _, _, self.X_test, _, _, self.y_test = preprocessor.get_data()
        print(f"Test set size: {len(self.X_test)} images")
        
    def predict(self):
        """Make predictions on test set"""
        print("\nMaking predictions...")
        self.y_pred_prob = self.model.predict(self.X_test)
        self.y_pred = (self.y_pred_prob > 0.5).astype(int).flatten()
        
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Healthy', 'Infected'],
                    yticklabels=['Healthy', 'Infected'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy text
        accuracy = np.sum(self.y_test == self.y_pred) / len(self.y_test)
        plt.text(1, -0.3, f'Overall Accuracy: {accuracy*100:.2f}%', 
                ha='center', fontsize=12, fontweight='bold',
                transform=plt.gca().transAxes)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {path}")
        plt.show()
        
    def print_classification_report(self):
        """Print detailed classification metrics"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(self.y_test, self.y_pred, 
                                       target_names=['Healthy', 'Infected'],
                                       digits=4)
        print(report)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f'classification_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {report_path}")
        
    def visualize_predictions(self, num_samples=16):
        """Visualize sample predictions"""
        # Get random samples
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for idx, ax in enumerate(axes):
            i = indices[idx]
            
            # Display image
            img = self.X_test[i]
            ax.imshow(img)
            
            # Get prediction
            true_label = 'Infected' if self.y_test[i] == 1 else 'Healthy'
            pred_label = 'Infected' if self.y_pred[i] == 1 else 'Healthy'
            confidence = self.y_pred_prob[i][0] if self.y_pred[i] == 1 else 1 - self.y_pred_prob[i][0]
            
            # Color: green if correct, red if wrong
            color = 'green' if self.y_test[i] == self.y_pred[i] else 'red'
            
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)',
                        color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f'sample_predictions_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Sample predictions saved to: {path}")
        plt.show()
        
    def analyze_errors(self, num_errors=8):
        """Visualize misclassified samples"""
        # Find misclassified samples
        error_indices = np.where(self.y_test != self.y_pred)[0]
        
        if len(error_indices) == 0:
            print("\n🎉 No errors found! Perfect classification!")
            return
        
        print(f"\n⚠️ Found {len(error_indices)} misclassified samples")
        
        # Take random subset if too many errors
        if len(error_indices) > num_errors:
            error_indices = np.random.choice(error_indices, num_errors, replace=False)
        
        num_cols = 4
        num_rows = (len(error_indices) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ax in enumerate(axes.ravel()):
            if idx < len(error_indices):
                i = error_indices[idx]
                
                # Display image
                img = self.X_test[i]
                ax.imshow(img)
                
                # Get prediction details
                true_label = 'Infected' if self.y_test[i] == 1 else 'Healthy'
                pred_label = 'Infected' if self.y_pred[i] == 1 else 'Healthy'
                confidence = self.y_pred_prob[i][0] if self.y_pred[i] == 1 else 1 - self.y_pred_prob[i][0]
                
                ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)',
                            color='red', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f'error_analysis_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Error analysis saved to: {path}")
        plt.show()
        
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("="*60)
        print("STARTING MODEL EVALUATION")
        print("="*60)
        
        self.load_test_data()
        self.predict()
        
        self.plot_confusion_matrix()
        self.print_classification_report()
        self.visualize_predictions()
        self.analyze_errors()
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    # Update this with your saved model path
    model_path = os.path.join('..', 'saved_models', 'malaria_cnn_20251122_220933.h5')
    dataset_path = os.path.join('..', 'dataset')
    
    evaluator = ModelEvaluator(model_path, dataset_path, img_size=64)
    evaluator.run_complete_evaluation()
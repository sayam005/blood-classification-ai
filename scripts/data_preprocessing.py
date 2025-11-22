import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DataPreprocessor:
    def __init__(self, dataset_path, img_size=64):
        """
        Initialize the data preprocessor
        
        Args:
            dataset_path: Path to dataset folder containing infected/healthy subfolders
            img_size: Size to resize images to (default 64x64)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.X = []
        self.y = []
        
    def load_images(self):
        """Load images from infected and healthy folders"""
        print("Loading images...")
        
        # Load infected images (label = 1)
        infected_path = os.path.join(self.dataset_path, 'infected')
        count = 0
        for img_name in os.listdir(infected_path):
            img_path = os.path.join(infected_path, img_name)
            if os.path.isdir(img_path):
                continue
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.X.append(img)
                self.y.append(1)  # 1 = infected
                count += 1
        
        print(f"Loaded {count} infected images")
        
        # Load healthy images (label = 0)
        healthy_path = os.path.join(self.dataset_path, 'healthy')
        count = 0
        for img_name in os.listdir(healthy_path):
            img_path = os.path.join(healthy_path, img_name)
            if os.path.isdir(img_path):
                continue
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.X.append(img)
                self.y.append(0)  # 0 = healthy
                count += 1
        
        print(f"Loaded {count} healthy images")
        print(f"Total images loaded: {len(self.X)}")
        
    def preprocess(self):
        """Normalize and shuffle the data"""
        # Convert to numpy arrays
        self.X = np.array(self.X, dtype='float32')
        self.y = np.array(self.y)
        
        # Normalize pixel values to 0-1 range
        self.X = self.X / 255.0
        
        # Shuffle data
        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        
        print(f"Data shape: {self.X.shape}")
        print(f"Labels shape: {self.y.shape}")
        
    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data(self):
        """Complete pipeline: load, preprocess, and split data"""
        self.load_images()
        self.preprocess()
        return self.split_data()


# Test the preprocessor
if __name__ == "__main__":
    dataset_path = os.path.join('..', 'dataset')
    
    preprocessor = DataPreprocessor(dataset_path, img_size=64)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_data()
    
    print("\n✅ Data preprocessing complete!")
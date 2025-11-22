# Blood Cell Classification for Malaria Detection

A CNN-based system to classify red blood cell images as healthy or malaria-infected. Built as part of an automated medical diagnostics project.

## What This Does

Takes microscope images of blood cells and tells you if they're infected with malaria or not. Achieved 96% accuracy on test data.

## Project Structure

```
blood-classification-ai/
├── dataset/
│   ├── infected/
│   └── healthy/
├── scripts/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── saved_models/
├── results/
└── requirements.txt
```

## Getting Started

### What You Need

- Python 3.8 or newer
- At least 4GB RAM
- About 30 minutes for training (on CPU)

### Setup

```bash
# Clone or download this repo
git clone https://github.com/yourusername/blood-classification-ai.git
cd blood-classification-ai

# Make a virtual environment
python -m venv venv
venv\Scripts\activate

# Install stuff
pip install -r requirements.txt
```

### Getting the Dataset

Download the malaria cell images dataset from Kaggle:
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Extract it so you have:
- `dataset/infected/` with infected cell images
- `dataset/healthy/` with healthy cell images

Should have around 27,000 images total.

## Running It

### Training

```bash
cd scripts
python train.py
```

This will:
- Load all the images
- Train the model (takes 15-20 mins)
- Save the trained model
- Show you some plots

### Evaluation

```bash
python evaluate.py
```

Shows you:
- Confusion matrix
- Accuracy metrics
- Sample predictions
- Where the model got things wrong

## How It Works

The model is a CNN with:
- 3 convolutional blocks that learn to recognize patterns
- MaxPooling to reduce image size while keeping important features
- Dropout layers to prevent overfitting
- Final sigmoid layer that outputs infection probability

Images are resized to 64x64 and normalized before feeding to the model.

## Results

Got 96.03% accuracy on the test set. Pretty good considering some infected cells are hard even for experts to identify.

The model file is about 2.6MB, so it's lightweight enough to potentially run on mobile devices.

## Files Explained

- **data_preprocessing.py** - Loads images, resizes them, normalizes pixel values, splits into train/val/test
- **model.py** - Defines the CNN architecture
- **train.py** - Trains the model with early stopping and learning rate adjustment
- **evaluate.py** - Tests the model and generates visualizations

## Tech Stack

- TensorFlow/Keras for the neural network
- OpenCV for image processing
- NumPy/Pandas for data handling
- Matplotlib/Seaborn for plotting results
- Scikit-learn for evaluation metrics

## Things That Could Be Better

- Try different image sizes (maybe 128x128 for more detail?)
- Add data augmentation (rotate, flip images during training)
- Try transfer learning with pre-trained models
- Make a simple web interface
- Deploy it as a mobile app

## Notes

The `.gitignore` excludes the dataset and model files since they're too big for GitHub. You'll need to download the dataset separately and train the model yourself.

Training uses early stopping, so it might finish before 20 epochs if the model stops improving. That's normal.

## License

MIT License - do whatever you want with this code.

## Acknowledgments

Dataset from Kaggle, originally from NIH research on malaria detection.

Built this to explore how deep learning can help with medical diagnosis in areas where trained microscopists aren't available.
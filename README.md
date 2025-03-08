# CNN Potato Leaf Disease Detection

## Project Overview
  This project implements a Convolutional Neural Network (CNN) for detecting diseases in potato leaves from images. The system is designed to classify potato leaf images into different disease categories, helping farmers and agricultural specialists identify plant diseases early and take appropriate measures.

## Features
  - Image-based potato leaf disease detection
  - CNN architecture for high accuracy classification
  - Pre-trained model available for immediate use
  - Support for multiple disease categories

## Requirements
  - Python 3.6+
  - TensorFlow 2.x
  - Keras
  - NumPy
  - Matplotlib
  - OpenCV
  - Other dependencies listed in `requirement.txt`

## Installation

1. Clone the repository:
  ```bash
    git clone https://github.com/Posuza/potato-leaf-disease-detection.git
    cd potato-leaf-disease-detection
  ```
  or
    .dowload project zip folder 

  
2. Extract the provided zip file to access all project elements:

  
  Extract the zip file containing dataset and model
  unzip potato_disease_dataset.zip -d ./data
  
  
  3. Install the required dependencies:
  
   ```python
    pip install -r requirement.txt
   ```


## Usage

### Using the Pre-trained Model

  - The project includes a pre-trained model that can be used immediately:

```python
from model import load_model
from preprocessing import preprocess_image

# Load the pre-trained model
model = load_model('path/to/model')

# Preprocess your image
img = preprocess_image('path/to/image.jpg')

# Make prediction
prediction = model.predict(img)
Copy
Insert

Training a New Model
If you want to train the model with your own dataset:

from model import create_model, train_model
from data_loader import load_dataset

# Load and prepare your dataset
X_train, y_train, X_val, y_val = load_dataset('path/to/dataset')

# Create a new model
model = create_model()

# Train the model
train_model(model, X_train, y_train, X_val, y_val, epochs=50)
```

### Dataset
 1. The dataset contains images of potato leaves with various diseases, including:

    - Early Blight
    - Late Blight
    - Healthy leaves
The dataset is already included in the project and will be available after extracting the provided zip file.

### Model Architecture
  - The CNN architecture used in this project consists of:

    - Multiple convolutional layers with ReLU activation
    - MaxPooling layers for downsampling
    - Dropout layers to prevent overfitting
    - Dense layers for classification
    - Softmax output layer for multi-class classification
    - Results
### The model achieves an accuracy of approximately X% on the test dataset, making it reliable for real-world applications.

  - Future Improvements
  - Mobile application integration
  - Real-time detection using camera feed
  - Expansion to other crop diseases
  - Deployment as a web service
  - 
## Contact
   - posu0009@gmail.com

## Acknowledgments
  - Thanks to all contributors who have helped shape this project
  - Special thanks to the Python community for excellent libraries and tools


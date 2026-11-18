
# VisionScene AI — CNN-Based Scene Image Classification

**VisionScene AI** is a deep learning and computer vision application that classifies images into six natural and urban scene categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The project includes a complete image classification workflow, from dataset preparation and CNN training to model evaluation and an interactive Streamlit web application.

## Project Highlights

| Feature | Details |
|---|---|
| Project Name | VisionScene AI |
| Domain | Deep Learning, Computer Vision |
| Algorithm | Convolutional Neural Network (CNN) |
| Dataset | Intel Image Classification |
| Classes | 6 |
| Test Images | 3,000 |
| Reported Test Accuracy | 81% |
| Framework | TensorFlow, Keras |
| Frontend | Streamlit |
| Programming Language | Python |

**Model Performance:** The training notebook reports approximately **81% test accuracy on 3,000 images**.

**Note:** The separately supplied deployment model has a different CNN architecture from the training notebook. Its test accuracy has not yet been independently verified.

---

## 1. Problem Statement

Image classification is an important application of computer vision that enables machines to recognize and categorize visual information automatically.

Manually categorizing large collections of images is time-consuming.

VisionScene AI addresses this problem by applying deep learning techniques to classify images into six scene categories.

### Objectives

- Build a CNN-based multi-class image classification model.
- Process and prepare image datasets for deep learning.
- Train the model to recognize visual features.
- Evaluate classification performance using test data.
- Create an interactive web application.
- Display predicted classes and confidence scores.
- Allow users to download classified images.

---

## 2. Dataset Description

### Dataset

**Intel Image Classification Dataset**

Source:

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset contains images of natural landscapes and urban environments.

### Image Categories

| Class ID | Category |
|---|---|
| 0 | Buildings |
| 1 | Forest |
| 2 | Glacier |
| 3 | Mountain |
| 4 | Sea |
| 5 | Street |

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Training | 11,228 |
| Validation | 2,806 |
| Testing | 3,000 |
| Unlabeled Prediction Images | 7,301 |

The training and validation datasets were created using an 80:20 split from 14,034 labeled training images.

The separate test dataset contains 3,000 labeled images.

### Dataset Configuration

- Image Size: 256 × 256 pixels
- Image Channels: RGB
- Batch Size: 32
- Validation Split: 20%
- Random Seed: 42
- Classification Type: Multi-class classification

---

## 3. Technologies Used

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)

### Data Processing
- NumPy
- Pandas
- Pillow

### Machine Learning Evaluation
- Scikit-learn
- Accuracy
- Precision
- Recall
- F1-Score

### Web Application
- Streamlit

### Additional Libraries
- Requests
- Matplotlib

---

## 4. Complete Project Workflow

The project follows the workflow:

Dataset Collection
       |
       v
Dataset Understanding
       |
       v
Image Preprocessing
       |
       v
Training / Validation Split
       |
       v
CNN Model Building
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
Image Classification
       |
       v
Streamlit Integration
       |
       v
User Image Input
       |
       v
Prediction and Confidence Scores
       |
       v
Download Classified Image

---

## 5. Step 1 — Dataset Collection

The Intel Image Classification dataset was downloaded from Kaggle.

The dataset includes labeled images for model training and testing.

Typical dataset structure:

```text
intel-image-classification/
│
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│
└── seg_pred/
    └── seg_pred/
```

The training folders contain labeled images.

The testing folders contain separate labeled images for performance evaluation.

The prediction directory contains unlabeled images that can be used for inference.

---

## 6. Step 2 — Image Preprocessing

Image preprocessing prepares input images for CNN training and prediction.

### Operations

**1. Image Resizing**

All images are resized to:

```python
image_size = (256, 256)
```

This ensures that images have consistent dimensions.

**2. RGB Conversion**

Images are converted to three-channel RGB format.

**3. Pixel Normalization**

Pixel values are normalized from 0–255 to 0–1.

```python
image = image / 255.0
```

Normalization helps maintain a consistent numerical scale during model training.

**4. Batch Preparation**

Image data is converted into batches of 32 samples during training.

For inference, the application converts one image into a batch of shape:

```text
(1, 256, 256, 3)
```

### Application Preprocessing

```python
def preprocess_image(image):
    image = image.convert("RGB").resize((256, 256))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)
```

---

## 7. Step 3 — Data Augmentation

Data augmentation creates transformations of images to improve model generalization.

The project materials include:

- Random Horizontal Flipping
- Random Rotation
- Random Zooming

These techniques help expose the CNN to different visual orientations.

### Important Implementation Detail

The training notebook defines an augmentation pipeline but does not apply that pipeline in its recorded training workflow.

The uploaded deployment model includes augmentation layers directly in its CNN architecture.

These differences should be considered when reproducing training results.

---

## 8. Step 4 — CNN Model Architecture

A Convolutional Neural Network learns visual features directly from image pixels.

CNNs use convolutional filters to identify patterns such as:

- Edges
- Shapes
- Textures
- Object structures
- Landscape patterns

### Training Notebook Architecture

```text
Input Image (256 × 256 × 3)
         |
         v
Conv2D (32 Filters, ReLU)
         |
         v
MaxPooling2D
         |
         v
Conv2D (64 Filters, ReLU)
         |
         v
MaxPooling2D
         |
         v
Conv2D (128 Filters, ReLU)
         |
         v
MaxPooling2D
         |
         v
Conv2D (256 Filters, ReLU)
         |
         v
MaxPooling2D
         |
         v
Flatten
         |
         v
Dense (128, ReLU)
         |
         v
Dense (64, ReLU)
         |
         v
Dense (6, Softmax)
         |
         v
Predicted Scene
```

The notebook model summary reports approximately **6.82 million parameters**.

### CNN Layer Explanation

**Convolutional Layers**

Extract useful visual features from images.

**ReLU Activation**

Introduces nonlinearity into the neural network.

**MaxPooling Layers**

Reduce feature-map dimensions while retaining important visual information.

**Flatten Layer**

Converts extracted feature maps into a one-dimensional feature vector.

**Dense Layers**

Learn combinations of extracted features for classification.

**Softmax Output Layer**

Produces six classification scores corresponding to the six scene categories.

---

## 9. Saved Deployment Model Architecture

The uploaded `intel_image_cnn.keras` file contains a different CNN architecture.

```text
Input Image
     |
     v
RandomFlip
     |
     v
RandomRotation
     |
     v
RandomZoom
     |
     v
Conv2D (32)
BatchNormalization
MaxPooling2D
     |
     v
Conv2D (64)
BatchNormalization
MaxPooling2D
     |
     v
Conv2D (128)
BatchNormalization
MaxPooling2D
     |
     v
Conv2D (256)
BatchNormalization
MaxPooling2D
     |
     v
GlobalAveragePooling2D
     |
     v
Dense (128)
Dropout (0.4)
     |
     v
Dense (64)
Dropout (0.3)
     |
     v
Dense (6, Softmax)
```

### Additional Components

**Batch Normalization**

Normalizes intermediate layer activations and can improve training stability.

**Global Average Pooling**

Reduces feature maps into compact feature representations.

**Dropout**

Regularizes the model by randomly disabling some activations during training.

The saved model has approximately **431,878 parameters**, based on its layer configuration.

The saved model is loaded by the Streamlit application.

---

## 10. Step 5 — Model Training

The notebook uses TensorFlow and Keras for model training.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Maximum Epochs | 20 |
| Batch Size | 32 |
| Early Stopping | Enabled |
| Monitor | Validation Loss |
| Patience | 3 |

### Adam Optimizer

The Adam optimizer adjusts network weights based on gradients during training.

### Sparse Categorical Crossentropy

This loss function is suitable for multi-class classification when target labels are represented as integers.

### Early Stopping

Early stopping monitors validation loss and can stop training when improvement stalls.

The notebook specifies restoration of the best recorded weights.

The available notebook training log is incomplete, so the exact number of completed epochs cannot be reliably confirmed.

---

## 11. Step 6 — Model Evaluation

The CNN was evaluated using labeled images from the Intel test dataset.

### Reported Test Performance

| Metric | Reported Result |
|---|---:|
| **Test Accuracy** | **81%** |
| Macro Precision | 82% |
| Macro Recall | 81% |
| Macro F1-Score | 81% |
| Weighted Precision | 82% |
| Weighted Recall | 81% |
| Weighted F1-Score | 81% |
| Test Images | 3,000 |

These results are transcribed from the classification report stored in the notebook.

### Class-Wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buildings | 0.75 | 0.81 | 0.78 |
| Forest | 0.90 | 0.97 | 0.93 |
| Glacier | 0.86 | 0.68 | 0.76 |
| Mountain | 0.68 | 0.84 | 0.75 |
| Sea | 0.84 | 0.81 | 0.82 |
| Street | 0.86 | 0.77 | 0.81 |

### Performance Analysis

- Forest achieved the highest class-wise F1-score of 0.93.
- Glacier showed lower recall compared with other classes.
- Mountain showed relatively lower precision.
- The reported overall accuracy was approximately 81%.

### Accuracy Interpretation

Test accuracy measures the proportion of correct predictions on the labeled test dataset.

An 81% test accuracy indicates that approximately 81% of the evaluated images were correctly classified.

It does not mean every individual prediction has an 81% probability of being correct.

### Evaluation Limitation

The training notebook contains a recorded classification report showing 81% accuracy, but its separate model evaluation cell has an execution error.

Because the notebook and saved deployment model use different architectures, the test performance of the deployed model must be verified separately.

---

## 12. Step 7 — Model Integration

The trained model is loaded into the Streamlit application using Keras.

```python
from tensorflow.keras.models import load_model

MODEL_PATH = "intel_image_cnn.keras"

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)
```

Caching avoids unnecessarily reloading the model during application reruns.

### Prediction Process

1. Receive an image.
2. Convert the image into RGB format.
3. Resize it to 256 × 256.
4. Normalize pixel values.
5. Add a batch dimension.
6. Load the saved CNN model.
7. Generate prediction scores.
8. Identify the highest-scoring scene category.
9. Display classification results.

---

## 13. Step 8 — Streamlit Web Application

VisionScene AI provides an interactive interface built with Streamlit.

The application is divided into three main stages:

**Stage 1 — Load Image**

**Stage 2 — Preview and Predict**

**Stage 3 — Display Results**

### Application Features

#### 1. Image Upload

Users can upload images from their devices.

Supported formats:

- JPG
- JPEG
- PNG
- WEBP

#### 2. Local Image Path

Users can provide a local image path.

This works when the application can access the specified path on the computer or server running Streamlit.

#### 3. Image URL

Users can provide a direct image URL.

The application fetches the image and prepares it for classification.

#### 4. Image Preview

Users can review the selected image before prediction.

The application also displays the image resolution.

#### 5. CNN Prediction

The application processes the image using the saved CNN model.

#### 6. Confidence Score

The application displays the top predicted category and its Softmax score.

#### 7. Top-Three Predictions

The three highest-scoring categories are displayed with their percentages.

#### 8. Mixed-Scene Estimate

The application can display a secondary category when the top two model scores meet predefined conditions.

#### 9. Downloadable Results

Users can download the classified image with its predicted scene label.

#### 10. Dark and Light Themes

Users can switch between dark and light interface themes.

---

## 14. Prediction Confidence System

The application divides predictions into three display categories.

| Model Score | Display Status |
|---|---|
| 80% or Above | High Confidence |
| 60% to Below 80% | Moderate Confidence |
| Below 60% | Low Confidence |

These thresholds are application display rules.

Softmax scores should not automatically be interpreted as statistically calibrated confidence.

---

## 15. Mixed-Scene Detection

The application includes a heuristic for displaying a secondary scene.

A mixed-scene estimate is displayed when:

```python
secondary_probability >= 0.20
```

and:

```python
primary_probability - secondary_probability <= 0.30
```

For example, an image could display:

```text
Primary Scene: Mountain
Secondary Scene: Sea

Displayed Result:
Mountain + Sea
```

### Important Note

The underlying CNN is a single-label Softmax classifier.

The mixed-scene feature interprets the two highest model scores. It does not convert the network into a true multi-label classification model.

---

## 16. Downloadable Prediction Results

After image classification, the application creates an annotated image.

The annotated image contains the predicted scene label.

Users can download the result as a PNG file.

Example:

```text
Input Image
     |
     v
CNN Prediction
     |
     v
Predicted Scene: Forest
     |
     v
Add Scene Label
     |
     v
Download PNG
```

---

## 17. Project Folder Structure

```text
VisionScene-AI/
│
├── app.py
│
├── intel_image_cnn.keras
│
├── CNN_Buildings (2).ipynb
│
├── requirement.txt
│
└── README.md
```

### File Description

**app.py**

Contains the Streamlit application, image processing and prediction workflow.

**intel_image_cnn.keras**

Saved Keras CNN model used for application inference.

**CNN_Buildings (2).ipynb**

Contains dataset preparation, CNN development, training and evaluation.

**requirement.txt**

Contains dependencies for running the application.

**README.md**

Contains complete project documentation.

---

## 18. Installation Guide

### Prerequisites

- Python 3 compatible with TensorFlow 2.21
- pip
- Internet connection for installation
- Project files

### Step 1 — Clone Repository

```bash
git clone <your-repository-url>
```

### Step 2 — Open Project Folder

```bash
cd VisionScene-AI
```

### Step 3 — Create Virtual Environment

```bash
python -m venv .venv
```

### Step 4 — Activate Virtual Environment

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### Step 5 — Install Dependencies

```bash
pip install -r requirement.txt
```

### Step 6 — Start Streamlit

```bash
streamlit run app.py
```

### Step 7 — Open Application

Streamlit normally provides a local URL such as:

```text
http://localhost:8501
```

Use the actual URL shown in your terminal.

---

## 19. How to Use VisionScene AI

### Step 1

Open the Streamlit application.

### Step 2

Choose an image input method:

- Upload Image
- Local Path
- Image URL

### Step 3

Load the image.

### Step 4

Check the image preview.

### Step 5

Click **Predict Scene**.

### Step 6

View the prediction results:

- Predicted Scene
- Model Score
- Confidence Status
- Top-Three Predictions
- Optional Secondary Scene

### Step 7

Download the classified image.

---

## 20. Application Prediction Workflow

```text
User Opens VisionScene AI
          |
          v
Select Image Source
          |
          v
Upload / Path / URL
          |
          v
Load Image
          |
          v
Preview Image
          |
          v
Click Predict
          |
          v
Convert to RGB
          |
          v
Resize to 256 × 256
          |
          v
Normalize Pixels
          |
          v
Load CNN Model
          |
          v
Generate Six Class Scores
          |
          v
Rank Predictions
          |
          v
Display Predicted Class
          |
          v
Display Confidence Score
          |
          v
Display Top-Three Classes
          |
          v
Create Annotated Image
          |
          v
Download Result
```

---

## 21. Applications and Use Cases

VisionScene AI demonstrates deep learning techniques applicable to:

- Scene recognition
- Automated image categorization
- Landscape image classification
- Image organization
- Computer vision experimentation
- Deep learning educational projects
- CNN-based web applications

The current implementation is a demonstration system rather than a production-validated image classification service.

---

## 22. Current Limitations

### 1. Model Architecture Difference

The training notebook and saved deployment model contain different CNN architectures.

Their training histories should not be assumed to be identical.

### 2. Unverified Deployment Accuracy

The notebook reports approximately 81% test accuracy.

The saved deployment model requires independent evaluation on labeled test images.

### 3. Limited Scene Categories

The model supports six predefined categories.

### 4. Single-Label Classification

The CNN uses a six-class Softmax output.

Mixed-scene estimates are heuristic and not true multi-label predictions.

### 5. Image Quality

Performance may vary with image resolution, lighting, composition and unfamiliar visual environments.

### 6. Confidence Calibration

The displayed percentages are model Softmax scores and have not been independently calibrated.

### 7. Public Deployment Security

The application accepts image URLs and server-local paths.

Before public deployment, URL fetching should be protected against server-side request forgery (SSRF), server-local file access should be restricted, and resource-size limits should be added.

### 8. Training Reproducibility

The notebook requires cleanup of some execution issues before it can serve as a fully reproducible training pipeline.

---

## 23. Future Improvements

Potential improvements include:

- Evaluate the exact deployment model on the held-out test set.
- Generate and analyze a confusion matrix.
- Add model performance charts.
- Experiment with transfer learning using pretrained CNN architectures.
- Improve model generalization through verified augmentation experiments.
- Add prediction history.
- Implement model confidence calibration.
- Optimize prediction speed.
- Add automated inference testing.
- Deploy the application with appropriate security restrictions.
- Support additional image categories.

---

## 24. Project Outcomes

VisionScene AI demonstrates practical experience in:

- Deep Learning
- Convolutional Neural Networks
- Image Classification
- Data Preprocessing
- Model Training
- Model Evaluation
- TensorFlow and Keras
- Streamlit Application Development
- Model Integration
- Computer Vision

### Reported Results

**Test Accuracy: 81%**

**Evaluation Dataset: 3,000 Images**

**Classification Categories: 6**

The above accuracy comes from the recorded training notebook classification report and should not be represented as independently verified accuracy for the separately supplied deployment model.

---

## 25. Resume Description

**VisionScene AI — CNN-Based Image Classification**

*Python | TensorFlow | Keras | CNN | Streamlit*

- Developed a CNN model to classify images into six scene categories using the Intel Image Classification dataset.
- Reported **81% test accuracy on 3,000 images** in the training notebook and built a Streamlit application featuring image uploads, confidence scores, top-three predictions and downloadable results.

---

## 26. Conclusion

VisionScene AI is an end-to-end computer vision project demonstrating the development and integration of Convolutional Neural Networks for scene classification.

The project covers image preprocessing, CNN architecture, model training, classification evaluation and interactive application development.

The training notebook reports approximately **81% classification accuracy on 3,000 test images**, while the Streamlit application provides a user-friendly interface for classifying new images.

Further validation of the separately supplied deployment model will help establish a reproducible, independently verified performance benchmark.

---

## Author

**Mahima Adhale**

Computer Science and Engineering Graduate

Interested in Data Science, Machine Learning, Deep Learning, Computer Vision and Generative AI.

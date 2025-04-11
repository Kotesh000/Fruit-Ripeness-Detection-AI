# Fruit Ripeness Detection using AI

**Fruit Ripeness Detection** is a machine learning project that aims to classify tropical fruits into **Fresh**, **Rotten**, and **Unripe** categories based on their images. For accurate predictions, this project uses convolutional neural networks (CNNs) and transfer learning with **MobileNetV2**. It is designed for real-time and image-based classification, which can be applied in agriculture, food processing, and smart farming.

This repository contains all the code, model files, dataset instructions, and documentation needed to run, reproduce, and extend the ripeness detection system.

---

## **Features**

### **Data Collection**
- Collected tropical fruit images representing three categories: **fresh**, **rotten**, and **unripe**.
- Dataset structured into subfolders for classification training.
- Stored on Google Drive for easy access and reproducibility.

### **Data Preprocessing**
- Image resizing and normalization.
- Image data augmentation (rotation, zoom, flips) using Keras `ImageDataGenerator`.
- Organized data into training, validation, and test sets.

### **Exploratory Data Analysis (EDA)**
- Visual analysis of sample images across all three classes.
- Distribution checks to ensure class balance.
- Insights into augmentation effects for improving model generalization.

### **Model Training**
- Implemented a Convolutional Neural Network (CNN) using:
  - **MobileNetV2** as a pre-trained base model (Transfer Learning).
  - Custom top layers for classification.
- Fine-tuned for accuracy and performance on limited datasets.

### **Model Evaluation**
- Used metrics for classification evaluation:
  - Accuracy  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-score)  
- Trained model saved in `.h5` format for easy reuse.

### **Prediction**
- **Real-time prediction** using OpenCV and webcam.
- **Static image prediction** using command-line tools.
- Displays predicted class (**Fresh**, **Rotten**, **Unripe**) on screen or image.

---

## **⚙️ Usage**

### **Step 1: Clone the repository**
```bash
git clone https://github.com/Kotesh000/Fruit-Ripeness-Detection-AI.git
cd Fruit-Ripeness-Detection-AI
```

### **Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Download and organize dataset**
- Download dataset from Google Drive: [Click here](https://drive.google.com/file/d/1_76dooEYFE6Ku-D_QyX1jLe_f_Pr0mgI/view?usp=sharing)
- Extract it and place in a folder named `dataset/` with subfolders:
```
dataset/
├── fresh/
├── rotten/
└── unripe/
```

### **Step 4: Train the model**
```bash
python fruit_classifier.py
```

### **Step 5: Predict using an image**
```bash
python predict_image.py --image path_to_your_image.jpg
```

### **Step 6: Real-time webcam prediction**
```bash
python webcam_predict.py
```

---

## **Requirements**

List of Python libraries used (also in `requirements.txt`):

- tensorflow  
- keras  
- opencv-python  
- numpy  
- matplotlib  
- gdown  

---

## **Outcomes**

- Successfully classified fruits into **Fresh**, **Rotten**, and **Unripe** categories.
- Achieved good accuracy on validation and test datasets.
- Real-time webcam prediction works effectively with unseen fruit samples.
- The model generalizes well across different lighting conditions and angles due to data augmentation.

---

## **Contributing**

Contributions are always welcome!  
Feel free to **fork the repository**, make your changes, and **submit a pull request**.

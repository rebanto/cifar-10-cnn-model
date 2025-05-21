# CIFAR-10 CNN Classifier 🧠📷

This repository contains a simple convolutional neural network (CNN) implementation for image classification on the CIFAR-10 dataset using PyTorch. The notebook is designed to be beginner-friendly, easy to run, and fully self-contained.

---

## 🚀 Features

* Trains a custom CNN model on the CIFAR-10 dataset
* **Uses**:
    * Early stopping to prevent overfitting
    * Learning rate scheduler for better convergence
* **Saves**:
    * Final model weights (`.pth`)
    * Training loss curve
    * Confusion matrix
    * Classification report
* **Evaluation metrics**:
    * Accuracy
    * Precision, Recall, F1-score per class
* Clean and reproducible Jupyter Notebook

---

## 📊 Dataset

CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training and 10,000 test images.

---

## 🧱 CNN Architecture

This version uses a simple 3-layer CNN structure with ReLU activations, max pooling, dropout, and a dense classifier head.

---

## 🔍 Results

Example performance (may vary slightly based on run):

* **Accuracy**: ~79%
* **Best-performing classes**: car, ship, truck
* **Most confusion**: between cat, dog, and bird

Detailed metrics and plots are saved to the output files.

---

## 🛠️ Setup

1.  Clone the repository
2.  Install dependencies (preferably in a virtual environment):

    ```bash
    pip install -r requirements.txt
    ```

3.  Launch Jupyter:

    ```bash
    jupyter notebook
    ```

4.  Open and run `CIFAR10_CNN_Classifier.ipynb` cell by cell or use "Run All"

---

## 📁 Outputs

* `cifar10_cnn_model.pth`: Trained model
* `loss_curve.png`: Training loss vs. epoch
* `confusion_matrix.png`: Confusion matrix plot
* `classification_report.txt`: Precision/Recall/F1 for all classes

---

## 🔮 Next Steps

In future versions, this repo will be updated to:

* Use deeper custom CNN architectures
* Integrate pretrained models like ResNet or EfficientNet (transfer learning)
* Add data augmentation and hyperparameter tuning

Stay tuned!

---

## 📜 License

This project is open-source and available under the MIT License.

# Intel Unnati Final Submission - Team Intellidians

## PS-13: Vehicle Movement Analysis and Insight Generation in a College Campus using Edge AI

### Intelligent Transportation and Surveillance System

This project involves developing an intelligent transportation and surveillance system using deep learning and computer vision techniques. The system comprises multiple modules, including vehicle classification, number plate detection, vehicle movement analysis, and parking lot occupancy monitoring. The aim is to enhance traffic management, improve security, and optimize parking space utilization within a college campus.

### Table of Contents

1. [Project Overview](#project-overview)
2. [Datasets Used](#datasets-used)
3. [Key Features](#key-features)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Edge Deployment](#edge-deployment)
10. [Contributing](#contributing)
11. [License](#license)
12. [References](#references)
13. [Google Colab](#google-colab)

### Project Overview

This project is part of the Intel Unnati program, aimed at leveraging edge AI for intelligent transportation and surveillance within a college campus. It involves developing a system using deep learning and computer vision techniques to enhance traffic management, improve security, and optimize parking space utilization.

### Datasets Used

- **Stanford Cars Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)  
  Citation: Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 4th IEEE Workshop on 3D Representation and Recognition, ICCV 2013, Sydney, Australia.

- **Pklot Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset)  
  Citation: Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., & Koerich, A. (2015). PKLot – A robust dataset for parking lot classification, Expert Systems with Applications.

### Key Features

- **Vehicle Matching Analysis:** Uses CNN models (VGG16, ResNet) for vehicle classification based on image features.
- **Number Plate Detection:** Implements Haar cascade classifiers and EasyOCR for accurate number plate detection and recognition.
- **Vehicle Movement Analysis:** Fusion model of EasyOCR and Haar cascade for identifying vehicle number plates and timestamping movements.
- **Parking Lot Analysis:** YOLOv5 object detection model for real-time monitoring of parking lot occupancy.

### Project Structure

- **Introduction:** Downloads necessary libraries, imports modules, and connects to Google Drive for dataset access.
- **Loading the Dataset:** Converts annotations from MATLAB to CSV, loads dataset annotations, and accesses training/testing directories.
- **Exploratory Data Analysis:** Creates dataframes, performs statistical analysis, applies data augmentation, and visualizes sample images.
- **Data Preprocessing:** Preprocesses images, visualizes preprocessed images, and labels/encodes data for model training.
- **Vehicle Matching Analysis:** Implements CNN models (VGG16, ResNet) for vehicle classification and evaluates model performance.
- **Number Plate Detection:** Uses Haar cascade model for number plate detection and validates detections.
- **Vehicle Movement Analysis:** Integrates EasyOCR and Haar cascade models for vehicle movement analysis and timestamping.
- **Parking Lot Analysis:** Loads and preprocesses Pklot dataset, trains YOLOv5 model for parking lot occupancy detection.
- **Edge Deployment:** Optimizes models for edge devices, including vehicle matching, vehicle movement analysis, and parking lot occupancy detection.

### Prerequisites

- **Google Colab:** Used for executing Python code in a cloud environment and accessing datasets stored in Google Drive.
- **Hardware:** Edge device with support for TensorFlow Lite (e.g., Raspberry Pi) (*to be implemented)
- **Software:** Python 3.x, TensorFlow, OpenCV, EasyOCR, Pandas, NumPy, Matplotlib, YOLOv5
- **Haar Cascade:** Haar cascade XML file - "https://github.com/spmallick/mallick_cascades/blob/master/haarcascades/haarcascade_russian_plate_number.xml"
- **YOLOv5:** "https://pytorch.org/hub/ultralytics_yolov5/"

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ArunyaPaul/ntel-Unnati-Final-Submission-PS-13.git
   cd ntel-Unnati-Final-Submission-PS-13
2. Install the required dependencies:
   pip install -r requirements.txt
3. Download the necessary datasets:
   -Stanford Cars Dataset
   -Pklot Dataset
4. Download the Haar cascade XML file.
5. Clone the YOLOv5 repository and install the required dependencies.

### Usage

1. Dataset Preparation:
   
  - Stanford Cars Dataset:
   
   Download from Kaggle.
   Extract the dataset and upload in Google Drive. The folder structure should be:
   stanford_cars/
    cars_annos.mat
    cars_test/
        cars_test/
            (car images)
    cars_train/
        cars_train/
            (car images)
   
  - Pklot Dataset:
   
   Download from Kaggle.
   Extract the dataset and upload in Google Drive. The folder structure should be:
   pklot/
    test/
        _annotations.coco.json
        (images)
    train/
        _annotations.coco.json
        (images)
    valid/
        _annotations.coco.json
        (images)

2. Google Colab Integration:
   
   -Mount Google Drive to access datasets.
   -Access datasets from Google Drive in your Colab environment.
   
3. Haar Cascade Installation:
   -Download Haar cascade XML file - 'haarcascade_russian_plate_number.xml'.
   -Upload the XML file to your Google Colab environment or edge device.

4.  Clone the YOLOv5 repository and install the required dependencies:

    ```bash
    rm -rf yolov5
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt
    ```

    Use YOLOv5 models for vehicle detection tasks.

5. Running the Code:
   -Ensure all libraries and dependencies are installed.
   -Follow the instructions in each script to load datasets, preprocess data, train models, and deploy the system.

### Model Training and Evaluation

Follow the instructions in the project structure to train and evaluate models for vehicle classification, number plate detection, vehicle movement analysis, and parking lot occupancy monitoring.

### Edge Deployment

TensorFlow Lite can be used to deploy models on edge devices. Ensure models are optimized for performance and accuracy on the target hardware. A rough code example for edge deployment is provided in the Python notebook, and a similar Flask example code under the section "Sample Code for WebDev Application in Flask" has also also included for future application. These codes generalised examples and are meant to be fine-tuned based on your specific requirements and hardware.

### Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements. As an electronics student and a novice in AI and ML (Computer Vision), I appreciate any help and guidance from experienced peers.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### References

- Stanford Cars Dataset
- Pklot Dataset
- Haar Cascade XML Files
- YOLO5 Model:https://pytorch.org/hub/ultralytics_yolov5/
- Haarcasscade Model: https://github.com/spmallick/mallick_cascades/blob/master/haarcascades/haarcascade_russian_plate_number.xml
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- YouTube Video: https://www.youtube.com/watch?v=fyJB1t0o0ms&t=389s
- YouTube Playlist: https://www.youtube.com/watch?v=Z78zbnLlPUA&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&index=2
- Chat GPT and Google Bard (for syntax references and debugging)

### Google Colab

Here's the link for the Google Colab Python Notebook: https://colab.research.google.com/drive/1hlNuFKMwPJZ9O75c7NlgXzcF_q7Qv5kI?usp=sharing

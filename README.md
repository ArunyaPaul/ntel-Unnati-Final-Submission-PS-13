# Intel-Unnati-Final-Submission-PS-13
Intel Unnati Final Submission - Team Intellidians - (PS-13: Vehicle Movement Analysis and Insight Generation in a College Campus using Edge AI)

## Intelligent Transportation and Surveillance System:
This repository contains code for an intelligent transportation and surveillance system developed using deep learning and computer vision techniques. The system includes modules for vehicle classification, number plate detection, vehicle movement analysis, and parking lot occupancy monitoring. It is designed to enhance traffic management, improve security, and optimize parking space utilization.

## Datasets Used:

-> Stanford Cards Dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset;
Citation: Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei, 4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

-> Pklot Dataset: https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset;
Citation: Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A., PKLot – A robust dataset for parking lot classification, Expert Systems with Applications, 42(11):4937-4949, 2015

##  Key Features:

->Vehicle Matching Analysis: Utilizes CNN models (VGG16, ResNet) for vehicle classification based on image features.

->Number Plate Detection: Implements Haar cascade classifiers and EasyOCR for accurate number plate detection and recognition.

->Vehicle Movement Analysis: Fusion model of EasyOCR and Haar cascade for identifying vehicle number plates and timestamping movements.

->Parking Lot Analysis: YOLOv5 object detection model for real-time monitoring of parking lot occupancy.

## Structure:

->Introduction: Downloads necessary libraries, imports modules, and connects to Google Drive for dataset access.

->Loading the Dataset (Stanford Cars): Converts annotations from MATLAB to CSV, loads dataset annotations, and accesses training/testing directories.

->Exploratory Data Analysis: Creates dataframes, performs statistical analysis, applies data augmentation, and visualizes sample images.

->Data Preprocessing: Preprocesses images, visualizes preprocessed images, and labels/encodes data for model training.

->Vehicle Matching Analysis: Implements CNN models (VGG16, ResNet) for vehicle classification and evaluates model performance.

->Number Plate Detection: Uses Haar cascade model for number plate detection and validates detections.

->Vehicle Movement Analysis: Integrates EasyOCR and Haar cascade models for vehicle movement analysis and timestamping.

->Parking Lot Analysis: Loads and preprocesses Pklot dataset, trains YOLOv5 model for parking lot occupancy detection.

->Edge Deployment: Optimizes models for edge devices, including vehicle matching, vehicle movement analysis, and parking lot occupancy detection.

## Deployment:

->For deployment on edge devices: TensorFlow Lite is used

->Vehicle Matching Analysis: Deploy optimized CNN models (VGG16, ResNet) for vehicle classification. ResNet50 proved to be best performing and suitable for deployment.

->Vehicle Movement Analysis: Deploy combined EasyOCR and Haar cascade models for real-time vehicle tracking and timestamp logging.

->Parking Lot Analysis: Deploy YOLOv5 model on edge devices for monitoring parking lot occupancy.

## Requirements:

->Python 3.x

->TensorFlow, OpenCV, EasyOCR, Pandas, NumPy, Matplotlib

->YOLOv5, Haar Cascade XML file

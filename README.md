🐔 Poultry Disease Detection using Deep Learning
This project detects common poultry diseases using a deep learning model based on transfer learning (VGG16), and provides a user-friendly Flask web application for uploading poultry images and receiving predictions.

📌 Objective
To classify poultry images into one of four categories—three common diseases and healthy—using a trained neural network, and make the system accessible through a simple web interface.

🧠 Model Details
Base Model: VGG16 (pretrained on ImageNet)

Custom Layers:

Flatten

Dense (128 units, ReLU)

Dense (4 units, Softmax)

Classes:

salmo – Salmonella

ncd – Newcastle Disease

cocci – Coccidiosis

healthy – No visible disease

📊 Dataset
~1,876 poultry images

Training Set: ~1500 images

Validation Set: ~376 images

Preprocessing: Resized to 224x224, normalized, augmented

🛠 Technologies Used
Python

TensorFlow / Keras

Flask

HTML, CSS (basic styling)

NumPy

🌐 Web Application Overview
A user uploads an image via the web interface.

The image is resized and passed to the trained model.


Author: Kavuri Divya Sumanvitha

The predicted disease (or healthy status) is returned and displayed with the uploaded image.

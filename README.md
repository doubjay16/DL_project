# DL_project

#name : Junjae Lee

#Title: Anomaly Detection for Pneumonia X-ray Images using Autoencoders


#Summary:
In this project, I will study anomaly detection in medical images using the PneumoniaMNIST dataset. PneumoniaMNIST contains chest X-ray images labeled as normal or pneumonia. Although this dataset is usually used for binary classification, I will reformulate it as an anomaly detection problem.

The main idea is to train an autoencoder using only normal X-ray images. Since the model only learns normal image patterns, it should reconstruct normal images better than pneumonia images. During testing, both normal and pneumonia images will be given to the model. If an image has a high reconstruction error, it will be considered an anomaly.

In my report, I will first explain the difference between binary classification and anomaly detection. Then, I will introduce the PneumoniaMNIST dataset and describe how the autoencoder works. I will implement a simple convolutional autoencoder and compare its results with a denoising autoencoder. I may also use a binary CNN classifier as a supervised benchmark. The performance will be evaluated using reconstruction error, AUROC, accuracy, and confusion matrix. I will also show original and reconstructed X-ray images to explain the results more clearly.

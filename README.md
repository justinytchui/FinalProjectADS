# Marine Animal Image Classification Project

## Project Overview

This project aims to develop advanced machine learning models to classify images of marine animals. Utilizing Convolutional Neural Networks (CNNs) and transfer learning from segmented image datasets, the goal is to enhance classification accuracy and model generalization across various marine species.

## Key Objectives

- **Develop a Baseline CNN Model**: Implement a basic CNN to establish a performance benchmark for marine animal classification.
- **Explore Advanced CNN Architectures**: Test and evaluate advanced architectures such as InceptionV3, MobileNetV2, and Xception to improve upon the baseline model.
- **Implement Transfer Learning**: Use a segmented dataset to leverage transfer learning techniques, enhancing the model's ability to generalize from the segmented to the original images.
- **Performance Evaluation**: Analyze and compare the models based on accuracy, loss, and other relevant metrics.

## Data Description

The project uses two types of datasets:
- **Original Dataset**: Our train set has 621 images (494 for the train set and 127 for the validation set) and our test set has 185 images.
- **Segmented Dataset**: Utilized for transfer learning, includes segmented images which provide detailed features of marine animals, aiding in better feature extraction by the models. This dataset contains over 1500 images with pixel annotations for eight object categories: fish (vertebrates), reefs
(invertebrates), aquatic plants, wrecks/ruins, human divers, robots, and sea-floor. It also includes a test set of 110 images. We only used the "Fisha and Vertebrates" category. 

## Methodology

1. **Data Preprocessing**: Images are resized and augmented to prepare for efficient model training.
2. **Model Training**: Baseline and advanced CNN models are trained using the original dataset. For all models, we did an 80/20 train-test split
3. **Transfer Learning Application**: The transfer learning model is trained on the segmented dataset and then adapted to predict on the original dataset.
4. **Evaluation**: Models are evaluated based on their performance metrics, and detailed analysis is provided through accuracy, loss plots, and confusion matrices.

## Results

- The advanced CNN models generally outperformed the baseline model, demonstrating the effectiveness of architectures like MobileNetV2 and InceptionV3.
- Transfer learning showed mixed results, indicating the need for further tuning and possibly more representative data.

## Future Work

- **Expanding the Dataset**: Acquiring more images to enhance the dataset's diversity and volume.
- **Model Refinement**: Continuously tuning and testing the models to optimize accuracy and reduce overfitting.
- **Exploration of Other Models**: Investigating other model architectures and machine learning techniques that could offer improvements.

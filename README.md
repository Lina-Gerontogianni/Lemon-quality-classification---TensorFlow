# Lemon-quality---TensorFlow
Binary image classification with Tensorflow '2.9.1'

Lemon images of good and bad quality - taken from a public Kaggle [dataset](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset) - are used for fruit quality control. A basic yet complete pipeline for the classification analysis is formed (validation accuracy: 97.11%), including data pre-processing, data configuration, creation of the Convolution Neural Network etc.

In this repo someone can access the Python script -named as lemon_qlt_analysis- along with two images: a) 16 lemon samples from the training set and b) the training/testing performance of the CNN.

The dataset is comprised of 2076 images in total (1125 of good quality, 951 of bad). For the constructed pipeline, the dataset folder should come in a format where two sub-directories are contained as follows:

    dataset/

      good_quality/
  
      bad_quality/

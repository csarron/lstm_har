
# LSTM for Human Activity Recognition

LSTM based human activity recognition using smart phone sensor dataset(a cellphone attached on the waist).
Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.


## Dataset

Dataset can be downloaded at https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

Follow this [link](https://youtu.be/XOEN9W05_4A) to see a video of the how data is collected

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then
sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window).
The sensor acceleration signal, which has gravitational and body motion components, was separated using a
Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed
to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used.

## Model
In this repo, we adopt a 2 layer stacked basic LSTM and use almost the raw data: only the gravity effect has been
filtered out of the accelerometer as a preprocessing step for another 3D feature as an input to help learning.

## Usage
1. Install TensorFlow r1.1
2. Clone this repo by `git clone https://github.com/csarron/lstm_har.git`
3. For training the model, use `python train_lstm.py` (optional step, since model data already provided)
4. For prediction, use `python predict_lstm.py`

## Results 

We got 93.48% test accuracy, it took 9186.56s for training on CPU
(MacBookPro12,1; Intel Core i7, 3.1GHz; Mem 16GB DDR3 1867 MHz)


## Acknowledgement
The original source code is modified from
[guillaume-chevalier](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)

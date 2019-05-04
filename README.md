# DFT Speaker Identification
The Discrete Fourier Transform (DFT) is a transformation which takes a discrete signal and finds it's constituent frequencies.

Since each person's voice is comprised of a unique set of frequencies, it is possible to use the DFT to take an audio sample and determine which individual out of a group is speaking in the audio clip.

# Methodology
## Data Processing
1. Take the audio sample and chunk it into segments equal to the length of the samples which were trained on.
2. Perform the DFT on both audio channels for each segment
3. Compute the magnitude of each entry of the frequency domain vector to get the relative strength of each frequency
4. Derive a feature vector out of the frequency domain vector by "bucketing" frequencies together and taking an aggregate value over each bucket (sum, average, max, etc)
5. Apply min-max scaling to the feature vector to normalize for different volume between clips.

## Training
1. Process the data
2. Train a K-Nearest Neighbors (KNN) classifier on the feature vectors

## Classification
1. Process the data
2. Use the KNN classifier to determine how likely it is that the sample belongs to each speaker. If the model's confidence is less than a particular threshold, return unidentified
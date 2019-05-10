# DFT Speaker Identification
The Discrete Fourier Transform (DFT) is a transformation which takes a discrete signal and finds it's constituent frequencies.

Since each person's voice is comprised of a unique set of frequencies, it is possible to use the DFT to take an audio sample and determine which individual out of a group is speaking in the audio clip.

# Methodology
## Data Processing
1. Take the audio sample and chunk it into segments equal to the length of the samples which were trained on.
2. Perform the DFT on both audio channels for each segment
3. Compute the magnitude of each entry of the frequency domain vector to get the relative strength of each frequency
4. Derive a feature vector out of the frequency domain vector by "bucketing" frequencies together and taking an aggregate value over each bucket (sum, average, max, etc)

## Training
1. Process the data
2. Train a K-Nearest Neighbors (KNN) classifier on the feature vectors which uses Cosine Similarity as its distance metric.

## Classification
1. Process the data
2. Use the KNN classifier to determine how likely it is that the sample belongs to each speaker. 
3. If the model's confidence is less than a particular threshold, return unidentified

# Results
The classifier does very well at identifying which audio clip a small 1-second clip comes from. However, different recordings of the same individual appear to have very different frequency components. This makes it difficult for the system to identify who is speaking when given a recording that it was not trained on. In other words, without more preprocessing of the signal, this basic model grossly overfits.

To fix the problem of overfitting, it is likely that the audio recordings need to be preprocessed more to properly extract the true voiceprint of the speaker.


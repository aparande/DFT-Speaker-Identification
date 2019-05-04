import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

class SpeakerIdentifier:
    def __init__(self, sample_rate=16000, chunk_length=1, both_channels=False, feature_num=200, aggregator=sum, signal_threshold=100, n_neighbors=3, certainty=0.5, data_dir="audio_samples/saved"):
        """
        sample_rate: How many samples/second do the audio clips use
        chunk_length: How many seconds of audio will you use to classify
        both_channels: Uses only the left audio channel if false
        feature_num: The number of features to use
        aggregator: How will the features be computed (sum, average, max, etc)
        signal_threshold: How loud must a signal be to not be considered silence
        n_neighbors: How many neighbors will be used to classify
        certainty: How certain must the classifier be to count its prediction
        data_dir: Where will the training data be saved?
        """
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.N = int(sample_rate * chunk_length)
        self.f0 = 1 / self.N
    
        assert self.N % feature_num == 0, "Feature Number must divide vector length evenly"

        self.speakers = []
        self.both_channels = both_channels
        self.feature_num = feature_num
        self.aggregator = aggregator
        self.signal_threshold = signal_threshold
        self.data_dir = data_dir
        self.left_model = KNeighborsClassifier(n_neighbors = n_neighbors)
        self.certainty = certainty
        if self.both_channels:
            self.right_model = KNeighborsClassifier(n_neighbors = n_neighbors)

    def extract_features(self, audio_sample):
        """
        Extracts the features out of an 2-channel audio sample
        """
        features_left = np.zeros((0, self.feature_num))
        features_right = np.zeros((0, self.feature_num))

        bucket_size = int(self.N / self.feature_num)

        for i in range(0, len(audio_sample), self.N):
            sample = audio_sample[i: i + self.N]

            #Make sure the length of the sample is consistent
            if len(sample) != self.N:
                continue

            left = sample[:, 0]
            #Don't include the section of the audio if it is too quiet
            if np.mean(np.abs(left)) < self.signal_threshold: 
                continue
            
            #Perform the DFT
            dft_left = np.abs(np.fft.fft(left) / self.N)
            #Extract the feature vector using the aggregator function
            left_feature_vec = np.array([self.aggregator(dft_left[x:x+bucket_size]) for x in range(0, len(dft_left), bucket_size)])
            #Apply Min-Max Scaling to account for different volumes of clips
            left_feature_vec = min_max_scaling(left_feature_vec)

            features_left = np.vstack((features_left, left_feature_vec))

            if self.both_channels:
                right = sample[:, 1]
                dft_right = np.abs(np.fft.fft(right) / self.N)
                right_feature_vec = np.array([self.aggregator(dft_right[x:x+bucket_size]) for x in range(0, len(dft_right), bucket_size)])
                right_feature_vec = min_max_scaling(left_feature_vec)
                features_right = np.vstack((features_right, right_feature_vec))

        return (features_left, features_right)

    def add_speaker(self, name, audio_sample):
        """
        Add a new speaker to be recognized by the classifier
        """
        if name in self.speakers:
            print("Error: %s is already a speaker" % name)
            return
        
        self.speakers.append(name)

        features_left, features_right = self.extract_features(audio_sample)

        if self.both_channels:
            self.save_data(name, (features_left, features_right))
        else:
            self.save_data(name, (features_left))

        self.train_model()

    def train_model(self):
        """
        Trains the KNN model on the given data
        """
        labels = np.zeros((0, 1))
        left_data = np.zeros((0, self.feature_num))
        right_data = np.zeros((0, self.feature_num))
        for i, speaker in enumerate(self.speakers):
            speaker_data = self.load_data(speaker)

            if len(speaker_data.shape) == 3:
                left_channel = speaker_data[0]
            else:
                left_channel = speaker_data

            speaker_labels = np.reshape(np.array([i for x in range(len(left_channel))]), (-1, 1))

            labels = np.vstack((labels, speaker_labels))
            left_data = np.vstack((left_data, left_channel))

            if self.both_channels:
                right_channel = speaker_data[1]
                right_data = np.vstack((right_data, right_channel))

        labels = np.reshape(labels, (labels.shape[0],))

        self.left_model.fit(left_data, labels)
        if self.both_channels:
            self.right_model.fit(right_data, labels)

    def classify(self, audio_sample, should_print=True):
        """
        Classifies a two-channel audio sample using the model
        """
        features_left, features_right = self.extract_features(audio_sample)
        classification_counts = [0 for x in range(len(self.speakers) + 1)]

        for i in range(len(features_left)):
            feature = np.reshape(features_left[i, :], (1, -1))

            left_proba = self.left_model.predict_proba(feature)[0]
            left_pred = np.argmax(left_proba)

            if left_proba[left_pred] >= self.certainty:
                classification_counts[left_pred] += 1
            else:
                classification_counts[-1] += 1

            if self.both_channels:
                right_proba = self.right_model.predict_proba(feature)[0]
                right_pred = np.argmax(right_proba)

                if right_proba[right_pred] >= self.certainty:
                    classification_counts[right_pred] += 1
                else:
                    classification_counts[-1] += 1

        classification = np.argmax(classification_counts)
        if should_print:
            sample_num = sum(classification_counts)
            print("Probabilities -- ", sep="")
            for i in range(len(classification_counts) - 1):
                print("%s: %f" % (self.speakers[i], classification_counts[i] / sample_num), sep=", ")
            print("Unknown: %f" % (classification_counts[-1] / sample_num))

        if classification == len(self.speakers):
            print("Unidentified Speaker")
            return -1
        else:
            print("Identified %s" % self.speakers[classification])
            return classification
            
    def save_data(self, speaker_name, data):
        """
        Saves the training sample files
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        path = os.path.join(self.data_dir, speaker_name + ".pkl")
        np.array(data).dump(path)

    def load_data(self, speaker_name):
        """
        Loads the training sample file for a speaker
        """
        path = os.path.join(self.data_dir, speaker_name + ".pkl")
        return np.load(path)

def min_max_scaling(data_vec):
    """
    Perform's min-max scaling on the data
    """
    return (data_vec - np.min(data_vec)) / (np.max(data_vec - np.min(data_vec)))
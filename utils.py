from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from preprocess import SAMPLE_RATE, CHUNK_LENGTH, TRAINING_DIR, N, f0

def plot_audio(name, sample):
    _, audio = wavfile.read(TRAINING_DIR+name+"/"+str(sample)+".wav")
    freq_data = np.fft.fft(audio, axis=0) / N

    plt.plot(np.arange(0, len(audio)) / N, audio)
    plt.title("Time Domain")
    plt.show()
    plt.title("Frequency Domain")
    plt.plot(np.arange(0, len(freq_data)) / f0, freq_data)
    plt.show()

def plot_signal(signal, title, x_scale):
    plt.plot(np.arange(0, len(signal)) * x_scale, signal)
    plt.title(title)
    plt.show()

def remove_outliers(data, threshold = 100):
    cleaned = np.zeros((0, data.shape[1]))
    for i in range(len(data)):
        row = np.reshape(data[i, :], (1, -1))
        if np.mean(np.abs(row)) > threshold:
            cleaned = np.vstack((cleaned, row))

    return cleaned

def find_centroids(data_vecs, labels):
    num_centroids = len(np.unique(labels))
    centroids = {}
    counts = list(range(num_centroids))
    for i in range(len(labels)):
        vec = data_vecs[i]
        label = labels[i, 0]
        if label in centroids:
            centroids[label] = centroids[label] + vec
        else:
            centroids[label] = vec
        
        counts[label] += 1

    return [centroids[i] / counts[i] for i in range(num_centroids)]
        
    
from scipy.io import wavfile
import numpy as np

from os import listdir
from os.path import isfile, join

SAMPLE_RATE = 16000 #Number of samples per second
CHUNK_LENGTH = 1 #Number of seconds in each chunk of audio
SAMPLE_DIR = "audio_samples/Original/"
TRAINING_DIR = "audio_samples/Training/"
SAMPLES_NAMES = ["anmol", "daniel", "emily", "kishan"]
N = CHUNK_LENGTH * SAMPLE_RATE
f0 = 1 / N

def break_data_in_file(filename):
    rate, audio_data = wavfile.read(SAMPLE_DIR + filename+".wav")
    for i in range(0, len(audio_data), N):
        sample_data = audio_data[i: i+N]
        wavfile.write(TRAINING_DIR+filename+"/"+str(i // N)+".wav", rate, sample_data)

def load_samples_for_name(name):
	DIR = TRAINING_DIR + name
	files = [f for f in listdir(DIR) if isfile(join(DIR, f))]

	files = sorted(files, key=lambda x: int(x[:-4]))

	left_channel = np.zeros((0, N))
	right_channel = np.zeros((0, N))

	for filename in files:
		rate, audio = wavfile.read(join(DIR, filename))

		if len(audio) != N:
			continue

		left = np.reshape(audio[:, 0], (1, -1))
		right = np.reshape(audio[:, 1], (1, -1))

		left_channel = np.vstack((left_channel, left))
		right_channel = np.vstack((right_channel, right))

	return left_channel, right_channel

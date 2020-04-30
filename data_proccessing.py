import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import librosa
import librosa.display

# Reading in all the audio files.
data_listdir = os.listdir('./fma-small')
data_listdir.sort()

data = []

for i in data_listdir:

    actor_listdir = os.listdir('./data/' + i)

    actor_listdir.sort()

    for wav_file in actor_listdir:

        identifiers = wav_file.split('.')[0].split('-')

        path = './data/' + i + '/' + wav_file
        modality = int(identifiers[0])
        vocal_channel = int(identifiers[1])
        emotion = int(identifiers[2])
        emotional_intensity = 0 if identifiers[3] == '01' else 1
        statement = 0 if identifiers[4] == '01' else 1
        repetition = 0 if identifiers[5] == '01' else 1
        actor = int(identifiers[6])
        gender = 'male' if actor % 2 else 'female'

        data.append([path, modality, vocal_channel, emotion,
            emotional_intensity, statement, repetition, actor, gender])

# Store all the data for each file in a pandas DataFrame object.
df = pd.DataFrame(data, columns=['path', 'modality', 'vocal channel', 'emotion',
    'emotional intensity', 'statement', 'repetition', 'actor', 'gender'])

filename = df.path[245]
# loads and decodes the audio as a time series y, represented as a
# one-dimensional NumPy floating point array. The variable sr contains
# the sampling rate of y, that is, the number of samples per second of audio.
y, sr = librosa.load(filename)

# Trim the parts of the audio where it is silent.
yt, index = librosa.effects.trim(y, top_db=30)

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
librosa.display.waveplot(yt, sr=sr)
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Time [sec.]')


# Spectrogram representation of the Power Spectral Density
D = np.abs(librosa.stft(yt, n_fft=512))**2
S = librosa.feature.melspectrogram(S=D, sr=sr)

# Convert to dB
S_dB = librosa.power_to_db(S, ref=np.max)

ax2 = fig.add_subplot(212)
ax2.set_title('Mel-frequency spectrogram of ' + filename)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
#librosa.display.specshow(DdB, sr=sr, x_axis='time', y_axis='hz')
#librosa.display.specshow(DdB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [sec.]')

plt.show()

# Feature Extraction:

# Modulation spectral features
# D, phase = librosa.magphase(librosa.stft(yt))
# rms = librosa.feature.rms(S=S)
#
# print(rms.shape)

S, phase = librosa.magphase(librosa.stft(yt))
rms = librosa.feature.rms(S=S)
flatness = librosa.feature.spectral_flatness(S=S)
cent = librosa.feature.spectral_centroid(S=S)
mfccs = librosa.feature.mfcc(yt, sr=sr)

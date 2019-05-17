import pickle
from preprocess import *
import time
import wave
import numpy as np
import pandas as pd
import pyaudio
from flask import Flask, render_template, redirect
from numpy.fft import fft
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import keras.backend as k
app = Flask(__name__)


feature_dim_2 = 11
feature_dim_1 = 20
channel = 1


@app.route('/')
def index():
    return render_template('index.html')

WAVE_OUTPUT_FILENAME = "recorded.wav"

@app.route('/listen')
def listen():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 2048
    RECORD_SECONDS = 1

    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    # redirect('/predict', 200, 'Voice Captured')
    return render_template('listen.html', response='Voice captured successfully')


#This function takes the dataset as numpy array, applies the StandardScalar and returns the transformed dataset numpy Array.
def extractXTestFromDS(ds):
    X_test = MinMaxScaler().fit_transform(ds)
    return X_test

def predictDataSet(ds):
    # filename = 'mlp_1500Files_4000Features.sav'
    filename = 'mlp_6_words.sav'
    mlp_loaded = pickle.load(open(filename, 'rb'))
   # X_test = extractXTestFromDS(ds)
    predictions = mlp_loaded.predict(ds)
    return predictions

#This function takes the input from the user and converts into fourier transformed sample and returns in the form of DataFrame
def prepareDsFromFile(inputFile,FeatureSize):
    sampleRate, sound = wavfile.read(inputFile)
    sound = np.array(sound)
    sound = sound * (20000 / max(sound))
    fftSample = fft(sound)
    fftSample = np.abs(fftSample)
    df = pd.DataFrame(fftSample[:FeatureSize])
    ds = np.array(df.T)
    return ds

@app.route('/predict')
def getPrediction():
    startTime = time.time()
    ds = prepareDsFromFile(WAVE_OUTPUT_FILENAME,4000)
    prediction = predictDataSet(ds)
    return render_template('prediction.html',prediction = prediction,timeDuration = (time.time()-startTime))

def loadModel(fileName):

    loadedModel = pickle.load(open(fileName, 'rb'))
    # X_test = extractXTestFromDS(ds)
    # print("aa",X_test)
    return loadedModel

@app.route('/predictByKeras')
def predictionByKeras():
    modelPath = "KerasModel.sav"
    fileName = "recorded.wav"
    model = pickle.load(open(modelPath, 'rb'))
    startTime = time.time()
    sample = wav2mfcc(fileName)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    prediction = get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]
    k.clear_session()
    return render_template('prediction.html', prediction=prediction, timeDuration=(time.time()-startTime))


@app.route('/predict/<fileName>')
def getPredictions(fileName):
    startTime = time.time()
    ds = prepareDsFromFile(fileName+'.wav', 4000)
    prediction = predictDataSet(ds)
    #return 'Predicted Value:'+  np.array2string(prediction)+'\t'+'Total time: %f sec'%(time.time()-startTime)
    return render_template('prediction.html', prediction=prediction, timeDuration=(time.time()-startTime))

if __name__ == '__main__':
    modelPath = "KerasModel.sav"
    app.run(host='0.0.0.0',port=8090)


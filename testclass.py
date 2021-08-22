import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow import keras
import librosa
import os

def getDataFromFolder(folder):
    outX = []
    files = os.listdir(folder)
    print("files : ", files)
    for file in files:
        if os.path.isfile(folder + "/" +file):
            data = getAudioData(folder + "/" + file)
            # print("data ", data)
            # print("classid ", classid)
            #outX.append(np.array(data))
            # pos = randint(0, 2)
            print(file)
            outX.append(data)
    #outX = np.array(outX)
    return outX

def getAudioData(filePath):
    audioData, _ = librosa.load(filePath, sr=16000)
    return audioData

def main():
    try:
        # X = getDataFromFolder("./testdata")
        X = getDataFromFolder("./testdata/300ms_0.6/")
        model = keras.models.load_model('./qrnn.h5')
        if model == None:
            print("Failed...")

        #model.summary()
        #print(X)

        for x in X:
            #print(x)
            output = model.predict(np.array([x]))
            print(output)

    except Exception as ex:
        print("error", ex)

if __name__ == "__main__":
    main()



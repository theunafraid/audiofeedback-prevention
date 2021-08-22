import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch
from model import AudioClass
from qrnn import QRNN
from numpy.random import seed
from numpy.random import randn
from random import randint
from lstmfcn import LSTM_FCN
import librosa
import os

def getData():
    outX = []
    outY = []
    for i in range(10):
        values = randn(16000)
        outX.append(np.array(values))
        pos = randint(0, 2)
        outY1=np.zeros(3)
        outY1[pos] = 1.0
        outY.append(outY1)

    outX = np.array(outX)
    return outX, np.array(outY)

def readFileData(dir, filename):
    class_id = (filename.split('-')[1]).split('.')[0]
    # print("found class : ", class_id, flush=True)
    filepath = dir + '/'+filename
    data, sample_rate = librosa.load(filepath,sr=16000)
    # a = np.vstack(data)
    # print(a.shape)
    return np.vstack(data), int(class_id)

def getDataFromFolder(folder):
    outX = []
    outY = []
    files = os.listdir(folder)
    print("files : ", files)
    for file in files:
        if os.path.isfile(folder + "/" +file):
            data, classid = readFileData(folder, file)
            # print("data ", data)
            # print("classid ", classid)
            outX.append(np.asarray(data).astype(np.float32))#np.array(data))
            # pos = randint(0, 2)
            outY1=np.zeros(3)
            outY1[classid] = 1.0
            outY.append(outY1)

    #print(outX, flush=True)

    outX = np.asarray(outX).astype(np.float32) #np.array(outX, dtype="object")
    return outX, np.array(outY)

def main():
    try:
        model = QRNN(16000, 5120) #16000)#AudioClass(3)
        model.printmodel()
        # return
        X, Y = getDataFromFolder("./audio/ds_0.3s/300ms_additional/")
        #print(Y.shape)
        #print(X.shape)
        #print(Y)
        #print(X)
        # return
        epochs = 350
        batch = 8
        model.train(X, Y, epochs, batch)
        print("save model...", flush=True)
        model.save("./qrnn.h5")
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()

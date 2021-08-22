import tensorflow as tf
from tensorflow.keras import backend
from keras.models import Model,Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten,BatchNormalization,Dropout, Activation, Input
from keras import regularizers

CLASSES=int(3)

class QRNN:
    def __init__(self, sample_rate, samples) -> None:
        # filter_size = int(sample_rate / 2.0)#/ 8.0) # was 8
        # filter_stride = int(sample_rate / 16.0)#/ 16.0) # was 16
        filter_size = 64#int(sample_rate / 2.0)#/ 8.0) # was 8
        filter_stride = 8#int(sample_rate / 16.0)#/ 16.0) # was 16
        print("filter_size=", filter_size)
        print("filter_stride=",filter_stride)
        kr_l2 = 0.0001
        # filter_units = 128
        filter_units = 64
        l0 = tf.keras.layers.Input(shape=(samples, 1))
        l1_ki = tf.keras.initializers.variance_scaling(distribution="normal")
#        l1 = tf.keras.layers.Conv1D(filters=128, kernel_size=filter_size, strides=filter_stride, kernel_initializer=l1_ki, padding="same")(l0)
        l1 = tf.keras.layers.Conv1D(filters=filter_units, kernel_size=filter_size, strides=filter_stride, kernel_initializer=l1_ki, padding="same", kernel_regularizer=tf.keras.regularizers.l2(kr_l2))(l0)
        l2 = tf.keras.layers.BatchNormalization()(l1)
        l3 = tf.keras.layers.ReLU()(l2)
        l4 = tf.keras.layers.MaxPool1D(8, 8, padding="same")(l3)
        #d1 = tf.keras.layers.Dropout(0.5)(l4)
        # gavg1 = tf.keras.layers.GlobalAveragePooling1D()(l1)

#        l5 = tf.keras.layers.Conv1D(128, 8, 1, kernel_initializer=l1_ki, padding="same")(l4)
        l5 = tf.keras.layers.Conv1D(filter_units, 8, 1, kernel_initializer=l1_ki, padding="same", kernel_regularizer=tf.keras.regularizers.l2(kr_l2))(l4)
        l6 = tf.keras.layers.BatchNormalization()(l5)
        l7 = tf.keras.layers.ReLU()(l6)
        # was bellow may
        # l8 = tf.keras.layers.Conv1D(128, 8, 1, kernel_initializer=l1_ki, padding="same")(l7)
        # l9 = tf.keras.layers.BatchNormalization()(l8)
        # l10 = tf.keras.layers.ReLU()(l9)

#        l11 = tf.keras.layers.Conv1D(128, 8, 1, kernel_initializer=l1_ki, padding="same")(l7)
        l11 = tf.keras.layers.Conv1D(filter_units, 8, 1, kernel_initializer=l1_ki, padding="same", kernel_regularizer=tf.keras.regularizers.l2(kr_l2))(l7)
        l12 = tf.keras.layers.BatchNormalization()(l11)
        l13 = tf.keras.layers.ReLU()(l12)
        #l14 = tf.keras.layers.MaxPool1D(4, 4, padding="same")(l13)
        l14 = tf.keras.layers.GlobalAveragePooling1D()(l13)

        l15 = tf.keras.layers.Flatten()(l14)
        lstm_input = tf.keras.layers.Reshape(target_shape=(filter_units,1))(l15)
        #tf.keras.layers.
#        l16 = tf.keras.layers.SimpleRNN(128)(lstm_input)
        l16 = tf.keras.layers.SimpleRNN(filter_units, kernel_regularizer=tf.keras.regularizers.l2(kr_l2))(lstm_input)
        #l16 = tf.keras.layers.LSTM(128, dropout=0.25)(lstm_input)
        flatten = tf.keras.layers.Flatten()(l16)
        FC_ki = tf.keras.initializers.variance_scaling(distribution="normal")
        FC = tf.keras.layers.Dense(CLASSES, kernel_initializer=FC_ki)(flatten)
        output = tf.keras.layers.Activation(activation="softmax")(FC)
        self.model = tf.keras.Model(inputs=l0, outputs=output)
        self.model.compile(loss=tf.keras.losses.mean_squared_logarithmic_error, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])

    def printmodel(self):
        self.model.summary()

    def train(self, X, Y, epochsValue, batchValue):
        self.model.fit(X, Y, epochs=epochsValue, batch_size=batchValue)

    def save(self, filePath):
        tf.keras.models.save_model(model=self.model, filepath=filePath, save_format='h5')

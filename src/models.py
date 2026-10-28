"""
A collection of models.
"""
import sys
from collections import deque

from keras.layers import (
    LSTM,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    MaxPooling3D,
    TimeDistributed,
    ZeroPadding3D,
)
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop


class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        Initialize the LSTM model.

        Parameters:
        model: The LSTM model to use            
        nb_classes: The number of classes to predict
        seq_length: The length of our video sequences
        saved_model: The path to a saved Keras model to load

        Returns:
        An instance of the ResearchModels class
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        print("Loading LSTM model.")
        self.input_shape = (seq_length, features_length)
        self.model = self.lstm()

        # Now compile the network.
        optimizer = Adam(learning_rate=1e-5, weight_decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network."""
        #Define an empty Sequential Model.
        model = Sequential()
        #Add a simple LSTM layer with 2048 units, do note return the last output, 
        #with self.input_shape input_shape and 0.5 dropout 
        model.add(LSTM(2048, return_sequences=False, input_shape=self.input_shape, dropout=0.5))
        #Add a fully-connected layer (densely-connected NN layer, "Dense layer") with 512 units and relu activation
        model.add(Dense(512, activation='relu'))
        #Add a Dropout layer with 0.5
        model.add(Dropout(0.5))
        #Add a logistic layer (densely-connected NN layer, "Dense layer") with the number of video classes units (self.nb_classes) and softmax activation
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    

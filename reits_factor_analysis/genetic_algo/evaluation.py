"""Import Modules"""
# Numerical Computation
import numpy as np
# Plot
# import matplotlib.pyplot as plt
# Deep Learning
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
# Random Forest
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.utils import to_categorical


class EvaluationClass:
    def __init__(self, X_train, Y_train, X_test, Y_test, model_type, save_computaion=False):

        if model_type == 1:  # Deep Learning Model
            input_shape = np.shape(X_train)[1]
            model = Sequential()
            model.add(Dense(input_shape // 2, activation='relu', input_dim=input_shape))
            model.add(Dropout(0.5))
            model.add(Dense(input_shape // 4, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(5, activation='softmax'))
            sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model = model
            self.Y_train = to_categorical(Y_train)
            self.Y_test = to_categorical(Y_test)
            self.X_train = X_train
            self.X_test = X_test

        elif model_type == 2:  # Random Forest
            model = RandomForestRegressor(n_estimators=100)
            self.model = model
            self.Y_train = Y_train
            self.Y_test = Y_test
            self.X_train = X_train
            self.X_test = X_test

        self.model_type = model_type
        self.save_computaion = save_computaion

    """Definition of the Back Test System"""

    def evalu(self):

        """*********************Training*********************"""
        if self.model_type == 1:
            if self.save_computaion:
                self.model.fit(self.X_train, self.Y_train, validation_split=0.33, epochs=5, batch_size=10, verbose=2)
            else:
                self.model.fit(self.X_train, self.Y_train, validation_split=0.33, epochs=20, batch_size=10, verbose=0)
        elif self.model_type == 2:
            self.model.fit(self.X_train, self.Y_train)

        """*********************Prediction*********************"""
        if self.model_type == 1:
            Y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
            Y_test = np.argmax(self.Y_test, axis=1)
        elif self.model_type == 2:
            Y_pred = self.model.predict(self.X_test)
            Y_test = self.Y_test

        """*********************Statistical Evaluating*********************"""
        accuracy = sum(Y_pred == Y_test)/len(Y_test)
        # print("accuracy = {:.4f}".format(accuracy))
        return accuracy

from keras.models import Sequential
from keras.layers import Dense

class PolicyModel():
    def __init__(self, input_size, num_actions, loss) -> None:
        self.input_size = input_size
        self.output_size = num_actions
        self.loss = loss #categorical_crossentropy or mse
    
    def build(self):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=self.input_size))
        model.add(Dense(128, activation="relu"))

        activation = "softmax" if self.loss == "categorical_crossentropy" else "linear"
        model.add(Dense(self.output_size, activation=activation))

        metric = "mse" if self.loss == "mse" else "categorical_accuracy"

        model.compile(optimizer='adam',loss=self.loss, metrics=[metric])
        return model

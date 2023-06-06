from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

class ConceptModel():
    model = None

    def __init__(self, input_size, num_concepts) -> None:
        self.input_size = input_size
        self.output_size = num_concepts
    
    def build(self):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=self.input_size))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.output_size, activation="softmax"))

        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model = model
        return model
    
    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = keras.models.load_model(filename)
        return self.model


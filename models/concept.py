from keras.models import Sequential
from keras.layers import Dense

class ConceptModel():
    def __init__(self, input_size, num_concepts) -> None:
        self.input_size = input_size
        self.output_size = num_concepts
    
    def build(self):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=self.input_size))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.output_size, activation="softmax"))

        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

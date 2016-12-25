from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from random import choice

class ModelBuilder:
    def __init__(self, activations):
        self.activations = activations

    def build(self, num_layers, num_hidden, input_size=2):
        model = Sequential()

        for i in range(num_layers):
            model.add(Dense(input_dim=(input_size if i == 0 else num_hidden), output_dim=num_hidden, init="normal"))
            
            activation = choice(self.activations)
            model.add(Lambda(lambda x: activation(10 * x)))

            model.add(BatchNormalization())
        
        model.add(Dense(output_dim=3))
        activation = choice(self.activations)
        model.add(Lambda(lambda x: activation(10 * x)))

        model.compile(optimizer="sgd", loss="mse")

        def evaluate(coords):
            return model.predict(coords, batch_size=coords.shape[0])

        return evaluate


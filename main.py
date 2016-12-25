from painter import ModelBuilder
from keras import backend as K
from PIL import Image
import numpy as np

print("Imported all")

IMAGE_SIZE = [768, 1024] #HW
NUM_LAYERS = 6
NUM_HIDDEN = 100

activations = [
    K.softplus,
    K.softsign,
    K.tanh,
    K.sigmoid,
    K.hard_sigmoid,
    K.sin,
    K.cos,
    K.abs,
    K.log,
    K.square,
    K.softmax,
    K.sqrt,
]

builder = ModelBuilder(activations)

colors = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

iter = 0
while True:
    model = builder.build(NUM_LAYERS, NUM_HIDDEN)

    print("Built model")

    for i in range(IMAGE_SIZE[0]):
        coords_x = [2 * (i / IMAGE_SIZE[0] - 0.5)] * IMAGE_SIZE[1]
        coords_y = [2 * (j / IMAGE_SIZE[1] - 0.5) for j in range(IMAGE_SIZE[1])]
        coords = np.array([coords_x, coords_y], dtype=np.float32).T

        # Get IMAGE_SIZEx3
        colors[i] = model(coords)

    data = (255 * (colors - np.min(colors)) / (np.max(colors - np.min(colors)))).astype(np.uint8)

    print("Mean:", np.mean(data))
    print("STDDev:", np.std(data))
    print("Min:", np.min(data))
    print("Max:", np.max(data))

    if np.min(data) != np.max(data): 
        img = Image.fromarray(data, "RGB")
        img.save("image_%d.png" % iter)
        img.show()
    else:
        print("Bad image")

    iter += 1

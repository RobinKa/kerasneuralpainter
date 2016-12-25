from painter import ModelBuilder
from keras import backend as K
from PIL import Image
import numpy as np

print("Imported all")

IMAGE_SIZE = [720, 1280] #HW
NUM_LAYERS = 10
NUM_HIDDEN = 100
START_SCALE = -3
SCALE_PER_ITER = 0.01

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
scale = START_SCALE

model = builder.build(NUM_LAYERS, NUM_HIDDEN, 3)
print("Built model")

iter = 0
while True:
    for i in range(IMAGE_SIZE[0]):
        coords_x = [2 * (i / IMAGE_SIZE[0] - 0.5)] * IMAGE_SIZE[1]
        coords_y = [2 * (j / IMAGE_SIZE[1] - 0.5) for j in range(IMAGE_SIZE[1])]
        coords_z = [scale] * IMAGE_SIZE[1]
        coords = np.array([coords_x, coords_y, coords_z], dtype=np.float32).T

        # Get IMAGE_SIZEx3
        colors[i] = model(coords)

    data = (255 * (colors - np.min(colors)) / (np.max(colors) - np.min(colors))).astype(np.uint8)

    if np.min(data) != np.max(data) or iter > 0:
        img = Image.fromarray(data, "RGB")
        img.save("gifimage_%d.png" % iter)

        iter += 1
        scale += SCALE_PER_ITER

        print("Iter", iter, "Scale", scale)
    else:
        print("Bad image")

        if iter == 0:
            model = builder.build(NUM_LAYERS, NUM_HIDDEN, 3)
            print("Recreating model")
        else:
            print("Failed")
            exit()

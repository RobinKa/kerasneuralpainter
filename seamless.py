from painter import ModelBuilder
from keras import backend as K
from PIL import Image
import numpy as np

print("Imported all")

IMAGE_SIZE = [240, 320] #HW
NUM_LAYERS = 10
NUM_HIDDEN = 100
STEP_COUNT = 250
RADIUS = 2

# Activation functions that can be called like f(x)
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

model = None

def make_model():
    global model
    model = builder.build(NUM_LAYERS, NUM_HIDDEN, 4)
    print("Built model")

make_model()

iter = 0
while True:
    for i in range(IMAGE_SIZE[0]):
        # The imagespace coordintes
        coords_x = [2 * (i / IMAGE_SIZE[0] - 0.5)] * IMAGE_SIZE[1]
        coords_y = [2 * (j / IMAGE_SIZE[1] - 0.5) for j in range(IMAGE_SIZE[1])]

        # The "virtual" time coordinates, go around in a circle over time
        angle = 2 * np.pi * iter / STEP_COUNT
        coords_u = [RADIUS * np.cos(angle)] * IMAGE_SIZE[1]
        coords_v = [RADIUS * np.sin(angle)] * IMAGE_SIZE[1]

        coords = np.array([coords_x, coords_y, coords_u, coords_v], dtype=np.float32).T

        # Get IMAGE_SIZEx3
        colors[i] = model(coords)

    # Normalize the output to [0, 255]
    data = (255 * (colors - np.min(colors)) / (np.max(colors) - np.min(colors))).astype(np.uint8)

    # Check that the image is not a constant color
    if np.min(data) != np.max(data) or iter > 0:
        img = Image.fromarray(data, "RGB")
        img.save("seamless_%d.png" % iter)

        iter += 1

        print("Iter", iter, "/", STEP_COUNT)

        if iter >= STEP_COUNT:
            break
    else:
        print("Bad image")

        if iter == 0:
            make_model()
        else:
            print("Failed")
            exit()

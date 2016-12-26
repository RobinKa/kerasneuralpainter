from painter import ModelBuilder
from keras import backend as K
from PIL import Image
import numpy as np

print("Imported all")

NUM_LAYERS = 10
NUM_HIDDEN = 100
INPUT_PATH = "KJU.jpg"

# Load image and normalize between -1 and 1
input_image = np.asarray(Image.open(INPUT_PATH), dtype=np.float32)
input_image[:, :, 0] = 2 * ((input_image[:, :, 0] - np.min(input_image[:, :, 0])) / (np.max(input_image[:, :, 0]) - np.min(input_image[:, :, 0])) - 0.5)
input_image[:, :, 1] = 2 * ((input_image[:, :, 1] - np.min(input_image[:, :, 1])) / (np.max(input_image[:, :, 1]) - np.min(input_image[:, :, 1])) - 0.5)
input_image[:, :, 2] = 2 * ((input_image[:, :, 2] - np.min(input_image[:, :, 2])) / (np.max(input_image[:, :, 2]) - np.min(input_image[:, :, 2])) - 0.5)

IMAGE_SIZE = input_image.shape
print("Loaded Image, Size:", IMAGE_SIZE)

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

coords_y = [2 * (j / IMAGE_SIZE[1] - 0.5) for j in range(IMAGE_SIZE[1])]

iter = 0
while True:
    model = builder.build(NUM_LAYERS, NUM_HIDDEN, 5)

    print("Built model")

    for i in range(IMAGE_SIZE[0]):
        coords_x = [2 * (i / IMAGE_SIZE[0] - 0.5)] * IMAGE_SIZE[1]

        image_r = input_image[i, :, 0]
        image_g = input_image[i, :, 1]
        image_b = input_image[i, :, 2]

        coords = np.array([coords_x, coords_y, image_r, image_g, image_b], dtype=np.float32).T

        # Get IMAGE_SIZEx3
        colors[i] = model(coords)

    '''
    data = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    data[:, :, 0] = (255 * (colors[:, :, 0] - np.min(colors[:, :, 0])) / (np.max(colors[:, :, 0]) - np.min(colors[:, :, 0]))).astype(np.uint8)
    data[:, :, 1] = (255 * (colors[:, :, 1] - np.min(colors[:, :, 1])) / (np.max(colors[:, :, 1]) - np.min(colors[:, :, 1]))).astype(np.uint8)
    data[:, :, 2] = (255 * (colors[:, :, 2] - np.min(colors[:, :, 2])) / (np.max(colors[:, :, 2]) - np.min(colors[:, :, 2]))).astype(np.uint8)
    '''

    data = (255 * (colors - np.min(colors)) / (np.max(colors) - np.min(colors))).astype(np.uint8)

    if np.min(data) != np.max(data): 
        img = Image.fromarray(data, "RGB")
        img.save("styled_kju_%d.png" % iter)
        img.show()
    else:
        print("Bad image")

    iter += 1

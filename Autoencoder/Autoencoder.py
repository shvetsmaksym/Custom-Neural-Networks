from PIL import Image
import numpy as np

from DataStructures.NeuralNetwork import NeuralNetwork


def load_image():
    img = Image.open("img1.png")
    img = img.convert('RGB')
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            print(pixels[i, j])

    img.show()


def create_image():
    img = Image.new('RGB', (256, 256))
    pixels = img.load()

    # Modify
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = (i, j, 100)

    img.show()


def process_image(path, block_size=8):
    """Return 5-dimensional array. Divide image into blocks of (block_size * block_size) dimensions.
     First and second dimensions correspond to indexes of blocks.
     Third and forth dimensions correspond to coordinates of blocks.
     Fifth dimension is RGB representation.

     In our example it is (32 x 32 x 8 x 8 x 3).

     Also rescales values from [0, 255] to [0.1, 0.9]."""

    img = Image.open(path)
    img = img.convert('RGB')

    np_image = np.asarray(img)
    np_image = np_image * 0.8 / 255 + 0.1  # Rescaling [0.1, 0.9]

    width, height = np_image.shape[0] // block_size, np_image.shape[1] // block_size
    blocks = np_image.reshape(width, height, block_size, block_size, 3)

    # decoded_blocks = blocks.reshape(256, 256, 3)
    # im = Image.fromarray(decoded_blocks)
    # im.show()

    return blocks


if __name__ == "__main__":
    # Preprocessing
    img1_blocks = process_image("train_data/img1.png")
    red = img1_blocks[:, :, :, :, 0].reshape(32**2, 8**2)

    # Define model
    encoder = NeuralNetwork()
    encoder.add_input_layer(n=32, input_shape=64)
    encoder.add_output_layer(n=64)

    encoder.fit(train_x=red, train_y=red, epochs=5)

    # Testing
    img2_blocks = process_image("train_data/img1.png")
    red2 = img2_blocks[:, :, :, :, 0].reshape(32 ** 2, 8 ** 2)


    encoder.feed_forward(red2)

    print("Done.")

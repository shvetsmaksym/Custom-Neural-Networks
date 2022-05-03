from PIL import Image
import numpy as np
from astropy.nddata import reshape_as_blocks
import matplotlib.pyplot as plt

from DataStructures.NeuralNetworks import NeuralNetwork


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


def image_to_blocks(path, bs=8):
    """Return the array of flatten blocks of bs**2 size.
    For RGB image 256x256 px and 8x8 blocks it's array of shape (3072, 64). (3x32x32, 8x8)

    Also rescale values from [0, 255] to [0.1, 0.9]."""

    img = Image.open(path)
    img = img.convert('RGB')

    np_image = np.asarray(img)
    np_image = np_image * 0.8 / 255 + 0.1  # Rescaling [0.1, 0.9]

    width, height = np_image.shape[0] // bs, np_image.shape[1] // bs

    blocks = reshape_as_blocks(np_image, (bs, bs, 3))
    blocks_ = blocks.swapaxes(2, 5)
    blocks_ = blocks_.reshape(3 * width * height, bs**2)

    return blocks_


def reconstruct_flatten(img, width=None, height=None):
    """width, height - number of blocks widely and highly respectivelly."""
    n_blocks = img.shape[0] // 3
    block_length = int(img.shape[1] ** (1 / 2))
    if width is None and height is None:
        width, height = int(n_blocks ** (1 / 2)), int(n_blocks ** (1 / 2))

    img = img.reshape(n_blocks, 3, img.shape[1])
    img = img.swapaxes(1, 2)
    img = img.reshape(height, width, block_length, block_length, 3)
    img = img.swapaxes(1, 2).reshape(height * block_length, width * block_length, 3)

    return img


# def reconstruct_image(flatten_blocks, img_size=(256, 256, 3)):
#     return flatten_blocks.reshape(img_size)


if __name__ == "__main__":
    # Preprocessing
    img1 = image_to_blocks("train_data/img1.png")
    img2 = image_to_blocks("train_data/img2.png")
    img3 = image_to_blocks("train_data/img3.png")
    img4 = image_to_blocks("train_data/img4.png")
    train_set = np.concatenate((img1, img2, img3, img4))
    print("Conversion from images to blocks completed.")

    # Define model
    encoder = NeuralNetwork(classification_problem=False)
    encoder.add_input_layer(n=32, input_shape=64, discretization=True)
    encoder.add_output_layer(n=64)

    encoder.add_metrics(metrics=['mse'])
    print("Model defining complete.")

    def fit_and_plot(ep=1):
        encoder.fit(train_x=train_set, train_y=train_set, validation_data=(train_set[:256], train_set[:256]), epochs=ep)

        # Testing
        test_images = [img1, img2, img3, img4]
        flatten_blocks = [np.zeros((3*32*32, 8*8)) for _ in test_images]

        for i, img in enumerate(test_images):
            for j, block in enumerate(img):
                encoder.feed_forward(block)
                flatten_blocks[i][j] = np.array([neu.F_net for neu in encoder.output_layer.neurons])

        im1_rec, im2_rec, im3_rec, im4_rec = reconstruct_flatten(flatten_blocks[0]), reconstruct_flatten(flatten_blocks[1]), \
                                             reconstruct_flatten(flatten_blocks[2]), reconstruct_flatten(flatten_blocks[3])
        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(im1_rec)
        ax[0][1].imshow(im2_rec)
        ax[1][0].imshow(im3_rec)
        ax[1][1].imshow(im4_rec)
        plt.show()

    for _ in range(10):
        # Żeby rysowało wyniki wytrenowanej sieci po każdej epoce
        fit_and_plot(ep=1)

    print("Done.")

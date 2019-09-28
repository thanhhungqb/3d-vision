import matplotlib.pyplot as plt


def save_3_images(fname, im1, im2, im3):
    """
    Save 3 images in one rows and make file
    :param fname:
    :param im1:
    :param im2:
    :param im3:
    :return:
    """
    fig = plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 3, 1)
    plt.imshow(im1)
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.imshow(im2)
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.imshow(im3)

    plt.axis('off')
    plt.savefig(fname)

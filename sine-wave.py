import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # read image
    img = np.array(Image.open('lena.bmp'))

    # fourier transform
    f = np.fft.fft2(img)

    # move to center in frequency domian
    f_shift = np.fft.fftshift(f)

    # import sine wave, position (124,128) and (132,128) could be changed if you want
    f_shift2 = f_shift.copy()
    f_shift2[124, 128] = 2000000
    f_shift2[132, 128] = 2000000

    # ifftshift
    f2 = np.fft.ifftshift(f_shift2)

    # ifft
    img2 = np.fft.ifft2(f2)

    # show image
    plt.figure(1), plt.imshow(np.abs(img), 'gray'), plt.title('image')
    plt.figure(2), plt.imshow(np.abs(img2), 'gray'), plt.title('image after importing sin wave')
    plt.figure(3), plt.imshow(np.abs(f_shift), 'gray'), plt.title('frequency spectrum')
    plt.figure(4), plt.imshow(np.abs(f_shift2), 'gray'), plt.title('frequency spectrum after importing sine wave')
    plt.show()


if __name__ == '__main__':
    main()
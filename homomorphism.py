import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def hom_filter_gene(M, N, gama_H=1.5, gama_L=0.1, c=8, D0=10):
    """
    :param M: image height
    :param N: image weight
    :param gama_H: parameter of homomorphism filter
    :param gama_L: parameter of homomorphism filter
    :param c: sharpen parameter
    :param D0: usually the variance(cut-of the frequency of the filter)
    :return: H,the homomorphism filter
    """
    D = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            D[i, j] = np.sqrt((i-M//2)**2 + (j-N//2)**2)
    H = (gama_H-gama_L)*(1-np.exp(-c*(D**2)/(D0**2)))+gama_L
    return H


def main():
    # read image
    img = np.array(Image.open('cave.jpg'))

    # log
    img[img == 0] = 1
    img_log = np.log(img)

    # fft
    f = np.fft.fft2(img_log)
    f_shift = np.fft.fftshift(f)

    # homomorphism filter
    (M, N) = f_shift.shape
    H = hom_filter_gene(M, N, gama_H=0.5, D0=1)
    f_shift_fil = f_shift * H

    # ifft
    f_fil = np.fft.ifftshift(f_shift_fil)
    img_exp = np.fft.ifft2(f_fil)

    # exp
    img_fil = np.exp(img_exp)

    # show image
    plt.figure(1), plt.imshow(np.abs(f_shift), 'gray'), plt.title('frequency spectrum')
    plt.figure(2), plt.imshow(np.abs(f_shift_fil), 'gray'), plt.title('frequency spectrum after filtering')
    plt.figure(3), plt.imshow(np.abs(img), 'gray'), plt.title('image')
    plt.figure(4), plt.imshow(np.abs(img_fil), 'gray'), plt.title('image after filtering')
    plt.show()


if __name__ == '__main__':
    main()
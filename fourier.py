import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # read image
    img_ori = np.array(Image.open('rect.bmp'))
    img_45 = np.array(Image.open('rect-45度.bmp'))
    img_move = np.array(Image.open('rect-move.bmp'))

    # fourier transform
    f_ori = np.fft.fft2(img_ori)
    f_45 = np.fft.fft2(img_45)
    f_move = np.fft.fft2(img_move)

    # move to center in frequency domian
    f_ori_shift = np.fft.fftshift(f_ori)
    f_45_shift = np.fft.fftshift(f_45)
    f_move_shift = np.fft.fftshift(f_move)

    # show image
    plt.figure(1), plt.imshow(np.abs(f_ori_shift), 'gray'), plt.title('rect')
    plt.figure(2), plt.imshow(np.abs(f_45_shift), 'gray'), plt.title('rect-45°')
    plt.figure(3), plt.imshow(np.abs(f_move_shift), 'gray'), plt.title('rect-move')
    plt.show()


if __name__ == '__main__':
    main()
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # read image
    img = np.array(Image.open('grid.bmp'))

    # fourier transform
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)

    # low-pass filter
    filt_low = np.ones(f_shift.shape)
    filt_low[126:130, 118:122] = 0
    filt_low[126:130, 135:139] = 0

    # high-pass filter
    filt_high = np.ones(f_shift.shape)
    filt_high[125:131, 125:131] = 0

    # filtering
    f_shift_low = f_shift * filt_low
    f_shift_high = f_shift * filt_high

    # recovery
    f_low = np.fft.ifftshift(f_shift_low)
    img_low = np.fft.ifft2(f_low)
    f_high = np.fft.ifftshift(f_shift_high)
    img_high = np.fft.ifft2(f_high)

    # show image
    plt.figure(1), plt.imshow(np.abs(f_shift), 'gray'), plt.title('frequency domian')
    plt.figure(2), plt.imshow(np.abs(f_shift_low), 'gray'), plt.title('low-pass in frequency domian')
    plt.figure(3), plt.imshow(np.abs(f_shift_high), 'gray'), plt.title('high-pass in frequency domian')
    plt.figure(4), plt.imshow(np.abs(img_low), 'gray'), plt.title('image after low-pass filtering')
    plt.figure(5), plt.imshow(np.abs(img_high), 'gray'), plt.title('image after high-pass filtering')
    plt.show()


if __name__ == '__main__':
    main()
# ===============================================================================
# Autor: Bogdan T. Nassu, Adriano J. Paulichi, Matheus D. de Freitas
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import numpy as np
import cv2

# ===============================================================================

INPUT_IMAGE = 'GT2.BMP'
MASK_THRESHOLD = 0.5
SRC_WEIGHT = 0.8
MASK_WEIGHT = 1 - SRC_WEIGHT


def main():
    src = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if src is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    print('start')

    src = src.reshape((src.shape[0], src.shape[1], 3))
    src = src.astype(np.float32) / 255

    src_HLS = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)
    src_H, src_L, src_S = cv2.split(src_HLS)
    mask_L = np.where(src_L > MASK_THRESHOLD, src_L, 0.0)
    mask_HLS = cv2.merge([src_H, mask_L, src_S])
    mask = src_HLS = cv2.cvtColor(mask_HLS, cv2.COLOR_HLS2BGR)

    # Gaussian Bloom
    gaussian_mask = np.zeros(mask.shape, np.float32)
    for i in range(3, 7):
        sigma = 2**i
        gaussian_blur = cv2.GaussianBlur(mask, (0, 0), sigma, cv2.BORDER_DEFAULT)
        gaussian_mask += gaussian_blur
    gaussian_bloom = cv2.addWeighted(src, SRC_WEIGHT, gaussian_mask, MASK_WEIGHT, 0)
    
    # Average Bloom
    avg_mask = np.zeros(mask.shape, np.float32)
    for i in range(4):
        ksize = 15 * (i + 1)
        avg_blur = mask.copy()
        for j in range(3):
            avg_blur = cv2.blur(avg_blur, (ksize, ksize), cv2.BORDER_DEFAULT)
        avg_mask += avg_blur
    avg_bloom = cv2.addWeighted(src, SRC_WEIGHT, avg_mask, MASK_WEIGHT, 0)

    cv2.imshow('src', src)
    cv2.imshow('mask', mask)
    cv2.imshow('gaussian_mask', gaussian_mask)
    cv2.imshow('gaussian_bloom', gaussian_bloom)
    cv2.imshow('avg_mask', avg_mask)
    cv2.imshow('avg_bloom', avg_bloom)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# ===============================================================================

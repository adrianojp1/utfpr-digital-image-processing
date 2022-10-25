# ===============================================================================
# Autor: Adriano J. Paulichi, Matheus D. de Freitas
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

from math import sqrt
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


IN_DIR = './img/'
WPP = './wallpapers/Wind Waker GC.bmp'


def get_green_level(bgr):
    b, g, r = bgr
    green_distance = sqrt(r*r + (g - 1)*(g - 1) + b*b)
    capped = min(1, green_distance)
    return capped


def to_green_scale(img):
    green_scale = img.copy()
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            green_scale[y, x] = get_green_level(img[y, x])
            
    normalized = cv2.normalize(green_scale, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized


def merge(img, green_scale, binary_green, wpp):
    merged = img.copy()
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            # Check if it is close to green
            if binary_green[y, x] == 0:
                # Remove green
                merged[y, x][1] = 0
                wpp_factor = (1 - green_scale[y, x]) * wpp[y, x]
                src_factor = green_scale[y, x] * merged[y, x]
                merged[y, x] = wpp_factor + src_factor
    return merged


def binarize(img):
    img = (img * 255).astype(np.uint8)[:, :, 0]

    # histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    # plt.plot(histr)

    (T, thresh_img) = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('Threshhold: ', T)

    # plt.show()
    return thresh_img


def save_img(src_name, img, img_name):
    write = (img * 255).astype(np.uint8)
    cv2.imwrite(f'{src_name}/{img_name}.bmp', write)


def execute(wpp, img_name, save, show):
    img_path = IN_DIR + img_name
    print('Executing for image: ' + img_path)
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if src is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()
    src = src.astype(np.float32) / 255

    wpp = cv2.resize(wpp, (src.shape[1], src.shape[0]), cv2.INTER_LANCZOS4)

    print('Calculating green scale')
    green_scale = to_green_scale(src)

    
    binary_green = binarize(green_scale)

    print('Merging')
    result = merge(src, green_scale, binary_green, wpp)

    if save:
        print('Saving images')
        save_img(img_name, green_scale, 'green_scale')
        
        binary_green = src.astype(np.float32) / 255
        save_img(img_name, binary_green, 'binary_green')
        save_img(img_name, wpp, 'wpp')
        save_img(img_name, result, 'result')

    if show:
        cv2.imshow('src', src)
        cv2.imshow('green_scale', green_scale)
        cv2.imshow('binary_green', binary_green)
        cv2.imshow('wpp', wpp)
        cv2.imshow('result', result)

        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    print('Using wallpaper: ' + WPP)
    wpp = cv2.imread(WPP, cv2.IMREAD_COLOR)
    if wpp is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()
    wpp = wpp.astype(np.float32) / 255

    execute(wpp, '1.bmp', True, False)


if __name__ == '__main__':
    main()

# ===============================================================================
# Autor: Adriano J. Paulichi, Matheus D. de Freitas
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

from math import sqrt
import sys
import numpy as np
import cv2

IN_DIR = './img/'
OUT_DIR = './resultados/'
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
    return green_scale


def merge(img, green_scale, wpp):
    merged = img.copy()
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            merged[y, x][1] = 0
            merged[y, x] = (1 - green_scale[y, x]) * \
                wpp[y, x] + green_scale[y, x] * merged[y, x]
    return merged


def execute(wpp, img_name, show_img):
    img_path = IN_DIR + img_name
    print('Executing for image: ' + img_path)
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if src is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()
    src = src.astype(np.float32) / 255

    wpp = cv2.resize(wpp, (src.shape[1], src.shape[0]), cv2.INTER_LANCZOS4)

    print('start')
    green_scale = to_green_scale(src)
    result = merge(src, green_scale, wpp)

    write = (result * 255).astype(np.uint8)
    cv2.imwrite(OUT_DIR + img_name, write)
    
    if show_img:
        cv2.imshow('src', src)
        cv2.imshow('green_scale', green_scale)
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

    execute(wpp, '0.BMP', True)


if __name__ == '__main__':
    main()

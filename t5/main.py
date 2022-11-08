# ===============================================================================
# Autor: Adriano J. Paulichi, Matheus D. de Freitas
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

from math import sqrt
import os
from pathlib import Path
import sys
import numpy as np
import cv2


IN_DIR = './img/'
OUT_DIR = './resultados/'
WPP = './wallpapers/Wind Waker GC.bmp'
SAVE_IMG = True
SHOW_IMG = False
# IMGS = os.listdir(IN_DIR).sort()
IMGS = ['4.bmp']


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


def remove_green(img, green_scale):
    green_removed = img.copy()
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            green_removed[y, x][1] = img[y, x][1] * green_scale[y, x][0]
    return green_removed


def merge(green_scale, green_removed, wpp):
    merged = green_removed.copy()
    h = green_removed.shape[0]
    w = green_removed.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            wpp_factor = (1 - green_scale[y, x]) * wpp[y, x]
            src_factor = green_scale[y, x] * merged[y, x]
            merged[y, x] = wpp_factor + src_factor
    return merged


def save_img(path, img, img_name):
    write = (img * 255).astype(np.uint8)
    cv2.imwrite(f'{path}/{img_name}.bmp', write)


def execute(wpp, img_name, save, show):
    print('=======================================================')
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

    print('Removing green')
    green_removed = remove_green(src, green_scale)

    print('Merging')
    result = merge(green_scale, green_removed, wpp)

    if save:
        print('Saving images')
        path = OUT_DIR + img_name
        Path(path).mkdir(parents=True, exist_ok=True)

        save_img(path, green_scale, 'green_scale')
        save_img(path, green_removed, 'green_removed')
        save_img(path, wpp, 'wpp')
        save_img(path, result, 'result')

    if show:
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

    print('Images:', IMGS)
    for img_name in IMGS:
        execute(wpp, img_name, SAVE_IMG, SHOW_IMG)


if __name__ == '__main__':
    main()

# ===============================================================================
# Autor: Bogdan T. Nassu, Adriano J. Paulichi, Matheus D. de Freitas
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import numpy as np
import cv2

# ===============================================================================

INPUT_IMAGE = 'a01 - Original.bmp'
WINDOW_SIZE = 3


def naive_algorithm(img, window_size):
    output_img = img.copy()

    h = output_img.shape[0]
    w = output_img.shape[1]

    for y in range(window_size, h - window_size):
        for x in range(window_size, w - window_size):
            for z in range(0, 3):
                sum = 0
                for a in range(-window_size, window_size + 1):
                    for b in range(-window_size, window_size + 1):
                        sum += img[y + b][x + a][z]
                output_img[y][x][z] = sum / ((window_size * 2 + 1) * (window_size * 2 + 1))

    cv2.imshow('naive_algorithm', output_img)
    cv2.imwrite('naive_algorithm.png', output_img * 255)


def separable_filter(img, window_size):
    buffer = img.copy()
    output_img = img.copy()

    h = output_img.shape[0]
    w = output_img.shape[1]

    for y in range(0, h):
        for x in range(window_size, w - window_size):
            for z in range(0, 3):
                sum = 0
                for a in range(-window_size, window_size + 1):
                    sum += img[y][x + a][z]
                buffer[y][x][z] = sum / (window_size * 2 + 1)

    for y in range(window_size, h - window_size):
        for x in range(0, w):
            for z in range(0, 3):
                sum = 0
                for a in range(-window_size, window_size + 1):
                    sum += buffer[y + a][x][z]
                output_img[y][x][z] = sum / (window_size * 2 + 1)

    cv2.imshow('separable_filter', output_img)
    cv2.imwrite('separable_filter.png', output_img * 255)


def integral_images_algorithm(img, window_size):
    integral_matrix = img.copy()
    output_img = img.copy()

    h = integral_matrix.shape[0]
    w = integral_matrix.shape[1]

    for y in range(0, h):
        for x in range(1, w):
            for z in range(0, 3):
                integral_matrix[y][x][z] = img[y][x][z] + integral_matrix[y][x - 1][z]

    for y in range(1, h):
        for x in range(0, w):
            for z in range(0, 3):
                integral_matrix[y][x][z] = integral_matrix[y][x][z] + integral_matrix[y - 1][x][z]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for z in range(0, 3):
                delta = clamp(min(y, x, h - 1 - y, w - 1 - x) - 1, 0, window_size)
                sum = integral_matrix[y + delta][x + delta][z] + \
                      integral_matrix[y - delta - 1][x - delta - 1][z] - \
                      integral_matrix[y + delta][x - delta - 1][z] - \
                      integral_matrix[y - delta - 1][x + delta][z]
                print(delta)
                output_img[y][x][z] = sum / ((delta * 2 + 1) * (delta * 2 + 1))

    cv2.imshow('integral_images_algorithm', output_img)
    cv2.imwrite('integral_images_algorithm.png', output_img * 255)


def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n


def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    print('start')

    img = img.reshape((img.shape[0], img.shape[1], 3))
    img = img.astype(np.float32) / 255

    # algoritmos implementados
    naive_algorithm(img, WINDOW_SIZE)
    separable_filter(img, WINDOW_SIZE)
    integral_images_algorithm(img, WINDOW_SIZE)

    cv2.imshow('original', img)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# ===============================================================================

from statistics import mean, median_low
import cv2
import numpy as np

import sys
sys.setrecursionlimit(3000)


def get_blob_areas(img):
    h = img.shape[0]
    w = img.shape[1]
    label = 1.0

    areas = []
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x] == 1:
                label += 0.1
                comp = {
                    'label': label,
                    'n_pixels': 0,
                    'T': h - 1,
                    'L': w - 1,
                    'B': 0,
                    'R': 0
                }
                flood_fill(img, y, x, comp)
                areas.append(comp['n_pixels'])
    return areas


def flood_fill(img, y, x, comp):
    img[y][x] = comp['label']
    comp['n_pixels'] += 1
    comp['T'] = min(comp['T'], y)
    comp['B'] = max(comp['B'], y)
    comp['L'] = min(comp['L'], x)
    comp['R'] = max(comp['R'], x)

    h = img.shape[0]
    w = img.shape[1]

    # Verifica pixel acima
    if y - 1 > -1 and img[y - 1][x] == 1:
        flood_fill(img, y - 1, x, comp)

    # Verifica pixel abaixo
    if y + 1 < h and img[y + 1][x] == 1:
        flood_fill(img, y + 1, x, comp)

    # Verifica pixel a esquerda
    if x - 1 > -1 and img[y][x - 1] == 1:
        flood_fill(img, y, x - 1, comp)

    # Verifica pixel a direita
    if x + 1 < w and img[y][x + 1] == 1:
        flood_fill(img, y, x + 1, comp)

    # Verifica pixel a direita e abaixo
    if x + 1 < w and y + 1 < h and img[y + 1][x + 1] == 1:
        flood_fill(img, y, x + 1, comp)

    # Verifica pixel a direita e acima
    if x + 1 < w and y - 1 > -1 and img[y - 1][x + 1] == 1:
        flood_fill(img, y, x + 1, comp)

    # Verifica pixel a esquerda e abaixo
    if x - 1 > -1 and y + 1 < h and img[y + 1][x - 1] == 1:
        flood_fill(img, y, x - 1, comp)

    # Verifica pixel a esquerda e acima
    if x - 1 > -1 and y - 1 > -1 and img[y - 1][x - 1] == 1:
        flood_fill(img, y, x - 1, comp)


def fill_borders(img):
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return img | im_floodfill_inv


def estimate_rice_size(areas):
    median = median_low(areas)
    upper_limit = median * 1.2
    lower_limit = median * 0.8
    filtered = [area for area in areas if lower_limit < area < upper_limit]
    avg = mean(filtered)
    return avg


def count_rices(areas, median_area):
    count = 0

    for area in areas:
        if median_area * 0.2 < area < median_area * 1.5:
            count += 1

        elif area >= median_area * 1.5:
            count += round(area / median_area)

    return count


def execute(img_name):
    print('Executing for image: ' + img_name)
    expected_count = int(img_name.split('.')[0])
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    img_in = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_blur = cv2.medianBlur(img, 7, cv2.BORDER_DEFAULT)

    img_border = cv2.Canny(img_blur, 50, 150, L2gradient=True)

    img_filled = fill_borders(img_border)
    img_filled = img_filled.astype(np.float32) / 255

    areas = get_blob_areas(img_filled)
    areas.sort()
    # print(areas)

    rice_size = estimate_rice_size(areas)
    print('Rice size: ', rice_size)

    count = count_rices(areas, rice_size)
    diff = abs(expected_count - count)
    error = diff/expected_count * 100

    print('Expected count: ', expected_count)
    print('Actual count: ', count)
    print('Percentage error: ', error)
    print('=========================================')

    cv2.imshow('original', img_in)
    cv2.imshow('blur', img_blur)
    cv2.imshow('borders', img_border)
    cv2.imshow('filled', img_filled)
    cv2.waitKey()
    cv2.destroyAllWindows()


imgs = ['60', '82', '114', '150', '205']
for img in imgs:
    execute(f'{img}.bmp')

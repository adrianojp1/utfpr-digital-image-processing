import cv2 
import numpy as np 

ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 20

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''
    h = img.shape[0]
    w = img.shape[1]
    label = 1.0

    components = []
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

                comp_height = comp['B'] - comp['T'] + 1
                comp_width = comp['R'] - comp['L'] + 1

                if comp['n_pixels'] >= n_pixels_min and comp_width >= largura_min and comp_height >= altura_min:
                    components.append(comp)
    
    return components

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

img = cv2.imread ('82.bmp', cv2.IMREAD_GRAYSCALE)

img = cv2.medianBlur(img, 5)

#TODO: Change this parameters
img = cv2.Canny(img, 40, 140)

img = img.astype (np.float32) / 255

# Mantém uma cópia colorida para desenhar a saída.
img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
n_componentes = len (componentes)

print(n_componentes)

for c in componentes:
        cv2.rectangle (img, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))
        
cv2.imshow('final', img)

""" kernel = np.ones((5,5), np.uint8) 
  
img_erosion = cv2.erode(img, kernel, iterations=2)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1) 

cv2.imshow('teste', img_dilation)
  
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

cv2.imshow('gy', gy) 
cv2.imshow('gx', gx) """
  
cv2.waitKey(0) 
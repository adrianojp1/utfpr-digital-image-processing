#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu, Adriano J. Paulichi, Matheus D. de Freitas
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE = 'arroz.bmp'

NEGATIVO = False
THRESHOLD = 0.6
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 20

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''
    return np.where(img > threshold, 1.0, 0.0)

#-------------------------------------------------------------------------------

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


#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================

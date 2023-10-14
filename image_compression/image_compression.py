import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!

def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    matrix_centered = matrix - np.mean(matrix, axis=1, dtype=np.float64)[:,None] # centered rows
    covmat = np.cov(matrix_centered) # calculation is easier since they are centered
    eigenvalues, eigenvectors = np.linalg.eigh(covmat) # each row is eigenvector
    count_vectors = eigenvectors.shape[1]
    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalue = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:,sorted_index]
    eigenvectors_top_p = sorted_eigenvectors[:,:p]
    new_matrix = np.dot(np.transpose(eigenvectors_top_p), matrix_centered) # projection onto a new space
    return eigenvectors_top_p, new_matrix, np.mean(matrix, axis=1, dtype=np.float64)

def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        result_img.append(np.dot(comp[0], comp[1])  + comp[2][:,None])
    return np.clip(np.dstack([result_img[0], result_img[1], result_img[2]]), 0, 255)

def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = [pca_compression(img[...,0], p) , pca_compression(img[...,1], p), pca_compression(img[...,2], p)]
        decomp_img = pca_decompression(compressed)

        axes[i // 3, i % 3].imshow(decomp_img[..., 0])
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    
    bias = np.array([[0], [128], [128]])
    trans_img = bias
    pass
     

    return ...

def rgb2ycbcr(img):
    img = img.astype(np.float64)
    trans = np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]], dtype=np.float64)
    ycbcr = np.dot(img , trans.T)
    ycbcr[:,:,1:3] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(img):
    img = img.astype(np.float64)
    inverse_trans = np.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.77, 0]], dtype=np.float64)
    img[:,:,1:3] -= 128
    rgb = np.dot(img, inverse_trans.T)
    rgb = np.clip(rgb, 0, 255)
    return np.uint8(rgb)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr = rgb2ycbcr(rgb_img)
    cbcr = gaussian_filter(ycbcr[..., 1:3], sigma = 2, radius=10)
    ycbcr[..., 1:3] = cbcr
    gauss_img = ycbcr2rgb(ycbcr)
    pixels = np.array(gauss_img)
    plt.imshow(pixels)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr = rgb2ycbcr(rgb_img)
    y = gaussian_filter(ycbcr[..., 0], sigma = 2, radius=10)
    ycbcr[...,0] = y
    gauss_img = ycbcr2rgb(ycbcr)
    pixels = np.array(gauss_img)
    plt.imshow(pixels)
    plt.savefig("gauss_2.png")

def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    gauss_component = gaussian_filter(component, sigma=10)
    return gauss_component[::2, ::2]

def dct_cosine(u, v, size_block=8):
    row = np.array([np.cos((2 * x + 1) * u * np.pi / 16) for x in range(size_block)])
    col = np.array([np.cos((2 * y + 1) * v * np.pi / 16) for y in range(size_block)])
    return np.outer(np.transpose(row), col)
    
def dct_coef(x):
    x[x > 0] = 1
    x[x == 0] = 1 / np.sqrt(2)
    return x
    
def dct(block):
    sum_cosine = np.array([(block * dct_cosine(u, v)).sum() for u in range(8) for v in range(8)])
    coef = np.fromfunction(lambda u, v: dct_coef(u) * dct_coef(v) / 4, (8, 8))
    return sum_cosine.reshape(8, 8) * coef


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    return np.round(block / quantization_matrix)

def get_scale_factor_s(q):
    if 1 <= q and q < 50:
        return 5000 / q
    elif 50 <= q and q <= 99:
        return 200 - 2 * q
    elif q == 100:
        return 1
    exit(1)
    
def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    s = get_scale_factor_s(q)
    compute = lambda x: (50 + s * x) /100
    own_matrix = compute(default_quantization_matrix).astype(int)
    own_matrix[own_matrix == 0] = 1
    return own_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    zigzag_scan = np.concatenate([np.diagonal(block[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-block.shape[0], block.shape[0])])

    return zigzag_scan
    
def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    comp = []
    count_zeros = 0
    for elem in zigzag_list:
        if elem != 0:
            if count_zeros != 0:
                comp.append(0)
                comp.append(count_zeros)
                count_zeros = 0
            comp.append(elem)
        else:
            count_zeros += 1
    if count_zeros != 0:
        comp.append(0)
        comp.append(count_zeros)
    return comp


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    ycbcr_img = rgb2ycbcr(img).astype(np.float64) # from RGB in YCbCr
    
    y = ycbcr_img[..., 0] - 128
    cb = downsampling(ycbcr_img[..., 1]) - 128 # downsampling Cb
    cr = downsampling(ycbcr_img[..., 2]) - 128 # downsampling Cr
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    h = img.shape[0] // 8
    w = img.shape[1] // 8
    blocks_y = [[y[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8]] for row in range(h) for col in range(w)]
    blocks_cb = [[cb[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8]] for row in range(h // 2) for col in range(w // 2)]
    blocks_cr = [[cr[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8]] for row in range(h // 2) for col in range(w // 2)]

    # transform: dct -> quantization -> zigzag -> compression
    compression_blocks_y = [compression(zigzag(quantization(dct(y), quantization_matrixes[0]))) for y in blocks_y]
    compression_blocks_cb = [compression(zigzag(quantization(dct(color), quantization_matrixes[1]))) for color in blocks_cb]
    compression_blocks_cr = [compression(zigzag(quantization(dct(color), quantization_matrixes[1]))) for color in blocks_cr]
    return [compression_blocks_y, compression_blocks_cb, compression_blocks_cr]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    # Your code here
    decompressed = []
    flag = True
    for elem in compressed_list:
        if flag and elem != 0:
            decompressed = np.append(decompressed, elem)
        elif flag:
            flag = False
        else:
            decompressed = np.append(decompressed, np.zeros(elem))
            flag = True
    return decompressed


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    inverse_index = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
                   [2,  4,  7,  13, 16, 26, 29, 42],
                   [3,  8,  12, 17, 25, 30, 41, 43],
                   [9,  11, 18, 24, 31, 40, 44,53],
                   [10, 19, 23, 32, 39, 45, 52,54],
                   [20, 22, 33, 38, 46, 51, 55,60],
                   [21, 34, 37, 47, 50, 56, 59,61],
                   [35, 36, 48, 49, 57, 58, 62,63]])

    return input[inverse_index]

def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    
    return block * quantization_matrix
    
def inverse_dct_cosine(x, y, size_block=8):
    row = np.array([np.cos((2 * x + 1) * u * np.pi / 16) for u in range(size_block)])
    col = np.array([np.cos((2 * y + 1) * v * np.pi / 16) for v in range(size_block)])
    return np.outer(np.transpose(row), col)
    
def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    coef = np.fromfunction(lambda u, v: dct_coef(u) * dct_coef(v) / 4, (8, 8))
    f = np.array([(coef * block * inverse_dct_cosine(x, y)).sum() for x in range(8) for y in range(8)])
    return np.round(f.reshape(8, 8))



def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    resize_component = np.zeros((2 * component.shape[0],2 *  component.shape[1]))
    resize_component[::2,::2] = component
    resize_component[::2,1::2] = component
    resize_component[1::2,:] = resize_component[::2,:]
    return resize_component


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    h = result_shape[1]
    w = result_shape[0]
    compression_blocks_y = result[0]
    compression_blocks_cb = result[1]
    compression_blocks_cr = result[2]

    blocks_y = [inverse_dct(
        inverse_quantization(
            inverse_zigzag(inverse_compression(y)), quantization_matrixes[0]
            )
        ) for y in compression_blocks_y]
    blocks_cb = [inverse_dct(
        inverse_quantization(
            inverse_zigzag(inverse_compression(color)), quantization_matrixes[1]
            )
        ) for color in compression_blocks_cb]
    blocks_cr = [inverse_dct(
        inverse_quantization(
            inverse_zigzag(inverse_compression(color)), quantization_matrixes[1]
            )
        ) for color in compression_blocks_cr]
    
    # from blocks to channel
    y = np.zeros((h, w))
    cb = np.zeros((h // 2, w // 2))
    cr = np.zeros((h // 2, w // 2))
    for row in range(h // 8):
        for col in range(w // 8):
            y[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8] += np.array(blocks_y[row * w // 8 + col]).reshape(8, 8)
    for row in range(h // 16):
        for col in range(w // 16):
            cb[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8] += np.array(blocks_cb[row * w // 16 + col]).reshape(8, 8)
            cr[row * 8: (row + 1) * 8, col * 8: (col + 1) * 8] += np.array(blocks_cr[row * w // 16 + col]).reshape(8, 8)
            
    ycbcr = np.dstack([y + 128, upsampling(cb + 128), upsampling(cr + 128)]) # upwnsampling Cr and Cb
    return np.clip(ycbcr2rgb(ycbcr), 0, 255).astype(np.uint8)

def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        y_quantization = own_quantization_matrix(y_quantization_matrix, p)
        color_quantization = own_quantization_matrix(color_quantization_matrix, p)
        matrixes = [y_quantization, color_quantization]
        compressed = jpeg_compression(img, matrixes)
        reimg = jpeg_decompression(compressed, img.shape, matrixes)
        axes[i // 3, i % 3].imshow(reimg)

        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    compressed = np.array(compressed, dtype=np.object_)
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()

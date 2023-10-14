import numpy as np

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """


    center = size // 2
    if size % 2 == 0:
        x = np.linspace(-center, center, size + 1, dtype=np.float64)
        y = np.linspace(-center, center, size + 1, dtype=np.float64)
        x = np.delete(x, center)
        y = np.delete(y, center)
    else:
        x = np.linspace(-center, center, size, dtype=np.float64)
        y = np.linspace(-center, center, size, dtype=np.float64)
    g = np.exp(-((x * x).reshape(-1, 1) + y*y).astype(np.float64) / (2 * sigma * sigma), dtype=np.float64)
    return g / g.sum()

def pad_kernel(kernel, target):
    th, tw = target
    kh, kw = kernel.shape[:2]
    ph, pw = th - kh, tw - kw
    padding = [((ph+1) // 2, ph // 2), ((pw+1) // 2, pw // 2)]
    kernel = np.pad(kernel, padding)
    return kernel
    
def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h_shape = h.shape
    h = pad_kernel(h, shape)
    h = np.fft.ifftshift(h)
    h_fourier = np.fft.fft2(h)
    return h_fourier

def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    #print(threshold, np.absolute(H))
    mask = (np.absolute(H) <= threshold)
    H[np.absolute(H) <= threshold] = 1
    H_inv = 1 / H
    H_inv[mask] = 0
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = np.fft.fft2(blurred_img)
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)
    G_estimate = G * H_inv
    f_estimate = np.fft.ifft2(G_estimate)
    return f_estimate.real
    
def wiener_filtering(blurred_img, h, K= 4.6666666666666665e-05):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = np.fft.fft2(blurred_img)
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conj(H)
    H_square = H * H_conj
    G_estimate = G * H_conj / (H_square + K)
    f_estimate = np.fft.ifft2(G_estimate)
    return f_estimate.real


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    
    if np.count_nonzero(img1 - img2):
        tmp = (img1 - img2) / (255 * np.sqrt(np.prod(img1.shape)))
        return - 10 * np.log10((tmp * tmp).sum())
    else:
        raise ValueError()

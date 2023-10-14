import numpy as np

def compute_energy(image):

    image = image.astype(np.float64)
    Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    I = np.pad(Y, ((1, 1), (1, 1)), 'symmetric')
    grad_x = (I[1: -1, 2:] - I[1: -1, :-2])
    grad_y = (I[2:, 1: -1] - I[: -2, 1: -1])
    grad_x[..., 1:-1] = grad_x[..., 1:-1] / 2
    grad_y[1:-1, ...] = grad_y[1:-1, ...] / 2
    grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return grad_norm
    
def compute_seam_matrix(energy, mode, mask=None):
    if mode == 'vertical':
        horizontal= np.zeros((energy.shape), dtype=np.float64)
        horizontal[:, 0:1] = energy[:, 0:1]
        n = energy.shape[1]
        if mask is not None:
            horizontal += (mask * horizontal.shape[0] * horizontal.shape[1] * 256).astype(np.float64)
        for i in range(1, n):
            extend_row = np.pad(horizontal[:, i - 1: i], ((1, 1), (0, 0)), 'symmetric')
            left_neighbor = np.min([extend_row[0: -2], extend_row[1: -1],  extend_row[2:]], axis=0)
            horizontal[:, i: i + 1] += energy[:, i: i + 1] + left_neighbor
        return horizontal
            
    elif mode == 'horizontal':
        vertical = np.zeros((energy.shape), dtype=np.float64)
        vertical[0: 1, ...] = energy[0:1, ...]
        n = energy.shape[0]
        if mask is not None:
            vertical += (mask * vertical.shape[0] * vertical.shape[1] * 256).astype(np.float64)
        for i in range(1, n):
            extend_col = np.pad(vertical[i - 1: i], ((0, 0), (1, 1)), 'symmetric')[0]
            left_neighbor = np.min([extend_col[0: -2], extend_col[1: -1],  extend_col[2:]], axis=0)
            vertical[i: i + 1, :] += energy[i: i + 1, :] + left_neighbor
        return vertical


def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    if mode == 'vertical shrink':
        vertical = np.zeros((seam_matrix.shape), dtype=np.int16)
        n = seam_matrix.shape[1]
        m = seam_matrix.shape[0]
        min_prev = -1
        for i in range(1, n + 1):
            if min_prev == -1:
                min_prev = np.argmin(seam_matrix[:, -i])
                vertical[min_prev][-i] = 1
            else:
                min_now = np.argmin(seam_matrix[max(min_prev - 1, 0) : min(min_prev + 2, m), -i])
                vertical[max(min_prev - 1, 0) + min_now][-i] = 1
                min_prev = max(min_prev - 1, 0) + min_now
        image_cut = np.zeros((image.shape[0] - 1, image.shape[1], 3))
        
        cut = np.argmax(vertical, axis=0)
        for row, col  in enumerate(cut):
            image_cut[:, row, 0] = np.delete(image[:, row, 0], col)
            image_cut[:, row, 1] = np.delete(image[:, row, 1], col)
            image_cut[:, row, 2] = np.delete(image[:, row, 2], col)
        mask_cut = None
        if mask is not None:
            mask_cut = np.zeros((mask.shape[0] - 1, mask.shape[1]), dtype=np.uint8)
            cut = np.argmax(vertical, axis=0)
            for row, col  in enumerate(cut):
                mask_cut[:, row] = np.delete(mask[:, row], col)
        return image_cut.astype(np.uint8), mask_cut, vertical.astype(np.uint8)
    if mode == 'horizontal shrink':
        horizontal = np.zeros((seam_matrix.shape), dtype=np.int16)
        n = seam_matrix.shape[0]
        m = seam_matrix.shape[1]
        min_prev = -1
        for i in range(1, n + 1):
            if min_prev == -1:
                min_prev = np.argmin(seam_matrix[-i, :])
                horizontal[-i][min_prev] = 1
            else:
                min_now = np.argmin(seam_matrix[-i, max(min_prev - 1, 0) : min(min_prev + 2, m)])
                horizontal[-i][max(min_prev - 1, 0) + min_now] = 1
                min_prev = max(min_prev - 1, 0) + min_now
        image_cut = np.zeros((image.shape[0], image.shape[1] - 1, 3))
        seam_matrix_cut = np.zeros((seam_matrix.shape[0], seam_matrix.shape[1] - 1), dtype=np.float64)
        cut = np.argmax(horizontal, axis=1)
        for col, row  in enumerate(cut):
            image_cut[col, :, 0] = np.delete(image[col, :, 0], row)
            image_cut[col, :, 1] = np.delete(image[col, :, 1], row)
            image_cut[col, :, 2] = np.delete(image[col, :, 2], row)
            
        mask_cut = None
        if mask is not None:
            mask_cut = np.zeros((mask.shape[0], mask.shape[1] - 1), dtype=np.uint8)
            cut = np.argmax(horizontal, axis=1)
            for col, row  in enumerate(cut):
                mask_cut[col, :] = np.delete(mask[col, :], row)
        return image_cut.astype(np.uint8), mask_cut, horizontal.astype(np.uint8)
        
def seam_carve(img, mode, mask):
    if mask is not None:
        mask = mask.astype(np.float64)
    energy = compute_energy(img)
    cut_mode = {"horizontal shrink": "horizontal", "vertical shrink": "vertical"}
    seam_matrix = compute_seam_matrix(energy, cut_mode[mode], mask)
    return remove_minimal_seam(img, seam_matrix, mode, mask)
    
    


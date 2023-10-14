import numpy as np
def get_bayer_masks(n_rows, n_cols):
    repetitions = ((n_rows + n_rows % 2) // 2, (n_cols + n_cols % 2) // 2)
    red_mask = np.tile([[False, True], [False, False]], repetitions)
    green_mask = np.tile([[True, False], [False, True]], repetitions)
    blue_mask = np.tile([[False, False], [True, False]], repetitions)
    return np.dstack(
        [red_mask[:n_rows, :n_cols],
        green_mask[:n_rows, :n_cols],
        blue_mask[:n_rows, :n_cols]])

def get_colored_img(raw_img):
    bayer_masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    return np.dstack(
        [np.multiply(raw_img,bayer_masks[:, :, 0]),
        np.multiply(raw_img,bayer_masks[:, :, 1]),
        np.multiply(raw_img,bayer_masks[:, :, 2])])

def conv2D(arr,kernel):
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=kernel.shape[:2]
    view_shape=(1+m1-m2,1+n1-n2,m2,n2)+arr.shape[2:]
    strides=(s0,s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    conv=np.sum(subs*kernel,axis=(2,3))
    return conv
    
def bilinear_interpolation(colored_img):
    n = colored_img.shape[0]
    m = colored_img.shape[1]
    colors = colored_img.shape[2]
    masks = get_bayer_masks(n, m)
    filter = np.ones((3, 3))
    res = []
    for color in range(colors):
        mask = np.pad(conv2D(masks[:,:,color], filter), ((1,1), (1, 1)), constant_values=1)
        img = np.pad(conv2D(colored_img[:,:,color], filter), ((1, 1), (1, 1)), constant_values=0)
        res += [(img // mask).astype(np.uint8) * ~masks[:,:,color] + colored_img[:,:,color]]
    return np.dstack([res[0], res[1], res[2]])
    
def improved_interpolation(raw_img):
    g_at_b_r = np.array([[0, 0, -1/8, 0, 0],
                         [0, 0, 0, 0, 0],
                         [-1/8, 0, 1/2, 0, -1/8],
                         [0, 0, 0, 0, 0],
                         [0, 0, -1/8, 0, 0]])
                         
    g_g = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1/4, 0, 0],
                    [0, 1/4, 0, 1/4, 0],
                    [0, 0, 1/4, 0, 0],
                    [0, 0, 0, 0, 0]])
                    
    color_at_g_in_row = np.array([[0, 0, 1/16, 0, 0],
                                  [0, -1/8, 0, -1/8, 0],
                                  [-1/8, 0, 5/8, 0, -1/8],
                                  [0, -1/8, 0, -1/8, 0],
                                  [0, 0, 1/16, 0, 0]])
                                  
    color_at_g_in_col = color_at_g_in_row.T
    r_b_in_row = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 1/2, 0, 1/2, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
                           
    r_b_in_col = r_b_in_row.T
    r_b = np.array([[0, 0, -3/16, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-3/16, 0, 6/8, 0, -3/16],
                    [0, 0, 0, 0, 0],
                    [0, 0, -3/16, 0, 0]])
                    
    r_b_row_col = np.array([[0, 0, 0, 0, 0],
                            [0, 1/4, 0, 1/4, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1/4, 0, 1/4, 0],
                            [0, 0, 0, 0, 0]])
                            
    colored_img = get_colored_img(raw_img)
    masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    
    r = np.pad(colored_img[..., 0], ((2, 2), (2, 2))).astype(np.float64)
    g = np.pad(colored_img[..., 1], ((2, 2), (2, 2))).astype(np.float64)
    b = np.pad(colored_img[..., 2], ((2, 2), (2, 2))).astype(np.float64)
    
    g_new = ((conv2D(b, g_at_b_r) + conv2D(g, g_g)) * masks[...,2] +
             (conv2D(r, g_at_b_r) + conv2D(g, g_g)) * masks[...,0] + colored_img[...,1])
    r_new = ((conv2D(b, r_b) + conv2D(r, r_b_row_col)) * masks[...,2] +
             (conv2D(g, color_at_g_in_row) + conv2D(r, r_b_in_row)) * np.pad(masks[...,0], ((0, 0), (1, 0)))[:,:-1] +
             (conv2D(g, color_at_g_in_col) + conv2D(r, r_b_in_col)) * np.pad(masks[...,0], ((1, 0), (0, 0)))[:-1,:] +
             colored_img[...,0])
    b_new = ((conv2D(r, r_b) + conv2D(b, r_b_row_col)) * masks[...,0] +
             (conv2D(g, color_at_g_in_row) + conv2D(b, r_b_in_row)) * np.pad(masks[...,2], ((0, 0), (1, 0)))[:,:-1] +
             (conv2D(g, color_at_g_in_col) + conv2D(b, r_b_in_col)) * np.pad(masks[...,2], ((1, 0), (0, 0)))[:-1,:] +
             colored_img[...,2])
    return np.dstack([np.clip(r_new, a_min = 0, a_max=255).astype(np.uint8),
                      np.clip(g_new, a_min = 0, a_max=255).astype(np.uint8),
                      np.clip(b_new, a_min = 0, a_max=255).astype(np.uint8)])

def compute_psnr(img_pred, img_gt):
    if np.count_nonzero(img_pred - img_gt):
        img_pred = img_pred.astype(np.float64)
        img_gt = img_gt.astype(np.float64)
        tmp = (img_pred - img_gt) / (img_gt.max() * np.sqrt(np.prod(img_pred.shape)))
        return - 10 * np.log10((tmp * tmp).sum())
    else:
        raise ValueError()



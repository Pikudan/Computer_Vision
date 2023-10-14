import numpy as np

def mse(I1, I2):

    return np.sum((I1 - I2) * (I1 - I2)) / (I1.shape[0] * I1.shape[1])

def best_shifting(channel1, channel2, x_range, y_range):
    
    best_mse = mse(channel1, channel2)
    best_shift = (0, 0)
    width, height = channel1.shape
    for x in range(x_range[0], x_range[1] + 1):
        for y in range(y_range[0], y_range[1] + 1):
            left_cut, right_cut = max(0, -x), max(0, x)
            top_cut, bot_cut = max(0, -y), max(0, y)
            channel1_shift = channel1[left_cut: width-right_cut, top_cut: height - bot_cut]
            channel2_cut = channel2[right_cut: width-left_cut, bot_cut: height - top_cut]
            metric = mse(channel1_shift, channel2_cut)
            if metric < best_mse:
                best_mse = metric
                best_shift = (x, y)
    return best_shift
            
def pyramid(channel1, channel2):
    if max(channel1.shape) < 500:
        return best_shifting(channel1, channel2, (-10, 10), (-10, 10))
    channel1_new = channel1[::2,::2]
    channel2_new = channel2[::2,::2]
    shift = pyramid(channel1_new, channel2_new)
    return best_shifting(channel1, channel2, (shift[0] * 2 - 1, shift[0] * 2 + 1), (shift[1] * 2 - 1 , shift[1] * 2 + 1))

def align(raw_img, coord):
    img = raw_img.astype(float)
    g_row, g_col = coord
    full_h, full_w = img.shape
    full_h -= full_h % 3
    
    # Разделение на 3 канала
    cur_h = full_h // 3
    cur_w = full_w
    blue = img[:cur_h]
    green = img[cur_h:2 * cur_h]
    red = img[2 * cur_h: full_h]
    
    # Обрезка краев
    k_cut = 0.10
    channel_row_cut = int(cur_h * k_cut)
    channel_col_cut = int(cur_w * k_cut)
    
    blue_cut = blue[channel_row_cut: cur_h - channel_row_cut,
               channel_col_cut: cur_w - channel_col_cut]
    green_cut = green[channel_row_cut: cur_h - channel_row_cut,
              channel_col_cut: cur_w - channel_col_cut]
    red_cut = red[channel_row_cut: cur_h - channel_row_cut,
              channel_col_cut: cur_w - channel_col_cut]
    
    width_shift, height_shift = pyramid(red_cut, green_cut)
    red_shifted = np.roll(red, (width_shift, height_shift), (0, 1))
    
    r_row = g_row - width_shift + raw_img.shape[0] // 3
    r_col = g_col - height_shift
    
    width_shift, height_shift = pyramid(blue_cut, green_cut)
    blue_shifted = np.roll(blue, (width_shift, height_shift), (0, 1))
    
    b_row = g_row - width_shift - raw_img.shape[0] // 3
    b_col = g_col - height_shift
    
    return np.stack((red_shifted, green, blue_shifted), axis=-1).astype(np.uint8), (b_row, b_col), (r_row, r_col)
    
    
    
    
    

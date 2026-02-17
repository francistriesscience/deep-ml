import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    x = x.astype(float)
    N, C, H, W = x.shape
    
    out_h = math.ceil((H - kernel_size) / stride) + 1
    out_w = math.ceil((W - kernel_size) / stride) + 1
    
    needed_h = (out_h - 1) * stride + kernel_size
    needed_w = (out_w - 1) * stride + kernel_size
    
    pad_h = max(0, needed_h - H)
    pad_w = max(0, needed_w - W)
    
    if pad_h > 0 or pad_w > 0:
        x_padded = np.pad(
            x, 
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 
            mode='constant', 
            constant_values=-np.inf
        )
    else:
        x_padded = x

    windows = sliding_window_view(x_padded, window_shape=(kernel_size, kernel_size), axis=(2, 3))
    
    windows_strided = windows[:, :, ::stride, ::stride, :, :]
    
    output = np.max(windows_strided, axis=(-2, -1))
    
    return output

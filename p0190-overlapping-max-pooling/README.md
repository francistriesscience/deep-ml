# Overlapping Max Pooling

## Imagine Taking Photos!

Imagine you have a big picture (Input) and a small square camera frame (Kernel). You want to take photos of the big picture by moving the frame across it.

1.  **Snap!** You take the first photo.
2.  **Slide!** You move the frame just a little bit (Stride).
    - _Uh oh!_ You didn't move it all the way past the first photo. The new photo has some of the same stuff as the first one!
    - This is called **Overlapping**. The photos share some parts.

### What if we run out of picture? (Ceil Mode)

Sometimes, when you get to the edge, your frame hangs off the side! Valid parts of the picture are still inside, but the rest is empty space.

- **Floor Mode (Strict):** "Nope! If the frame doesn't fit perfectly, I won't take the photo!"
- **Ceil Mode (Friendly):** "It's okay! We'll just take the photo of whatever is left and ignore the empty space!"

---

## How the Solution Works

We wrote a Python function `overlapping_max_pool2d` that uses a sliding window approach with padding.

### The Steps:

1.  **Convert to Float:** We first make sure our picture is made of decimal numbers (`float`). This is important because we need to use "negative infinity" (`-inf`) for empty space, and integers can't handle that!
2.  **Calculate Size:** We figure out how many windows will fit, including the partial ones at the end (`Ceil Mode`).
    - Formula: `ceil((Input - Kernel) / Stride) + 1`
3.  **Add Padding:** If the windows hang off the edge, we add "negative infinity" to the input. This is like painting the empty space with a super small number so it doesn't mess up our `max` calculation.
4.  **Slide and Max:**
    - We use `sliding_window_view` to look at every possible window.
    - We jump by our `stride` to pick the ones we want.
    - We find the **Max** value in each window.

### The Code:

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    # 1. Be precise! We need decimals for our special padding trick.
    x = x.astype(float)
    N, C, H, W = x.shape

    # 2. Figure out how many photos we can take (Ceil Mode)
    # Even if the frame hangs off the edge, we count it!
    out_h = math.ceil((H - kernel_size) / stride) + 1
    out_w = math.ceil((W - kernel_size) / stride) + 1

    # 3. Calculate how much extra space (padding) we need
    needed_h = (out_h - 1) * stride + kernel_size
    needed_w = (out_w - 1) * stride + kernel_size

    pad_h = max(0, needed_h - H)
    pad_w = max(0, needed_w - W)

    # 4. Add the padding
    # We use -infinity because it's smaller than any real number.
    # It's like "empty space" that won't win the "Who is Biggest?" contest (Max Pooling).
    if pad_h > 0 or pad_w > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=-np.inf
        )
    else:
        x_padded = x

    # 5. Get all the windows at once!
    # Imagine seeing every possible photo frame position simultaneously.
    windows = sliding_window_view(x_padded, window_shape=(kernel_size, kernel_size), axis=(2, 3))

    # 6. Jump by stride (The "Slide" part)
    # We only keep the photos where we stopped sliding.
    windows_strided = windows[:, :, ::stride, ::stride, :, :]

    # 7. Find the winner (Max) in each photo
    output = np.max(windows_strided, axis=(-2, -1))

    return output
```

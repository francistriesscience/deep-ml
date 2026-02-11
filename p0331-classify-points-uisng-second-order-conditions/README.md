# Classify Critical Points Using Second-Order Conditions

LINK: https://www.deep-ml.com/problems/311

## Imagine You're on a Hill!

Imagine you are walking on a bumpy landscape. You stop walking and look around. You want to know where you are standing. Are you at the very top of a hill? Are you at the very bottom of a valley? Or are you somewhere in between, like on a horse's saddle?

In math, we call this spot a **"Critical Point"**. But just knowing you stopped isn't enough. We need to know the **shape** of the land around you to figure out if you're high up or low down.

We use a special grid of numbers called a **Hessian Matrix** to tell us this shape!

## The "Magic Numbers" (Eigenvalues)

The Hessian Matrix gives us some secret "magic numbers" called **Eigenvalues**. These numbers tell us how the land curves.

Think of the eigenvalues as directions:

- **Positive Number (+):** The land curves **UP** like a smiley face (a bowl).
- **Negative Number (-):** The land curves **DOWN** like a frowny face (a hill).
- **Zero (0):** The land is flat, or we can't tell for sure.

### The 4 Rules:

1.  **Local Minimum (The Valley)**
    - **Magic Numbers:** All are **POSITIVE** (+).
    - **What it means:** Everywhere you look, the land goes UP. You are at the bottom of a bowl!
    - **Result:** `local_minimum`

2.  **Local Maximum (The Peak)**
    - **Magic Numbers:** All are **NEGATIVE** (-).
    - **What it means:** Everywhere you look, the land goes DOWN. You are at the top of a mountain!
    - **Result:** `local_maximum`

3.  **Saddle Point (The Horse Saddle)**
    - **Magic Numbers:** Some are **POSITIVE** (+), and some are **NEGATIVE** (-).
    - **What it means:** In one direction, the land goes UP. In another direction, it goes DOWN. You are sitting in a saddle!
    - **Result:** `saddle_point`

4.  **Inconclusive (Wait, I'm confused!)**
    - **Magic Numbers:** Any of them are **ZERO** (0).
    - **What it means:** The land is flat in one direction, and we can't tell if it's a hill, a valley, or a saddle just by looking here. We need more info!
    - **Result:** `inconclusive`

## How the Solution Works?

We wrote a Python function `classify_critical_point` to do this check for us.

### The Steps:

1.  **Get the Hessian Matrix:** We take the input matrix (grid of numbers).
2.  **Find the Eigenvalues:** We ask the computer to find the "magic numbers" using `np.linalg.eigvalsh`. We use `eigvalsh` because Hessian matrices are special (symmetric), so the math is faster and more accurate!
3.  **Check the Signs:**
    - If any number is `0` (or super close to it), we return `inconclusive`.
    - If all numbers are `> 0`, it's a `local_minimum`.
    - If all numbers are `< 0`, it's a `local_maximum`.
    - If we have both positive and negative numbers, it's a `saddle_point`.

### The Code:

```python
import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10) -> str:
    # 1. Be safe! Make sure it's a proper grid of numbers.
    H = np.asarray(hessian, dtype=float)

    # 2. Find the magic numbers (eigenvalues)!
    eigenvalues = np.linalg.eigvalsh(H)

    # 3. Check for zeros (flat spots)
    if np.any(np.abs(eigenvalues) < tol):
        return "inconclusive"

    # 4. Check generally positive (Bowl shape)
    if np.all(eigenvalues > 0):
        return "local_minimum"

    # 5. Check generally negative (Hill shape)
    if np.all(eigenvalues < 0):
        return "local_maximum"

    # 6. Must be mixed (Saddle shape)
    return "saddle_point"
```

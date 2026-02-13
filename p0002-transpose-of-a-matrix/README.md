# Transpose of a Matrix

LINK: https://www.deep-ml.com/problems/2

## Imagine Your LEGO Tower!

Imagine you built a LEGO tower with 2 layers of 3 bricks each.

- **Top Layer:** Red, Blue, Green 🟥 🟦 🟩
- **Bottom Layer:** Yellow, Orange, Purple 🟨 🟧 🟪

Right now, it's a wide tower. But what if you wanted to tip it on its side to make it a tall tower?

That's exactly what **Transposing** a matrix is! It's like taking your sideways rows and turning them into standing-up columns.

---

## Tipping the Blocks

1. **The Rows (Horizontal):**
   - Row 1: `[1, 2, 3]`
   - Row 2: `[4, 5, 6]`

2. **The Columns (Vertical):**
   - The first numbers in each row (`1` and `4`) become the new **first row**.
   - The middle numbers in each row (`2` and `5`) become the new **second row**.
   - The last numbers in each row (`3` and `6`) become the new **third row**.

**Old Wide Tower (2x3):**

```
1  2  3
4  5  6
```

**New Tall Tower (3x2):**

```
1  4
2  5
3  6
```

---

## How the Solution Works

We wrote a Python function `transpose_matrix` that uses a neat trick to flip the matrix.

### The Magic Trick: `zip(*a)`

1. **Unpacking (`*a`):** We take all the rows out of the matrix "envelope".
2. **Zipping (`zip`):** We grab the first items from every row and group them together. Then we do the same for the second items, and so on.
3. **Listing (`list`):** We turn those new groups back into a list of lists.

### The Code:

```python
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    # The zip(*) trick flips the matrix automatically!
    # It turns row 1, 2, 3 and row 4, 5, 6 into:
    # (1, 4), (2, 5), and (3, 6)
    return [list(row) for row in zip(*a)]
```

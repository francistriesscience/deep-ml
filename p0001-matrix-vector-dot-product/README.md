# Matrix-Vector Dot Product

LINK: https://www.deep-ml.com/problems/1

## Imagine You're Shopping!

Imagine you and your friends all have shopping lists.

- **You (Row 1):** 1 Apple 🍎, 2 Bananas 🍌
- **Your Friend (Row 2):** 2 Apples 🍎, 4 Bananas 🍌

And you know the prices:

- **Price List (The Vector):** Apple costs $1, Banana costs $2.

We want to find out **how much each person has to pay in total**.

This is what a **Matrix-Vector Dot Product** does! It takes a list of lists (the shopping carts) and a single list of numbers (the prices), matches them up, multiplies them, and adds them together to get the total for each cart.

## The Math Magic

1.  **The Matrix (The Shopping Carts):** A big list of lists. Each inner list is like one person's shopping cart.

    ```python
    shopping_carts = [
        [1, 2],  # Your cart: 1 Apple, 2 Bananas
        [2, 4]   # Friend's cart: 2 Apples, 4 Bananas
    ]
    ```

2.  **The Vector (The Price List):** A single list of numbers.

    ```python
    prices = [1, 2]  # Apple cost, Banana cost
    ```

3.  **The Dot Product (The Checkout):**
    - **For You:** (1 Apple × $1) + (2 Bananas × $2) = $1 + $4 = **$5**
    - **For Friend:** (2 Apples × $1) + (4 Bananas × $2) = $2 + $8 = **$10**

    The result is a new list of totals: `[5, 10]`.

### The 1 Big Rule

**The lists must match in size!**
If your shopping list has 3 items (Apple, Banana, Orange), but the price list only has 2 prices (Apple, Banana), the cashier gets confused!

- If the sizes don't match, we return `-1`.

## How the Solution Works

We wrote a Python function `matrix_dot_vector` to do this calculation for us.

### The Steps:

1.  **Check Dimensions:** First, we make sure the "shopping list" length matches the "price list" length. If `len(matrix_row) != len(vector)`, we return `-1`.
2.  **Iterate Through Rows:** We look at each row (person's cart) in the matrix one by one.
3.  **Multiply and Sum (Zip):**
    - We use `zip(row, b)` to pair up the items: (1st item with 1st price), (2nd item with 2nd price).
    - We multiply each pair.
    - We add all the results together (`sum`).
4.  **Create Result List:** We collect all the totals into a new list and return it.

### The Code:

```python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # 1. Check if the inputs are empty
    if not a or not b:
        return -1

    # 2. Check the Rule: Row length must equal Vector length
    if len(a[0]) != len(b):
        return -1

    # 3. The Computation:
    # We use a "list comprehension" to do it all in one go!
    # For every row in the matrix 'a':
    #   - Pair it with vector 'b'
    #   - Multiply the pairs
    #   - Sum them up
    return [sum(x * y for x, y in zip(row, b)) for row in a]
```

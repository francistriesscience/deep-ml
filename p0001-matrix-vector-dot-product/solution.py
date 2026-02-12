def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if not a or not b:
        return -1
        
    if len(a[0]) != len(b):
        return -1
        
    return [sum(x * y for x, y in zip(row, b)) for row in a]

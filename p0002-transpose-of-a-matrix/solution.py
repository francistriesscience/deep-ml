def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(row) for row in zip(*a)]

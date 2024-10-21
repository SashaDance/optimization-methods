import random
from typing_extensions import Self


class Matrix:
    def __init__(self, n=None, m=None, matrix=None):
        if matrix is not None:
            self.matrix = matrix
            # checking if matrix shape is (m, 1), i. e. it is a list
            self.check_matrix()
            self.m = len(matrix[0])
            self.n = len(matrix)
        if n is not None and m is not None:
            self.n = n
            self.m = m
            self.matrix = self.fill_zeros(n, m)

    def __len__(self):
        return self.n

    def __call__(self):
        return self.matrix

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __mul__(self, multiplier: Self | float | int) -> Self:

        if isinstance(multiplier, Matrix):
            result = Matrix(self.n, multiplier.m)
            if self.m == multiplier.n:
                for i in range(self.n):
                    for j in range(multiplier.m):
                        for k in range(self.m):
                            result[i][j] += self.matrix[i][k] * multiplier[k][
                                j]
            else:
                raise ValueError(
                    'Unsupported matrices shapes for multiplication'
                )

        elif isinstance(multiplier, (int, float)):
            result = Matrix(self.n, self.m)
            for i in range(self.n):
                for j in range(self.m):
                    result[i][j] = self.matrix[i][j] * multiplier
        else:
            raise TypeError(
                'Unsupported operand type: Matrix or float or int expected'
            )

        return result

    def __add__(self, other: Self) -> Self:
        if self.m == other.m and self.n == other.n:
            result = Matrix(self.n, self.m)
            for i in range(self.n):
                for j in range(self.m):
                    result[i][j] = self[i][j] + other[i][j]
        else:
            raise ValueError('Matrices shapes should be the same')
        return result

    def __sub__(self, other: Self) -> Self:
        return self + other * (-1)

    def __str__(self):
        return '\n'.join(" ".join(map(str, row)) for row in self.matrix)

    def check_matrix(self) -> None:
        m = len(self.matrix[0])
        for row in self.matrix:
            if len(row) == m:
                continue
            else:
                raise ValueError(
                    'Invalid matrix: Rows have inconsistent lengths')

    def add_row(self, row: list) -> None:
        if isinstance(row, list):
            if len(row) == self.m:
                self.matrix.append(row)
                self.n += 1
            else:
                raise ValueError(
                    f'Invalid length of row: len(row) should be {self.m}')
        else:
            raise TypeError(
                'Appending row should be a list of float'
            )

    def add_column(self, column: list) -> None:
        if isinstance(column, list):
            if len(column) == self.n:
                for i in range(self.n):
                    self.matrix[i].append(column[i])
                    self.m += 1
            else:
                raise ValueError(
                    f'Invalid length of column: len(column) should be {self.m}')
        else:
            raise TypeError(
                'Appending column should be a list of float'
            )

    @staticmethod
    def transpose(matrix: Self) -> Self:
        n = matrix.n
        m = matrix.m
        result = Matrix(m, n)
        for i in range(n):
            for j in range(m):
                result[j][i] = matrix[i][j]

        return result

    @staticmethod
    def fill_zeros(n: int, m: int) -> Self:

        matrix = [[0 for j in range(m)] for i in range(n)]
        return matrix

    @staticmethod
    def create_random_matrix(n: int, m: int, segment: tuple = (0, 1)) -> Self:
        matrix = Matrix(n, m)
        for i in range(n):
            for j in range(m):
                matrix[i][j] = random.uniform(segment[0], segment[1])

        return matrix

    @staticmethod
    def identity_matrix(n: int):
        identity = Matrix(n, n)
        for i in range(n):
            identity[i][i] = 1
        return identity

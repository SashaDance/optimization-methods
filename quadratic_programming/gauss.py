from matrix import Matrix

def gauss(n: int, matrix: Matrix) -> Matrix:
    solution = Matrix(1, n)
    # straight Gauss
    # finding the greatest element in i'th column
    for i in range(n):
        max_elem = matrix[i][i]
        max_row = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(max_elem):
                max_elem = matrix[j][i]
                max_row = j

        # exchanging max_row and i'th row
        for j in range(i, n + 1):
            '''
            n + 1 because we need to change
            the right sight of the equation as well
            '''
            h = matrix[i][j]
            matrix[i][j] = matrix[max_row][j]
            matrix[max_row][j] = h

        # dividing i'th row by max_elem
        h = max_elem
        for j in range(i, n + 1):
            matrix[i][j] = matrix[i][j] / h

        '''
        making zeros below the i'th diagonal element,
        which is 1 by that time
        '''
        for k in range(i + 1, n):
            h = matrix[k][i]
            for j in range(i, n + 1):
                matrix[k][j] = matrix[k][j] - matrix[i][j] * h

    # backwards Gauss
    solution[0][n - 1] = matrix[n - 1][n]
    for i in range(n - 1):
        k = n - i - 2
        cum_sum = 0
        for j in range(k + 1, n):
            cum_sum = cum_sum + solution[0][j] * matrix[k][j]
        solution[0][k] = matrix[k][n] - cum_sum

    return solution


def check_huge_system(n: int = 1024) -> None:
    matrix = Matrix.create_random_matrix(n, n)

    column = Matrix(matrix=[[i + 1] for i in range(n)])
    right_sight = matrix * column

    right_sight = [elem[0] for elem in right_sight.matrix]
    matrix.add_column(right_sight)

    print(gauss(1024, matrix))


if __name__ == '__main__':
    # print('1024 x 1024 system:')
    # check_huge_system()

    matrix = [
        [-1, 3, 2],
        [3, -3, 3],
        [2, 3, -3]
    ]

    matrix_instance = Matrix(matrix=matrix)
    right_sight = [2, 9, 6]
    matrix_instance.add_column(right_sight)

    print('3 x 3 system')
    print(gauss(3, matrix_instance).matrix[0])

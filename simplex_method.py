import numpy as np

M = int(1e12)


class SimplexMethod:
    """
    Simplex method solver when all constraints are equalities
    """
    def __init__(self, A: list[list[int]], b: list[int], c: list[int]):
        """
        :param A: matrix with constraints coefficients (m x n)
        :param b: vector with right sights of the constraints (m x 1)
        :param c: vector with coefficients in the cost function (n x 1)
        """
        self.A = A
        self.b = b
        self.c = c
        self.__check_valid_input()
        self.m = len(A)
        self.n = len(A[0])
        self.__process_inputs()

    def __check_valid_input(self) -> None:
        """
        Checks shapes of the inputs
        """
        assert len(self.A) == len(self.b), \
            f'Invalid input: got {len(self.A)} rows in A matrix and {len(self.b)} rows in b vector, must be the same'
        assert len(self.A[0]) == len(self.c), \
            f'Invalid input: got {len(self.A[0])} columns in A matrix and {len(self.c)} rows in c vector, must be the same'

    def __process_inputs(self) -> None:
        """
        Prepares inputs and adds artificial variables
        """
        # coefficients matrix with right sights
        self.A = np.concatenate(
            [
                np.array(self.A, dtype=np.float64),
                np.identity(self.m, dtype=np.float64),  # artificial variables
                np.array(self.b, dtype=np.float64).reshape(self.m, 1)
            ],
            axis=1
        )
        self.c = np.concatenate(
            [
                np.array(self.c, dtype=np.float64),
                -M * np.ones(self.m + 1, dtype=np.float64)  # artificial variables
            ],
            axis=0
        )

    def _step(self) -> bool:
        # selecting variable to append to basis
        print(f'Current basis: {self._basis_indices}')
        # updating deltas
        self.deltas = (
            self.c
            - self.c[self._basis_indices] @ self.A
        )
        print(f'Deltas: {self.deltas}')
        new_var_ind = self.deltas[:-1].argmax()  # variable to append
        # selecting variable to remove from basis
        if self.deltas[new_var_ind] <= 0:
            print('Loop was stopped: all deltas are negative now')
            return False
        print(f'New variable: {new_var_ind}')
        ratios = np.where(
            self.A[:, new_var_ind] != 0,
            self.A[:, -1] / self.A[:, new_var_ind],
            0
        )
        if (ratios <= 0).all():
            print('Loop was stopped: the problem is unbounded')
            return False
        ratios = np.where(ratios <= 0, 1e12, ratios)  # handling non positive values
        pivot_row = ratios.argmin()
        old_var_ind = self._basis_indices[pivot_row]  # variable to remove
        print(f'Old variable:{old_var_ind}')
        print(f'pivot row: {pivot_row}')
        # updating basis variables
        self._basis_indices[pivot_row] = new_var_ind
        # updating coefficients
        self.A[pivot_row, :] /= self.A[pivot_row, new_var_ind]
        for i in range(self.m):
            if i == pivot_row:
                continue
            else:
                self.A[i, :] = (
                    self.A[i, :]
                    - self.A[i, new_var_ind] * self.A[pivot_row, :]
                )
        print(f'A: {self.A}')

        return True

    def solve(self, print_solution: bool = True):
        # initializing basis variables with artificial variables
        self._basis_indices = [self.n + i for i in range(self.m)]
        while True:
            if not self._step():
                result = (
                    self.c[self._basis_indices]
                    @ self.A[:, -1]
                )
                variables_values = self.A[:, -1]
                if print_solution:
                    print(f'Problem solved successfully ')
                    print(result)
                    print(variables_values, self._basis_indices)
                    return


if __name__ == '__main__':
    m, n = [int(_) for _ in input().split()]

    A = []
    for i in range(m):
        A.append([int(_) for _ in input().split()])

    b = [int(_) for _ in input().split()]
    c = [int(_) for _ in input().split()]

    solver = SimplexMethod(A, b, c)
    solver.solve()

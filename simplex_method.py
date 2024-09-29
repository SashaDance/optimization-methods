import numpy as np
from typing import Optional

M = int(1e12)


class SimplexMethod:
    """
    Simplex method solver when all constraints are equalities
    """
    def __init__(self, A: list[list[int]], b: list[int], c: list[int]) -> None:
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
        self.m = len(self.b)
        self.n = len(self.c)
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

    def _step(self) -> int:
        """
        Performs one step of simplex method, returns the execution code
        0 - algorithm is not finished yet
        1 - successfully solved
        2 - problem is unbounded
        3 - constraints of problem are incompatible
        :return: execution code
        """
        # updating deltas
        self.deltas = (
            self.c
            - self.c[self._basis] @ self.A
        )
        new_var_ind = self.deltas[:-1].argmax()  # variable to append
        # selecting variable to remove from basis
        if self.deltas[new_var_ind] <= 0:
            # checking that there are no artificial variables left in basis
            if any(basis_var >= self.n for basis_var in self._basis):
                return 3
            return 1
        # ignoring zero division warnings
        with np.errstate(divide='ignore'):
            ratios = np.where(
                self.A[:, new_var_ind] != 0,
                self.A[:, -1] / self.A[:, new_var_ind],
                0
            )
        if (ratios <= 0).all():
            return 2
        ratios = np.where(ratios <= 0, 1e12, ratios)  # handling non positive values
        pivot_row = ratios.argmin()  # index of variable to remove
        # updating basis variable
        self._basis[pivot_row] = new_var_ind
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

        return 0

    def __print_solution(self, solution: list[tuple[int, float]]) -> None:
        print('Solution:')
        ind = 0
        for i in range(self.n):
            if ind < len(solution) and i == solution[ind][0]:
                print(f'{i + 1}: {round(solution[ind][1], 5)} ')
                ind += 1
            else:
                print(f'{i + 1}: 0')

    def solve(self,
              print_solution: bool = True) -> Optional[list[tuple[int, float]]]:
        # initializing basis variables with artificial variables
        self._basis = [self.n + i for i in range(self.m)]
        code = 0
        # algorithm loop
        while not code:
            code = self._step()

        if code == 2:
            print('Problem is unbounded')
            return None
        elif code == 3:
            print('Constraints of problem are incompatible')
            return None
        else:
            result = (
                self.c[self._basis]
                @ self.A[:, -1]
            )
            variables_values = self.A[:, -1]
            solutions = list(
                zip(self._basis, variables_values)
            )
            solutions = sorted(
                solutions, key=lambda x: x[0]
            )
            if print_solution:
                print(f'Problem solved successfully')
                print(f'Function value is {round(result, 5)}')
                self.__print_solution(solutions)
            return solutions


if __name__ == '__main__':
    m, n = [int(_) for _ in input().split()]

    A = []
    for i in range(m):
        A.append([int(_) for _ in input().split()])

    b = [int(_) for _ in input().split()]
    c = [int(_) for _ in input().split()]

    solver = SimplexMethod(A, b, c)
    solver.solve()

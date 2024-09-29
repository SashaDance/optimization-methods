import numpy as np
from typing import Optional

M = int(1e12)


class SimplexMethod:
    """
    Simplex method solver using artificial variables
    """
    def __init__(self, constrains: list[str],
                 c: list[float], mode: str) -> None:
        """
        :param constrains: list of constrains of the following format:
        1 -1 3 < 3
        1 -2 5 > 5
        1 4 2 = 1
        :param c: vector with coefficients in the cost function
        :param mode: mode to solve a problem (max/min)
        """
        self.constrains = constrains
        self.mode = mode
        self.c = c
        self.__process_input()
        self.__add_artificial_variables()

    def __process_input(self) -> None:
        """
        Parses inputs and converts problem to canonical form:
        all inequalities in constrains replaced with equalities,
        minimum problem converted to maximum problem
        """
        # converting min to max
        if self.mode == 'min':
            self.c = [(-1) * num for num in self.c]
        self.num_init_vars = len(self.c)
        symbols = ['=', '>', '<']
        self.A = []  # constraints coefficients matrix
        self.b = []  # right sights of the constraints
        num_residues = len([
            1 for _ in self.constrains if '>' in _ or '<' in _
        ])
        cnt = 0
        for constraint in self.constrains:
            for ch in symbols:
                if ch in constraint:
                    symb = ch
                    break
            row, right_sight = constraint.split(symb)
            row = [float(num) for num in row[:-1].split()]
            right_sight = float(right_sight[1:])
            # appending residues
            if symb == '<':
                for i in range(num_residues):
                    if i == cnt:
                        row.append(1.0)
                    else:
                        row.append(0.0)
                cnt += 1
                self.c.append(0.0)
            elif symb == '>':
                for i in range(num_residues):
                    if i == cnt:
                        row.append(-1.0)
                    else:
                        row.append(0.0)
                cnt += 1
                self.c.append(0.0)
            else:
                for i in range(num_residues):
                    row.append(0.0)
            self.A.append(row)
            self.b.append(right_sight)
        self.m = len(self.b)
        self.n = len(self.c)


    def __add_artificial_variables(self) -> None:
        """
        Adds artificial variables to problem in a canonical form
        """
        # coefficients matrix with artificial variables and right sights
        self.A = np.concatenate(
            [
                np.array(self.A, dtype=np.float64),
                np.identity(self.m, dtype=np.float64),  # artificial variables
                np.array(self.b, dtype=np.float64).reshape(self.m, 1)
            ],
            axis=1
        )
        # adding artificial variables and column for the function value
        self.c = np.concatenate(
            [
                np.array(self.c, dtype=np.float64),
                -M * np.ones(self.m + 1, dtype=np.float64)
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
        print('Initial variables:')
        for i in range(self.n):
            if i == self.num_init_vars:
                print('Residues:')
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
            result = result * (-1) if self.mode == 'min' else result
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
    constrains = []
    while True:
        row = input()
        if not any(symb in row for symb in ['=', '>', '<']):
            break
        constrains.append(row)

    c = [float(_) for _ in row.split()]
    mode = input()

    solver = SimplexMethod(constrains, c, mode)
    solver.solve()

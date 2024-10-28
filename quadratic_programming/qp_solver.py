import sympy as sp
import numpy as np
import re
import itertools
from typing import Optional
from gauss import gauss
from matrix import Matrix


class QPSolver:
    """
    Implementation of quadratic programming solver
    Time complexity: O( (n + m)^3 * 2 ^ (n + m) )
    where n is the number of variables and m is the number of conditions
    """
    def __init__(self, n: int,
                 m: int,
                 function: str,
                 mode: str,
                 constraints: list[str],
                 print_lagr_system: bool = False):
        self.n = n
        self.m = m
        self.mode = mode
        self.constraints = constraints
        self.f_str = function
        self.__process_constraints()
        self.__get_diff_equations()
        if print_lagr_system:
            self.__print_system()

    def __get_diff_equations(self) -> None:
        """
        Returns equations that are a derivatives of the Lagrangian function
        """
        # getting coefficients for initial n variables
        self.x_vars = sp.symbols(
            ' '.join([f'x{i + 1}' for i in range(self.n)])
        )
        self.f_str_new = self.f_str[:]
        for i in range(self.n):
            self.f_str_new = self.f_str_new.replace(
                f'x{i + 1}', f'self.x_vars[{i}]'
            )

        f = eval(self.f_str_new)
        coeffs = []
        consts = []
        self.diff_right_sights = []
        for var_ in self.x_vars:
            diff_coeffs = self.parse_linear_equation(
                var_name='x',
                equation=str(sp.diff(f, var_)),
                n=self.n,
                extr_const=True
            )
            # converting max problem to min problem
            diff_coeffs = (
                [-num for num in diff_coeffs] if self.mode == 'max'
                else diff_coeffs
            )
            coeffs.append(diff_coeffs[:-1])
            self.diff_right_sights.append((-1) * diff_coeffs[-1])
        self.diff_right_sights = np.array(self.diff_right_sights)
        self.diff_equations = np.array(coeffs)
        # adding residual variables
        self.diff_equations = np.concatenate(
            [
                self.diff_equations,
                np.zeros(
                    shape=(len(self.diff_equations), self.m),
                    dtype=np.int64
                )
            ],
            axis=1
        )
        # adding lambda coefficients
        self.diff_equations = np.concatenate(
            [
                self.diff_equations,
                self.A[:, :self.n].T  # transpose first n columns of A
            ],
            axis=1
        )
        # adding mu coefficients
        self.diff_equations = np.concatenate(
            [
                self.diff_equations,
                (-1) * np.identity(n=self.n, dtype=np.int64)
            ],
            axis=1
        )

    def __process_constraints(self) -> None:
        """
        Parses inputs and converts problem to canonical form:
        all inequalities in constrains replaced with equalities
        """
        self.A = []  # constraints coefficients matrix
        self.b = []  # right sights of the constraints
        num_residues = self.m
        cnt = 0
        for constraint in self.constraints:
            row, right_sight = constraint.split('<')
            row = [int(num) for num in row[:-1].split()]
            right_sight = int(right_sight[1:])
            # appending residues
            for i in range(num_residues):
                if i == cnt:
                    row.append(1)
                else:
                    row.append(0)
            cnt += 1
            self.A.append(row)
            self.b.append(right_sight)
        self.A, self.b = np.array(self.A), np.array(self.b)
        # adding lambda and mu coefficients
        self.A = np.concatenate(
            [
                self.A,
                np.zeros(shape=(len(self.A), self.n + self.m), dtype=np.int64)
            ],
            axis=1
        )

    def __print_system(self) -> None:
        """
        Prints system that will be solved
        """
        # diff equations
        all_right_sights = np.concatenate([
            self.diff_right_sights, self.b
        ])
        init_system = np.concatenate([self.diff_equations, self.A])
        for ind, row in enumerate(init_system):
            x_ = ' + '.join(
                f'{coef} * x_{i + 1}' for i, coef in enumerate(row[:self.n])
            )
            xr_ = ' + '.join(
                f'{coef} * xr_{i + 1}' for i, coef in
                enumerate(row[self.n: self.n + self.m])
            )
            lambda_ = ' + '.join(
                f'{coef} * lambda_{i + 1}' for i, coef in
                enumerate(row[self.n + self.m: self.n + 2 * self.m])
            )
            mu_ = ' + '.join(
                f'{coef} * mu_{i + 1}' for i, coef in
                enumerate(row[self.n + self.m: self.n + 2 * self.m:])
            )
            print(
                x_ + ' + ' + xr_
                + ' + ' + lambda_
                + ' + ' + mu_
                + f' = {all_right_sights[ind]}'
            )

    @staticmethod
    def parse_linear_equation(var_name: str,
                              equation: str,
                              n: int,
                              extr_const: bool = True) -> list[int]:
        """
        Returns list of coefficients
        from string representation of linear equation
        :param var_name: name of variable (x1 + x2 then name is x)
        :param equation: string representation of equation
        :param n: number of variables
        :param extr_const: whether append constant term to answer or not
        :return: list of coefficients
        """
        # prepare regex to find all coefficients and the variable indices
        pattern = rf'([+-]?\d*)\*?{var_name}(\d+)'

        # find all variable coefficients
        matches = re.findall(pattern, equation.replace(" ", ""))

        # initialize a list for coefficients
        coefficients = [0] * n

        # fill the list with the coefficients of variables
        for match in matches:
            coeff, var_index = match
            var_index = int(var_index) - 1  # Adjust for zero-indexing

            # handle cases where coefficient is implicit (e.g. +x1 is +1)
            if coeff == '+' or coeff == '':
                coeff = 1
            elif coeff == '-':
                coeff = -1
            else:
                coeff = int(coeff)

            coefficients[var_index] = coeff

        if extr_const:
            # extract the constant term
            constant = re.sub(
                rf'[+-]?\d*\*?{var_name}\d+', '', equation.replace(" ", "")
            )
            if constant:
                constant_term = int(constant)
            else:
                constant_term = 0

            # append constant term to the coefficient list
            coefficients.append(constant_term)

        return coefficients

    @staticmethod
    def calculate_func_val(func: str, x_vec: list[float]) -> float:
        """
        :param func: string representation of function
        :param x_vec: point to evaluate function at
        :return: function value in given point
        """
        for ind, x in enumerate(x_vec):
            func = func.replace(f'x{ind + 1}', str(x))
        return eval(func)

    def solve(self, print_solution: bool) -> Optional[list[float]]:
        """
        :param print_solution: whether print solution or not
        :return: optimal x vector for given problem
        """
        for i in range(2 ** (self.n + self.m)):
            # selecting x and xr that are zero
            x_mask = np.array(
                [int(_) for _ in bin(i)[2:].zfill(self.n + self.m)],
                dtype=np.int64
            )
            # getting x that are zero
            x_indices = itertools.compress(
                list(range(self.n + self.m)),
                x_mask
            )
            # getting mu that are zero
            mu_indices = itertools.compress(
                list(range(self.n)),
                1 - x_mask[:self.n]  # inverting bytes
            )
            # getting lambda that are zero
            lambda_indices = itertools.compress(
                list(range(self.m)),
                1 - x_mask[self.n:]  # inverting bytes
            )
            # creating new equations corresponding to variables that are zero
            new_equations = []
            first_indices = (0, self.n + self.m, self.n + 2 * self.m)
            all_indices = list(map(
                lambda x: list(x),
                [x_indices, lambda_indices, mu_indices]
            ))
            for first_ind, indices in zip(first_indices, all_indices):
                for ind in indices:
                    new_row = [0] * (2 * (self.n + self.m))
                    new_row[first_ind + ind] = 1
                    new_equations.append(new_row)
            # adding new equations
            new_equations = np.array(new_equations, dtype=np.int64)
            new_right_sights = np.array([0] * len(new_equations), dtype=np.int64)
            right_sights = np.concatenate([
                self.diff_right_sights, self.b, new_right_sights
            ])
            system = np.concatenate([
                self.diff_equations, self.A, new_equations
            ])
            # solving systems
            matr = Matrix(matrix=system.tolist())
            matr.add_column(right_sights.tolist())
            try:
                cur_solution = gauss(len(matr), matr).matrix[0]
            except ZeroDivisionError:
                # matrix is singular
                cur_solution = [-1]
            # checking if solution was reached
            eps = -1e-9
            if all(num >= eps for num in cur_solution):
                sol = cur_solution[:]
                if print_solution:
                    print('Problem solved successfully')
                    print(f'x vector: {sol}')
                    val = self.calculate_func_val(self.f_str, sol)
                    val = val
                    print(f'Function value: {val}')
                return sol

        if print_solution:
            print('Problem is unsolvable')
            return None


if __name__ == '__main__':
    n, m = [int(_) for _ in input().split()]
    func = input()
    mode = input()
    constraints = []
    for i in range(m):
        constraints.append(input())

    solver = QPSolver(
        n=n,
        m=m,
        function=func,
        mode=mode,
        constraints=constraints
    )
    solver.solve(print_solution=True)

import sympy as sp
import numpy as np
import re


class QPSolver:
    def __init__(self, n: int,
                 m: int,
                 function: str,
                 mode: str,
                 constraints: list[str]):
        self.n = n
        self.m = m
        self.mode = mode
        self.constraints = constraints
        self.f_str = function
        self.__process_constraints()
        self.__get_diff_equations()
        self.__print_system()

    def __get_diff_equations(self) -> None:
        # getting coefficients for initial n variables
        self.x_vars = sp.symbols(
            ' '.join([f'x{i + 1}' for i in range(self.n)])
        )
        for i in range(self.n):
            self.f_str = self.f_str.replace(f'x{i + 1}', f'self.x_vars[{i}]')

        f = eval(self.f_str)
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
            diff_coeffs = (
                [-num for num in diff_coeffs] if self.mode == 'max'
                else diff_coeffs
            )
            coeffs.append(diff_coeffs[:-1])
            self.diff_right_sights.append((-1) * diff_coeffs[-1])
        self.diff_right_sights = np.array(self.diff_right_sights)
        self.diff_equations = np.array(coeffs)
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
                (-1) * np.identity(self.n, dtype=np.int64)
            ],
            axis=1
        )

    def __print_system(self) -> None:
        # diff equations
        for ind, row in enumerate(self.diff_equations):
            x_ = ' + '.join(
                f'{coef} * x_{i + 1}' for i, coef in enumerate(row[:self.n])
            )
            lambda_ = ' + '.join(
                f'{coef} * lambda_{i + 1}' for i, coef in enumerate(row[self.n: self.n + self.m])
            )
            mu_ = ' + '.join(
                f'{coef} * mu_{i + 1}' for i, coef in enumerate(row[self.n + self.m:])
            )
            print(x_ + ' + ' + lambda_ + ' + ' + mu_ + f' = {self.diff_right_sights[ind]}')
        # constrains equations
        for ind, row in enumerate(self.A):
            x_ = ' '.join(
                f'{coef} * x_{i + 1}' for i, coef in enumerate(row[:self.n])
            )
            xr_ = ' '.join(
                f'{coef} * xr_{i + 1}' for i, coef in enumerate(row[self.n:])
            )
            print(x_ + ' + ' + xr_ + f' = {self.b[ind]}')
    def __process_constraints(self) -> None:
        symbols = ['=', '>', '<']
        self.A = []  # constraints coefficients matrix
        self.b = []  # right sights of the constraints
        num_residues = len([
            1 for _ in self.constraints if '>' in _ or '<' in _
        ])
        cnt = 0
        for constraint in self.constraints:
            for ch in symbols:
                if ch in constraint:
                    symb = ch
                    break
            row, right_sight = constraint.split(symb)
            row = [int(num) for num in row[:-1].split()]
            right_sight = int(right_sight[1:])
            # appending residues
            if symb == '<':
                for i in range(num_residues):
                    if i == cnt:
                        row.append(1)
                    else:
                        row.append(0)
                cnt += 1
            elif symb == '>':
                for i in range(num_residues):
                    if i == cnt:
                        row.append(-1)
                    else:
                        row.append(0)
                cnt += 1
            else:
                for i in range(num_residues):
                    row.append(0)
            self.A.append(row)
            self.b.append(right_sight)
        self.A, self.b = np.array(self.A), np.array(self.b)

    @staticmethod
    def parse_linear_equation(var_name: str,
                              equation: str,
                              n: int,
                              extr_const: bool = True) -> list[int]:
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


import sympy as sp
import numpy as np
import re


class QPSolver:
    def __init__(self, n: int,
                 m: int,
                 function: str,
                 mode: str,
                 constrains: list[str]):
        self.n = n
        self.m = m
        self.constrains = constrains
        self.f_str = function
        # self.__process_constraints()
        self.__get_lagr_coef()

    def __get_lagr_coef(self) -> list[list[int]]:
        self.x_vars = sp.symbols(
            ' '.join([f'x{i + 1}' for i in range(self.n)])
        )
        for i in range(self.n):
            self.f_str = self.f_str.replace(f'x{i + 1}', f'self.x_vars[{i}]')

        f = eval(self.f_str)
        coeffs = []
        for var_ in self.x_vars:
            diff_coeffs = self.parse_linear_equation(
                var_name='x',
                equation=str(sp.diff(f, var_)),
                n=self.n,
                extr_const=False
            )
            coeffs.append(diff_coeffs)

        print(coeffs)
        return coeffs

    def __process_constraints(self) -> None:
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
        self.m = len(self.b)

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

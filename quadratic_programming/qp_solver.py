import sympy as sp
import numpy as np
import re

class QPSolver:
    def __init__(self, function: str, constrains: list[str]):
        pass





x1, x2 = sp.symbols('x1 x2', real=True)
y = eval('x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 - 2 * x1 - 6 * x2')
dy = sp.diff(y, x2)
print(type(y))
print(dy)
print(str(dy))


def parse_linear_equation(equation: str, n: int) -> list[int]:
    # prepare regex to find all coefficients and the variable indices
    pattern = r'([+-]?\d*)\*?x(\d+)'

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

    # extract the constant term
    constant = re.sub(r'[+-]?\d*\*?x\d+', '', equation.replace(" ", ""))
    if constant:
        constant_term = int(constant)
    else:
        constant_term = 0

    # append constant term to the coefficient list
    coefficients.append(constant_term)

    return coefficients


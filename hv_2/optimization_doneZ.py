import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from scipy.optimize import line_search


class LineSearchTool(object):
    def __init__(self, c1=1e-4, c2=0.9, alpha_0=1.0):
        self.c1 = c1
        self.c2 = c2
        self.alpha_0 = alpha_0

    def line_search(self, oracle, x_k, d_k):
        alpha = self.alpha_0
        wolfe_result = line_search(oracle.func, oracle.grad, x_k, d_k,
                                   f_prime=oracle.grad_directional(x_k, d_k),
                                   c1=self.c1, c2=self.c2)

        if wolfe_result[0] is None:
            return self._armijo_line_search(oracle, x_k, d_k, alpha)

        return wolfe_result[0]

    def _armijo_line_search(self, oracle, x_k, d_k, alpha):
        while oracle.func_directional(x_k, d_k, alpha) > oracle.func_directional(x_k, d_k,
                                                                                 0) + self.c1 * alpha * oracle.grad_directional(
                x_k, d_k):
            alpha *= 0.5
        return alpha


def get_line_search_tool(line_search_options=None):
    return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tolerance:
            return x_k, 'success', history

        d_k = -grad
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        x_k += alpha * d_k

        if trace:
            history['time'].append(iteration)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            history['x'].append(np.copy(x_k))

    return x_k, 'iterations_exceeded', history


import numpy as np
from collections import defaultdict
from scipy.linalg import cho_factor, cho_solve


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tolerance:
            return x_k, 'success', history

        # Получаем гессиан
        hess = oracle.hess(x_k)

        # Проверяем, является ли гессиан положительно определённым
        try:
            L = cho_factor(hess)
        except np.linalg.LinAlgError:
            return x_k, 'newton_direction_error', history

        # Решаем систему уравнений L * y = -grad для нахождения направления
        d_k = -cho_solve(L, grad)

        # Выполняем линейный поиск для определения шага
        alpha = line_search_tool.line_search(oracle, x_k, d_k)

        # Обновляем точку
        x_k += alpha * d_k

        if trace:
            history['time'].append(iteration)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            history['x'].append(np.copy(x_k))

        if display:
            print(f"Iteration {iteration}: x = {x_k}, f(x) = {oracle.func(x_k)}, ||grad|| = {grad_norm}")

    return x_k, 'iterations_exceeded', history
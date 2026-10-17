import numbers as nmbs
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import math

def is_empty(x):
    return np.asarray(x).size == 0

def flat_and_array(x):
    return np.array(x).flatten()


def loc(nums):
    nums = flat_and_array(nums)
    if is_empty(nums):
        return None
    return (nums).sum()/len(nums)


def variance(nums):
    nums = flat_and_array(nums)
    if is_empty(nums):
        return None
    l = loc(nums)
    return None if l is None else ((nums - l)**2).sum()/len(nums)

    
def scale(nums):
    nums = flat_and_array(nums)
    if is_empty(nums):
        return None
    return math.sqrt(variance(nums))


def chebyshev(nums, k):
    if k <= 0:
        raise ValueError('k should be >0')
    nums = flat_and_array(nums)
    if is_empty(nums):
        return None
    l = loc(nums)
    v = variance(nums)
    s = math.sqrt(v) if v is not None else 0.0
    return float(l-(k*s)),float(l+(k*s))


def norm(x_s):
    x_s = flat_and_array(x_s)
    if is_empty(x_s):
        return None
    l = loc(x_s)
    s = scale(x_s)        
    if s == 0 :
        return x_s 
    return (x_s - l)/s


def detect_matrix(M):
    M = np.asarray(M)
    if is_empty(M):
        return None,M.shape

    if M.ndim == 0 or  M.ndim == 1:
        M = M.reshape(1, -1)

    if M.ndim > 2:
        raise ValueError('The matrix must ultimately be 2-dimensional.')    
    
    typ = 'rectangular'
    s = set()
    n_columns = len(M[0])
    n_rows = len(M)
    dgnl = True
    zero = True
    o = M.shape

    for row in range(n_rows):
        for col in range(n_columns):
            if M[row][col] != 0:
                zero = False
    if zero:
            return 'null',o

    if n_columns == n_rows:
        
        for row in range(n_rows):
            if not dgnl :
                break
            for col in range(n_columns):

                if row != col:
                    if M[row][col] != 0:
                        dgnl = False
                        break
        
                else:
                    s.add(M[row][col])
        
        typ = 'diagonal' if dgnl else 'square'
        if dgnl and len(s) == 1:
            typ = 'scalar'

    elif n_columns == 1 and n_rows > 1:
        typ = 'column'

    elif n_rows == 1 and n_columns > 1:
        typ = 'row'

    if typ == 'scalar' and 1 in s:
        typ = 'identity'
    return typ,o


def mse(y_true_array, y_pred_array):
    y_pred_array, y_true_array= flat_and_array(y_pred_array), flat_and_array(y_true_array)
    if y_pred_array.size == y_true_array.size and y_pred_array.size != 0:
        return ((y_true_array - y_pred_array)**2).sum() / len(y_true_array)
    return None


def rmse(mse_value):
    if mse_value is None:
        return None
    return math.sqrt(mse_value)


def comparison_plot(f_true, f_pred):
    x = np.linspace(1, 10, 100)
    plt.plot(x, f_true(x), label='truth')
    plt.plot(x, f_pred(x), label='guess')
    plt.legend()
    plt.show()


def sigma(i, n, f=lambda x: x):
    is_bool = isinstance(i, bool) or isinstance(n, bool)
    not_int =  not isinstance(i, nmbs.Integral) or not isinstance(n, nmbs.Integral)
    if is_bool or not_int:
        raise TypeError('i and n must be integers (bool not allowed)')
    i,n = int(i), int(n)
    return sum(f(j) for j in range(i,n+1))


def matrix_equal(A,B):
    A, B = np.array(A), np.array(B)
    if A.shape == B.shape:
        return np.array_equal(A, B)
    return False


def derivative(f, *symbols):
    symbols_list = [x if isinstance(x, sp.Symbol) else sp.Symbol(str(x)) for x in symbols]
    return [sp.diff(f, x) for x in symbols_list]


if __name__ == "__main__":
    B = [
        [0, 1, -3],
        [4, 2, 7],
        [2, -9, 1]
    ]

    A = [
        [6, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ]

    print('is_empty\n', is_empty(A))
    input('\npress enter...\n')
    print('')
    print('flat_and_array: \n', flat_and_array(A))
    input('\npress enter...\n')
    print('')
    print('loc: \n', loc(A))
    input('\npress enter...\n')
    print('')
    print('variance: \n', variance(A))
    input('\npress enter...\n')
    print('')
    print('scale: \n', scale(A))
    input('\npress enter...\n')
    print('')
    print('chebyshev: \n', chebyshev(A,3))
    input('\npress enter...\n')
    print('')
    print('norm: \n', norm(A))
    input('\npress enter...\n')
    print('')
    print('detect_matrix: \n', detect_matrix(A))
    input('\npress enter...\n')
    print('')
    print('mse: \n',  mse(A,B))
    input('\npress enter...\n')
    print('')
    print('rmse: \n',  rmse( mse(A,B)))
    input('\npress enter...\n')
    print('')
    print('sigma: \n', sigma(i = 0, n = 10, f = lambda x: 2*x**2 + 3*x - 3))
    input('\npress enter...\n')
    print('')
    x = sp.Symbol('x')
    f = 2*x**2 + 3*x - 3
    print('derivative: \n', derivative(f, 'x', 'y'))
    input('\npress enter...\n')
    print('')
    comparison_plot(f_true=lambda x: 2*x**2 + 3*x - 3, f_pred=lambda x: 2*x**2 + 2*x - 3)
    input('\npress enter...\n')
    print('')
    print('equality: \n',  matrix_equal(A,B))
    input('\npress enter to exit...\n')
    print('bye...')

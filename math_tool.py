import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import math


def empty(x):
    return len(x) == 0


def flat(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()
    return np.array(x)


def loc(nums):
    nums = flat(nums)
    if empty(nums):
        return None
    return (nums).sum()/len(nums)


def variance(nums):
    nums = flat(nums)
    if empty(nums):
        return None
    l = loc(nums)
    return None if l is None else ((nums - l)**2).sum()/len(nums)

    
def scale(nums):
    nums = flat(nums)
    if empty(nums):
        return None
    return math.sqrt(variance(nums))


def chebyshev(k, nums):
    if k <= 0:
        raise ValueError('k should be >0')
    nums = flat(nums)
    if empty(nums):
        return None
    l = loc(nums)
    v = variance(nums)
    s = math.sqrt(v) if v is not None else 0.0
    return (l-(k*s)),(l+(k*s))


def norm(x_s):
    x_s = flat(x_s)
    if empty(x_s):
        return None
    l = loc(x_s)
    s = scale(x_s)        
    if s == 0 :
        return x_s 
    return (x_s - l)/s


def detect_matrix(M):
    typ = 'rectangular'
    s = set()
    n_columns = len(M[0])
    n_rows = len(M)
    dgnl = True
    zero = True

    for row in range(n_rows):
        for col in range(n_columns):
            if M[row][col] != 0:
                zero = False
    if zero:
            return 'null'

    if n_columns == n_rows:
        
        for row in range(n_rows):
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
    return typ


def mse(y_true_array, y_pred_array):
    y_pred_array, y_true_array= flat(y_pred_array), flat(y_true_array)
    if y_pred_array.size == y_true_array.size and y_pred_array.size != 0:
        return ((y_true_array - y_pred_array)**2).sum() / len(y_true_array)
    return None


def rmse(mse_value):
    if mse_value is None:
        return None
    return math.sqrt(mse_value)


def mse_plot(f_true, f_pred):
    x = np.linspace(1, 10, 100)
    plt.plot(x, f_true(x), label='truth')
    plt.plot(x, f_pred(x), label='guess')
    plt.legend()
    plt.show()


def sigma(i, n, f):
    if isinstance(i, int) and isinstance(n, int):
        return 0 if i>n else sum(f(j) for j in range(i,n+1))
    raise ValueError('i & n should be an integer number')


def matrix_equality(A,B):
    l = (A == B)
    for i in l :
        for j in i :
            if not j:
                return False
    return True


def partial_derivative(f, *sympols_list):
    sympols_list = [sp.Symbol(s) for s in sympols_list]
    return [sp.diff(f, x) for x in sympols_list]


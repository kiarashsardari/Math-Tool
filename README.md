# ╭──────────────────────────────────────────────╮
# │                                              │
# │              M A T H   T O O L               │
# │      ─────────────────────────────────       │
# │      Statistical & Mathematical Toolkit      │
# │                for Python 3                  │
# │                                              │
# ╰──────────────────────────────────────────────╯

> **A lightweight, dependency-aware Python module** for statistical analysis, array
> processing, matrix classification, error metrics, symbolic differentiation, and
> comparative plotting — built on top of `NumPy`, `SymPy`, and `Matplotlib`.

---

## ◈ Table of Contents

<table>
<tr>
<td width="50%" valign="top">

**Getting Started**

- [Overview](#-overview)
- [Feature Set](#-feature-set)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Demo](#-quick-demo)

</td>
<td width="50%" valign="top">

**API Reference**

- [Utility Helpers](#-utility-helpers)
- [Statistics](#-statistics)
- [Normalization](#-normalization)
- [Matrix Analysis](#-matrix-analysis)
- [Error Metrics](#-error-metrics)
- [Symbolic Math](#-symbolic-math)
- [Visualization](#-visualization)

</td>
</tr>
</table>

---

## ◈ Overview

**`math_tool.py`** is a self-contained Python module designed to provide clean,
readable shortcuts for everyday mathematical and statistical tasks. It is intended
for users who work with `NumPy` arrays, `SymPy` expressions, or `Matplotlib` plots
and want a consistent, reusable interface.

The module is organized around the following pillars:

| Pillar | Description |
|:---|:---|
| **Statistics** | Mean, variance, and standard deviation (population-based) |
| **Normalization** | Robust z-score normalization with zero-variance handling |
| **Matrix Analysis** | Detection of 8 structural matrix types |
| **Error Metrics** | MSE and RMSE for regression and prediction evaluation |
| **Symbolic Math** | Multi-variable differentiation via `SymPy` |
| **Visualization** | Side-by-side function comparison plots |

---

## ◈ Feature Set

<table>
<tr><td>

**Stable**
- ✔ Handles list, array, and matrix inputs uniformly
- ✔ Automatically flattens inputs for statistical functions
- ✔ Returns `None` gracefully for empty inputs
- ✔ Population-based variance and standard deviation

</td><td>

**Robust**
- ✔ Correct handling of zero standard deviation
- ✔ Safe integer validation in `sigma` (rejects `bool`)
- ✔ Shape-aware matrix equality checking
- ✔ Comprehensive matrix type classification

</td></tr>
</table>

---

## ◈ Requirements

```
Python  ≥ 3.7
NumPy   ≥ 1.20
SymPy   ≥ 1.9
Matplotlib ≥ 3.4
```

Install all dependencies at once:

```bash
pip install numpy sympy matplotlib
```

---

## ◈ Installation

Place `math_tool.py` in your project directory. Then import it as a module:

```python
import math_tool as mt
```

Or import only what you need:

```python
from math_tool import loc, variance, scale, norm
```

---

## ◈ Quick Demo

Run the built-in demo:

```bash
python math_tool.py
```

The script walks through each function interactively, pausing with `Enter` between
steps, and finishes with a comparison plot.

**Quick start example:**

```python
import math_tool as mt

A = [
    [6, 0, 0],
    [0, 2, 0],
    [0, 0, 3]
]

print(mt.loc(A))            # 3.666...
print(mt.variance(A))       # population variance
print(mt.scale(A))          # population std-dev
print(mt.chebyshev(A, 3))   # Chebyshev interval
print(mt.norm(A))           # z-score normalization
print(mt.detect_matrix(A))  # ('diagonal', (3, 3))
```

---

# ╭──────────────────────────────────────────────╮
# │              A P I   R E F E R E N C E       │
# ╰──────────────────────────────────────────────╯

---

## ◈ Utility Helpers

### ▸ `is_empty(x)`

Checks whether the input contains zero elements.

```python
is_empty(x: Any) -> bool
```

| Parameter | Type | Description |
|:---|:---|:---|
| `x` | any | Value convertible to an array |

```python
mt.is_empty([])        # True
mt.is_empty([1, 2])    # False
```

---

### ▸ `flat_and_array(x)`

Converts the input to a 1-D NumPy array.

```python
flat_and_array(x: Any) -> np.ndarray
```

```python
mt.flat_and_array([[1, 2], [3, 4]])
# array([1, 2, 3, 4])
```

---

## ◈ Statistics

### ▸ `loc(nums)`

Returns the arithmetic mean of the elements.

```python
loc(nums: array_like) -> float | None
```

> **Note:** Returns `None` for an empty input.

```python
mt.loc([1, 2, 3, 4])   # 2.5
mt.loc([])             # None
```

---

### ▸ `variance(nums)`

Returns the **population variance** (divides by `n`).

```python
variance(nums: array_like) -> float | None
```

```python
mt.variance([1, 2, 3, 4])   # 1.25
mt.variance([])             # None
```

---

### ▸ `scale(nums)`

Returns the **population standard deviation** (square root of variance).

```python
scale(nums: array_like) -> float | None
```

```python
mt.scale([1, 2, 3, 4])
# 1.118033988749895
```

---

### ▸ `chebyshev(nums, k)`

Computes the Chebyshev interval `(μ − k·σ, μ + k·σ)`.

```python
chebyshev(nums: array_like, k: float) -> tuple[float, float] | None
```

> **Raises:** `ValueError` if `k <= 0`.

```python
mt.chebyshev([1, 2, 3, 4], 2)
# (0.2639320225002102, 4.73606797749979)
```

---

## ◈ Normalization

### ▸ `norm(x_s)`

Performs z-score normalization:

```
z = (x − μ) / σ
```

```python
norm(x_s: array_like) -> np.ndarray | None
```

> **Edge case:** If the standard deviation is zero, the flattened input is returned
> unchanged.

```python
mt.norm([1, 2, 3])
# array([-1.22474487,  0.        ,  1.22474487])
```

---

## ◈ Matrix Analysis

### ▸ `detect_matrix(M)`

Classifies a matrix by its structural type and returns its shape.

```python
detect_matrix(M: array_like) -> tuple[str | None, tuple]
```

| Type | Description |
|:---|:---|
| `null` | All elements are zero |
| `identity` | Identity matrix |
| `scalar` | Diagonal matrix with equal non-zero diagonal entries |
| `diagonal` | Diagonal matrix with unequal entries |
| `square` | Square matrix (non-diagonal) |
| `column` | Single-column matrix with more than one row |
| `row` | Single-row matrix with more than one column |
| `rectangular` | General rectangular matrix |

> **Raises:** `ValueError` for arrays with more than two dimensions.

```python
mt.detect_matrix([[1, 0], [0, 1]])   # ('identity', (2, 2))
mt.detect_matrix([[2, 0], [0, 2]])   # ('scalar',   (2, 2))
mt.detect_matrix([[1, 2], [3, 4]])   # ('square',   (2, 2))
mt.detect_matrix([[0, 0], [0, 0]])   # ('null',     (2, 2))
```

---

### ▸ `matrix_equal(A, B)`

Checks element-wise equality between two matrices.

```python
matrix_equal(A: array_like, B: array_like) -> bool
```

```python
mt.matrix_equal([[1, 2]], [[1, 2]])   # True
mt.matrix_equal([[1, 2]], [[1, 3]])   # False
```

---

## ◈ Error Metrics

### ▸ `mse(y_true_array, y_pred_array)`

Computes Mean Squared Error.

```python
mse(y_true_array, y_pred_array) -> float | None
```

> **Condition:** Both inputs must have the same, non-zero number of elements;
> otherwise, returns `None`.

```python
mt.mse([1, 2, 3], [1, 2, 4])
# 0.3333333333333333
```

---

### ▸ `rmse(mse_value)`

Returns the square root of the given MSE value.

```python
rmse(mse_value: float | None) -> float | None
```

```python
mt.rmse(0.3333333333333333)
# 0.5773502691896257
```

---

## ◈ Symbolic Math

### ▸ `sigma(i, n, f)`

Computes an inclusive summation:

```
Σ  f(j)    for j = i .. n
```

```python
sigma(i: int, n: int, f: Callable = lambda x: x) -> number
```

> **Raises:** `TypeError` if `i` or `n` is not an integer or is of type `bool`.

```python
mt.sigma(0, 10, lambda x: 2*x**2 + 3*x - 3)
# 902
```

---

### ▸ `derivative(f, *symbols)`

Returns symbolic derivatives of a `SymPy` expression.

```python
derivative(f: sp.Expr, *symbols) -> list[sp.Expr]
```

```python
import sympy as sp

x = sp.Symbol('x')
f = 2*x**2 + 3*x - 3

mt.derivative(f, 'x', 'y')
# [4*x + 3, 0]
```

---

## ◈ Visualization

### ▸ `comparison_plot(f_true, f_pred)`

Plots two functions on the interval `[1, 10]` with 100 sample points.

```python
comparison_plot(f_true: Callable, f_pred: Callable) -> None
```

> **Labels:** `truth` for the true function, `guess` for the predicted function.

```python
mt.comparison_plot(
    f_true=lambda x: 2*x**2 + 3*x - 3,
    f_pred=lambda x: 2*x**2 + 2*x - 3
)
```

---

# ╭──────────────────────────────────────────────╮
# │              E X A M P L E S                 │
# ╰──────────────────────────────────────────────╯

### ▸ Statistics on a Matrix

```python
import math_tool as mt

A = [[6, 0, 0], [0, 2, 0], [0, 0, 3]]

print("mean:      ", mt.loc(A))
print("variance:  ", mt.variance(A))
print("std:       ", mt.scale(A))
print("chebyshev: ", mt.chebyshev(A, 3))
print("normalized:", mt.norm(A))
```

### ▸ Matrix Type Detection

```python
matrices = {
    "identity": [[1, 0], [0, 1]],
    "scalar":   [[5, 0], [0, 5]],
    "diagonal": [[1, 0], [0, 2]],
    "square":   [[1, 2], [3, 4]],
    "column":   [[1], [2]],
    "row":      [[1, 2]],
    "null":     [[0, 0], [0, 0]],
}

for name, M in matrices.items():
    print(f"{name:>9} -> {mt.detect_matrix(M)}")
```

### ▸ Error Evaluation

```python
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse_value = mt.mse(y_true, y_pred)
print("MSE: ", mse_value)
print("RMSE:", mt.rmse(mse_value))
```

### ▸ Summation and Differentiation

```python
import sympy as sp
import math_tool as mt

total = mt.sigma(1, 5, lambda x: x**2)
print("sum:", total)

x = sp.Symbol('x')
expr = x**3 + 2*x**2 + x
print("derivative:", mt.derivative(expr, 'x'))
```

### ▸ Comparative Plot

```python
import math_tool as mt

mt.comparison_plot(
    f_true=lambda x: 2*x**2 + 3*x - 3,
    f_pred=lambda x: 2*x**2 + 2*x - 3
)
```

---

# ╭──────────────────────────────────────────────╮
# │         I M P L E M E N T A T I O N          │
# ╰──────────────────────────────────────────────╯

> **Design Notes** — Important behaviors to keep in mind when using the module.

- Statistical functions flatten their inputs via `flat_and_array`, so matrices are
  processed as a single sequence of elements.
- `variance` and `scale` use the **population** formula (division by `n`, not `n-1`).
- In `norm`, a zero standard deviation returns the flattened input unchanged.
- `detect_matrix` checks the all-zero case first, so the zero matrix is always
  reported as `null`.
- `detect_matrix` treats 1-D arrays as single-row matrices.
- `mse` returns a numeric value only when both inputs have equal, non-zero size.
- `sigma` iterates inclusively from `i` to `n`.
- `derivative` always returns a list, even when a single symbol is provided.
- `comparison_plot` uses a fixed interval `[1, 10]` with 100 sample points.
- `derivative` expects a `SymPy` expression as input.

---

<div align="center">

**math_tool.py** — crafted for clarity, built for reuse.

</div>

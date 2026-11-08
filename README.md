<div align="center">

# 🧮 Math-Tool

### کتابخانه جامع ابزارهای ریاضی، آماری و تحلیل داده در پایتون

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Required-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SymPy](https://img.shields.io/badge/SymPy-Required-3B5526?style=for-the-badge&logo=sympy&logoColor=white)](https://www.sympy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Required-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Not%20Specified-red?style=for-the-badge)]()

</div>

---

<div dir="rtl">

**Math-Tool** یک کتابخانه‌ی جامع و ماژولار برای پایتون است که مجموعه‌ای از ابزارهای پرکاربرد ریاضی، آماری، ماتریسی، رسم نمودار و رگرسیون خطی را در قالبی ساده، تمیز و یکپارچه فراهم می‌کند.

این پروژه با هدف فراهم‌سازی یک جعبه‌ابزار سبک و در عین حال کامل برای دانشجویان، مهندسان و علاقه‌مندان به علم داده طراحی شده است.

---

## 📑 فهرست محتویات

- [✨ ویژگی‌های کلیدی](#features)
- [📋 پیش‌نیازها](#requirements)
- [📦 نصب و راه‌اندازی](#installation)
- [🚀 شروع سریع](#quickstart)
- [📚 مستندات API](#api)
  - [&lrm;⚙️ Settings](#settings)
  - [&lrm;∂ derivative](#derivative)
  - [&lrm;∑ sigma](#sigma)
  - [&lrm;📊 Stats](#stats)
  - [&lrm;🔢 Matrix](#matrix)
  - [&lrm;📐 Metrics](#metrics)
  - [&lrm;📈 Plots](#plots)
  - [&lrm;📉 LinearRegression](#linearregression)
  - [&lrm;🗂️ DataHandler](#datahandler)
- [🧪 اجرای تست‌ها](#tests)
- [📁 ساختار پروژه](#structure)
- [🤝 مشارکت](#contributing)
- [📄 لایسنس](#license)
- [📬 ارتباط با من](#contact)

---

<a id="features"></a>
## ✨ ویژگی‌های کلیدی

- 🧩 **تخت‌سازی و نرمال‌سازی ورودی**: پذیرش هر ساختار تو در تو (لیست، تاپل، جنریتور، آرایه‌ی نامپای، Series پانداز و…) و تبدیل خودکار به آرایه‌ی یک‌بعدی
- ∂ **مشتق‌گیری نمادین**: محاسبه‌ی مشتق معمولی و جزئی با پشتیبانی از SymPy
- ∑ **سیگمای جامع**: پشتیبانی از بازه، لیست، ترکیب دو لیست، دو تابع و عملگر دلخواه (‎+‎، ‎−‎، ‎×‎، ‎÷‎، توان و باقیمانده)
- 📊 **ابزارهای آماری**: میانگین، واریانس، انحراف معیار، نرمال‌سازی و نامساوی چبیشف
- 🔢 **تشخیص نوع ماتریس**: شناسایی خودکار ماتریس‌های همانی، قطری، اسکالر، مربعی، سطری، ستونی و تهی
- 📐 **معیارهای خطا**: محاسبه‌ی MSE و RMSE با پشتیبانی از ورودی‌های نامنظم
- 📉 **رگرسیون خطی ساده**: ساخت مدل روی دیتافریم، پیش‌بینی و محاسبه‌ی خطای مدل
- 🗂️ **مدیریت داده**: ساخت دیتاست نمونه و خواندن/ساخت خودکار فایل‌های CSV
- 📈 **رسم نمودار**: نمودار خطی، پراکندگی و مقایسه‌ی دو تابع
- 🧪 **تست کامل**: پوشش تست واحد برای تمامی کلاس‌ها و توابع

---

<a id="requirements"></a>
## 📋 پیش‌نیازها

قبل از استفاده، مطمئن شوید که نسخه‌ی **Python 3.8 یا بالاتر** روی سیستم شما نصب است.

```bash
python --version
```

---

<a id="installation"></a>
## 📦 نصب و راه‌اندازی

### ۱. کلون کردن مخزن

```bash
git clone https://github.com/kiarashsardari/Math-Tool.git
cd Math-Tool
```

### ۲. نصب وابستگی‌ها

```bash
pip install numpy pandas matplotlib sympy
```

### ۳. استفاده در پروژه

فایل `math_tool.py` را در کنار پروژه‌ی خود قرار دهید یا از طریق زیر ایمپورت کنید:

```python
from math_tool import (
    Settings, Stats, Matrix, Metrics,
    LinearRegression, DataHandler, Plots,
    derivative, sigma,
)
```

---

<a id="quickstart"></a>
## 🚀 شروع سریع

یک نمونه‌ی کامل از کاربرد کتابخانه:

```python
import numpy as np
import pandas as pd
import sympy as sp

from math_tool import (
    Settings, Stats, Matrix, Metrics,
    LinearRegression, DataHandler, Plots,
    derivative, sigma,
)

# --- ۱. تخت‌سازی ورودی ---
data = [1, 2, [3, (4, 5)], np.array([[6, 7], [8, 9]])]
flat = Settings(data).x
print(flat)  # [1 2 3 4 5 6 7 8 9]

# --- ۲. مشتق‌گیری نمادین ---
x = sp.Symbol('x')
f = 2 * x**3 + 3 * x - 6
print(derivative(f, x))  # 6*x**2 + 3

# --- ۳. سیگمای بازه‌ای ---
total = sigma(0, 10, lambda i: 2 * i)
print(total)  # 110

# --- ۴. محاسبات آماری ---
s = Stats([1, 2, 3, 4, 5])
print(s.loc())       # 3.0
print(s.variance())  # 2.0
print(s.scale())     # 1.4142...
print(s.norm())      # [-1.414 -0.707 0. 0.707 1.414]
print(s.chebyshev(2)) # (-1.414, 7.414)

# --- ۵. تشخیص نوع ماتریس ---
print(Matrix([[1, 0], [0, 1]]).detect_matrix())  # ('identity', (2, 2))

# --- ۶. معیار خطا ---
print(Metrics.mse([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))  # 0.375

# --- ۷. رگرسیون خطی ---
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [7, 4, 1, -2, -5],
})
model = LinearRegression(df, 'x', 'y')
print(model.t0, model.t1)  # 10.0 -3.0
print(model.predict(10))   # -20.0

# --- ۸. مدیریت داده ---
df_sample = DataHandler.make_sample_dataset(50)
print(df_sample.head())

# --- ۹. رسم نمودار ---
Plots.show_plot([1, 2, 3, 4], [1, 4, 9, 16])
```

---

<a id="api"></a>
## 📚 مستندات API

---

<a id="settings"></a>
### &lrm;⚙️ Settings

مدیریت و تخت‌سازی ورودی‌های پیچیده. دیکشنری و مجموعه پذیرفته نمی‌شوند.

#### `Settings(x)`

```python
Settings([1, 2, [3, (4, 5)], np.array([[6, 7]])]).x
# خروجی: np.array([1, 2, 3, 4, 5, 6, 7], dtype=object)
```

#### `is_empty()`

بررسی خالی بودن ورودی (همه‌ی اعضا `None` یا آرایه‌ی خالی).

```python
Settings([None]).is_empty()        # True
Settings([None, 1]).is_empty()     # False
```

#### `flat_and_array()`

تخت‌سازی کامل و بازگرداندن آرایه‌ی نامپای یک‌بعدی.

#### `is_iter(x)` — استاتیک

بررسی iterable بودن، با استثنای `str`، `dict` و `set`.

```python
Settings.is_iter([1, 2])   # True
Settings.is_iter(12)       # False
Settings.is_iter("hello")  # False
```

---

<a id="derivative"></a>
### &lrm;∂ derivative

محاسبه‌ی مشتق معمولی یا مشتق جزئی با استفاده از SymPy.

#### `derivative(f, *symbols)`

```python
import sympy as sp
from math_tool import derivative

x = sp.Symbol('x')
derivative(2*x**3 + 3*x - 6, x)  # 6*x**2 + 3

# مشتق جزئی
a, b = sp.symbols('a b')
f = 2*a**3 + 3*b**2
derivative(f, a, b)  # [6*a**2, 6*b]
```

> 💡 اگر فقط یک نماد بدهید، نتیجه یک عبارت تکی خواهد بود؛ در غیر این صورت لیستی از مشتق‌ها برمی‌گردد.

---

<a id="sigma"></a>
### &lrm;∑ sigma

سیگمای جامع با پشتیبانی از بازه، لیست، دو لیست، دو تابع و عملگر دلخواه.

#### `sigma(i=None, n=None, f=identity, g=identity, operator=None)`

##### حالت ۱: بازه با یک تابع

```python
sigma(0, 10, lambda i: 2*i)   # 110
```

##### حالت ۲: لیست با یک تابع

```python
sigma([1, 2, 3, 4], f=lambda x: 2*x)   # 20
sigma(n=[1, 2, 3, 4], g=lambda x: 2*x) # 20
```

##### حالت ۳: دو لیست با عملگر و یک تابع

```python
i = [1, 2, 3, 4, 5]
n = [6, 7, 8, 9, 10]
sigma(i, n, f=lambda x: 2*x, operator='-')   # -50
```

##### حالت ۴: دو لیست با دو تابع و عملگر

```python
sigma(
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    f=lambda x: 2*x,
    g=lambda x: x - 1,
    operator='-',
)  # -5
```

**عملگرهای پشتیبانی‌شده:** `+`، `-`، `*`، `/`، `**`، `%`

---

<a id="stats"></a>
### &lrm;📊 Stats

ابزارهای آماری پایه روی هر ورودی قابل‌تخت‌سازی.

#### `Stats(nums)`

```python
from math_tool import Stats

s = Stats([[1, 2], (3, 4), [5]])
```

#### متدها

| متد | فرمول | توضیح |
|---|---|---|
| `loc()` | μ = Σ(xᵢ)/n | میانگین |
| `variance()` | σ² = Σ(xᵢ−μ)²/n | واریانس (جامعه) |
| `scale()` | σ = √variance | انحراف معیار |
| `norm()` | (xᵢ−μ)/σ | نرمال‌سازی |
| `chebyshev(k)` | μ ± k·σ | بازه‌ی نامساوی چبیشف |

```python
s = Stats([1, 2, 3, 4, 5])
s.loc()          # 3.0
s.variance()     # 2.0
s.scale()        # 1.4142...
s.norm()         # [-1.414, -0.707, 0, 0.707, 1.414]
s.chebyshev(2)   # (-1.414, 7.414)
```

> ⚠️ مقادیر `None` یا آرایه‌ی خالی خطا می‌دهند. مقادیر غیرعددی نیز خطای `ValueError` یا `TypeError` می‌دهند.

---

<a id="matrix"></a>
### &lrm;🔢 Matrix

تشخیص نوع ماتریس و بررسی تساوی.

#### `Matrix(M)`

#### `detect_matrix()`

خروجی: تاپل (نوع، shape)

| نوع | توضیح |
|---|---|
| `null` | همه‌ی اعضا صفر |
| `rectangular` | مستطیلی |
| `row` | سطری (۱×n) |
| `column` | ستونی (n×۱) |
| `square` | مربعی |
| `diagonal` | قطری |
| `scalar` | مضربی از همانی |
| `identity` | همانی |

```python
Matrix([[1, 0], [0, 1]]).detect_matrix()   # ('identity', (2, 2))
Matrix([[3, 0], [0, 3]]).detect_matrix()   # ('scalar', (2, 2))
Matrix([[0, 0], [0, 0]]).detect_matrix()   # ('null', (2, 2))
```

#### `matrix_equal(B)`

بررسی تساوی دو ماتریس (به‌همراه shape).

```python
Matrix([[1, 2], [3, 4]]).matrix_equal([[1, 2], [3, 4]])  # True
```

---

<a id="metrics"></a>
### &lrm;📐 Metrics

معیارهای خطا.

#### `mse(y_true, y_pred)` — استاتیک

```python
from math_tool import Metrics
Metrics.mse([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])  # 0.375
```

#### `rmse(mse_value)` — استاتیک

```python
Metrics.rmse(0.375)  # 0.6123...
```

> 💡 اگر اندازه‌ی دو ورودی برابر نباشد، خروجی `None` است.

---

<a id="plots"></a>
### &lrm;📈 Plots

رسم نمودارهای آماده با Matplotlib.

#### `show_plot(x, y)`

```python
Plots.show_plot([1, 2, 3, 4], [1, 4, 9, 16])
```

#### `show_scatter(x, y)`

```python
Plots.show_scatter([1, 2, 3], [4, 5, 6])
```

#### `comparison_plot(f_true, f_pred)`

مقایسه‌ی دو تابع روی بازه‌ی [۱، ۱۰].

```python
Plots.comparison_plot(
    f_true=lambda x: 2*x,
    f_pred=lambda x: 3*x + 3,
)
```

---

<a id="linearregression"></a>
### &lrm;📉 LinearRegression

رگرسیون خطی ساده با فرمول بسته روی دیتافریم.

#### `LinearRegression(df, x_name, y_name)`

مدل به‌فرم y = θ₀ + θ₁·x

```
θ₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²
θ₀ = ȳ − θ₁·x̄
```

```python
import pandas as pd
from math_tool import LinearRegression

df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [7, 4, 1, -2, -5],
})

model = LinearRegression(df, 'x', 'y')
print(model.t0, model.t1)      # 10.0 -3.0
print(model.predict(10))       # -20.0
print(model.model_mse(df['x'], df['y']))  # 0.0
```

---

<a id="datahandler"></a>
### &lrm;🗂️ DataHandler

ساخت دیتاست نمونه و مدیریت فایل‌های CSV.

#### `make_sample_dataset(n, seed=313)` — استاتیک

ساخت دیتاست نمونه از سن و قد با نویز مصنوعی و ذخیره در `sample_dataset.csv`.

```python
df = DataHandler.make_sample_dataset(100)
print(df.head())
```

#### `read_or_make_csv(address, make_dataset_function, *args)` — استاتیک

اگر فایل CSV موجود باشد می‌خواند، در غیر این صورت با تابع داده‌شده آن را می‌سازد.

```python
df = DataHandler.read_or_make_csv(
    "data.csv",
    DataHandler.make_sample_dataset,
    100,
)
```

---

<a id="tests"></a>
## 🧪 اجرای تست‌ها

کتابخانه دارای مجموعه‌ی کامل تست واحد با `unittest` است.

```bash
python -m unittest test_math_tool.py -v
```

خروجی مورد انتظار:

```
Ran 62 tests in 0.xxx s

OK
```

---

<a id="structure"></a>
## 📁 ساختار پروژه

```
Math-Tool/
├── math_tool.py         # هسته‌ی کتابخانه
├── test_math_tool.py    # مجموعه تست‌های واحد
└── README.md            # مستندات پروژه
```

---

<a id="contributing"></a>
## 🤝 مشارکت

از هرگونه مشارکت، پیشنهاد و گزارش باگ استقبال می‌شود! 🙌

1. مخزن را **Fork** کنید
2. یک شاخه‌ی جدید بسازید:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. تغییرات خود را **Commit** کنید:
   ```bash
   git commit -m "Add some amazing feature"
   ```
4. شاخه را **Push** کنید:
   ```bash
   git push origin feature/amazing-feature
   ```
5. یک **Pull Request** باز کنید

### راهنمای سبک کد

- رعایت **PEP 8**
- افزودن **تست** برای هر قابلیت جدید
- استفاده از **docstring** برای توابع عمومی
- حفظ سازگاری با ورژن‌های قبلی

---

<a id="license"></a>
## 📄 لایسنس

لایسنس این پروژه در حال حاضر مشخص نشده است. اگر قصد استفاده‌ی تجاری یا بازتوزیع دارید، لطفاً با نویسنده تماس بگیرید.

---

<a id="contact"></a>
## 📬 ارتباط با من

**Kiarash Sardari**

- 🐙 GitHub: [@kiarashsardari](https://github.com/kiarashsardari)
- 📦 مخزن پروژه: [Math-Tool](https://github.com/kiarashsardari/Math-Tool)
- 📧 ایمیل: [kiarash123sardari@gmail.com](mailto:kiarash123sardari@gmail.com)

</div>

---

<div align="center" dir="rtl">

**⭐ اگر این پروژه برایتان مفید بود، فراموش نکنید یک ستاره بدهید! ⭐**

ساخته شده با ❤️ و Python

</div>

import math
import numbers as nmbs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


class Settings:
    def __init__(self, x):
        if isinstance(x, (dict, set)):  # دیکشنری و مجموعه قبول نکن
            raise ValueError("this class doesn't apply dicts and sets.")
        # بای دیفالت روی ورودی متد flat_and_array رو اعمال کن
        self.x = x
        self.x = self.flat_and_array()

    # بررسی خالی بودن به جز دیکشنری و مجموعه
    def is_empty(self):
        x = self.x
        s = set(x)
        return True if len(s) == 1 and None in s else x.size == 0

    # تبدیل به آرایه نامپای و تخت کردن ماتریس های چند بعدی
    def flat_and_array(self):
        x = self.x
        if isinstance(x, np.ndarray):
            # تبدیل وروردی آرایه به لیست با در نظر گرفتن آرایه های 0 بعدی
            x = [x.item()] if x.ndim == 0 else x.tolist()
        elif self.is_iter(x):
            # تبدیل ورودی هایی ایتریتور دیگر مثل تاپل و جنریتور و غیره به لیست
            x = list(x)
        else:
            # تبدیل وروئی های تکی مثل اینتجر یا فلات و ... به لیست تک عضوی
            x = [x]
        is_list_inside = True
        # تازمانی که یک ایتریتور (به جز دیکشنری و ست و رشته) در ورودی
        # وجود داشته باشد این حلقه ادامه دارد
        while is_list_inside:
            l = []
            is_list_inside = False
            for i in x:
                if isinstance(i, (dict, set)):
                    raise ValueError("can't be used for dict and set")
                elif not self.is_iter(i) and not isinstance(
                    i, (np.ndarray, dict, set)
                ):
                    l.append(i)  # افزودن آیتم های تکی به لیست موقت
                else:
                    if isinstance(i, np.ndarray):
                        # تبدیل آرایه های داخلی به لیست با در نظر گرفتن
                        # آرایه های 0 بعدی
                        i = [i.item()] if i.ndim == 0 else i.tolist()
                    i = list(i)  # برای تبدیل سایر ایتریتور ها به لیست
                    # افزودن تک تک اعضای درون لیست ساخته شده به لیست موقت
                    l.extend(i)
            x = l
            # بررسی وجود اعضای غیر اتمی در لیست
            for o in x:
                if isinstance(o, (dict, set)):
                    raise ValueError("can't be used for dict and set")
                elif self.is_iter(o):
                    is_list_inside = True

        # خروجی >> آرایه تخت شده ی آیتم ورودی
        return np.array(x, dtype=object).flatten()

    @staticmethod
    # بررسی ایتریتور بودن به جز رشته ها و دیکشنری و ست
    def is_iter(x):
        if isinstance(x, (str, dict, set)):
            # رشته ها و دیکشنری را ایتریتور حساب نمی کند
            return False
        try:
            iter(x)
        except TypeError:
            return False
        return True


# مشتق و مشتق جزئی
# ∂f/∂x
def derivative(f, *symbols):
    symbols_list = [
        x if isinstance(x, sp.Symbol) else sp.Symbol(str(x))
        for x in symbols
    ]
    results = [sp.diff(f, x) for x in symbols_list]
    return results[0] if len(results) == 1 else results


_none = lambda x: x


# سیگمای جامع
# ∑ᵢ₌₁ⁿ xᵢ
def sigma(i=None, n=None, f=_none, g=_none, operator=None):
    if i is None and n is None and operator is None:
        raise TypeError("sigma missing required arguments")
    # اگر آی و ان ایتریتور بودند، تخت و تبدیل به آرایه شوند
    if Settings.is_iter(i):
        i = Settings(i).x
    if Settings.is_iter(n):
        n = Settings(n).x
    ops = {
        '+': lambda a, b: a + b,
        '%': lambda a, b: a % b,
        '*': lambda a, b: a * b,
        '**': lambda a, b: a ** b,
        '/': lambda a, b: a / b,
        '-': lambda a, b: a - b,
    }
    if operator is not None:  # اگر کاربر اپراتور وارد کرده بود
        operator = operator.strip()
        if operator in ops:  # و اپراتور معتبر بود

            # <<-------------
            '''
            وقتی کاربر اپراتور را وارد کرده
            طبق شرط های زیر اگر کاربر فقط یک تابع وارد کرد
            توابع اف و جی هردو یکسان و برابر با همان تابع داده شده می شوند
            همان تابع بر دو ایکس جدا یعنی لیست های آی و ان اعمال شود
            و بر دو لیست به دست امده عمل اپراتور را انجام و سیگما را انجام دهد
            '''
            if f is _none:
                f = g
            elif g is _none:
                g = f
            # ------------->>

            # <<-------------
            '''
            وقتی کاربر اپراتور را وارد کرده
            طبق شرط های زیر اگر کاربر فقط یک لیست وارد کرد
            مقادیر آی و ان هردو یکسان و برابر با همان لیست داده شده می شوند
            و در آخر هم به تابع جی و هم به تابع اف، آن لیست به طور جداگانه
            داده شود و بعد از اجرای تابع ها با ایکس یکسان، روی وای های بدست
            آمده عمل اپراتور انجام شود و در نهایت روی لیست نهایی عمل سیگما
            انجام شود
            '''
            if n is None:
                n = i
            elif i is None:
                i = n

            # اگر هم ان و هم آی، نان بودند
            if i is None:
                raise TypeError(
                    "at least one of i or n must be given when operator "
                    "is set"
                )
            # ------------->>

            # نامعتبر بودن ورودی
            # نامعتبر بودن آی و ان
            if (not Settings.is_iter(i) or not Settings.is_iter(n)
                    or len(i) != len(n)):
                raise TypeError(
                    "at least one of i or n must be given and been valid "
                    "when operator is set (length of should be equal)"
                )

            # ----------- سیگما با دو تابع و دو لیست و اپراتور -----------
            # حالتی که درآن دوتابع به ترتیب روی دولیست اعمال میشوند و
            # برای خروجی های انها عمل اپراتور انجام شده و سیگما انجام میشود
            # یا اینکه کاربر فقط یکی از آی یا ان را به همراه اپراتور و
            # دو تابع وارد کرده برای
            l1 = np.array([f(v) for v in i])  # اعمال تابع اف روی آی
            l2 = np.array([g(j) for j in n])  # اعمال تابع جی روی ان
            # اعمال اپراتور روی لیست های خروجی و اعمال سیگما
            return (ops[operator](l1, l2)).sum()
            # -----------------------------------------------------------

        else:  # اگر اپراتور معتبر نبود
            raise ValueError("Invalid operation symbol")

    # <<---
    # نامعتبر بودن ورودی
    # اپراتور داده نشده ولی آی و ان هردو ایتریتور هستند
    if Settings.is_iter(i) and Settings.is_iter(n):
        raise ValueError(
            "you have to give an operator when both of i and n are iterator"
        )
    # اپراتور نداده ولی دو تابع داده شده
    if g is not _none and f is not _none:
        raise ValueError(
            "2 functions have been given, but i and n are not iterator "
            "and you didn't give any operator"
        )
    # اگر کاربر اف را وارد نکرد، اف را مساوی جی قرار بده
    if g is not _none and f is _none:
        f = g
    # وقتی هم آی و هم ان داده شده اند ولی یکی از آنها ایتریتور است
    if (Settings.is_iter(i) and n is not None
            or Settings.is_iter(n) and i is not None):
        raise TypeError(
            "you can not give both of i & n when one of them is iterator"
        )
    # --->>

    # ----------- سیگما با یک تابع و یک لیست -----------
    # حالت هایی که کاربر یک لیست داده که روی اعضای آن تابع اف اعمال و
    # سیگما انجام شود
    if Settings.is_iter(i):
        return sum(f(j) for j in i)
    elif Settings.is_iter(n):
        return sum(f(j) for j in n)
    # -------------------------------------------------

    # ----------- سیگما با یک تابع و بازه ی آی تا ان -----------
    # حالتی که کابر بازه آی تا ان رو تعیین کرده که یکی یکی روی اعداد
    # این بازه تابع اف اعمال و سیگما انجام شود
    is_bool = isinstance(i, bool) or isinstance(n, bool)
    not_int = (not isinstance(i, nmbs.Integral)
               or not isinstance(n, nmbs.Integral))
    if is_bool or not_int:
        raise TypeError('i and n must be integers (bool not allowed)')
    i, n = int(i), int(n)
    return sum(f(j) for j in range(i, n + 1))
    # --------------------------------------------------------


class Stats:
    def __init__(self, nums):
        stg = Settings(nums)
        # بای دیفالت روی ورودی متد flat_and_array رو اعمال کن
        nums = stg.x
        if stg.is_empty():
            raise ValueError("you should not enter None items or empty.")
        try:
            nums = nums.astype(float)
        except ValueError as e:
            raise ValueError(f'items should be numbers: {e}') from e
        except TypeError as e:
            raise TypeError(f'items should be numbers: {e}') from e
        self.nums = nums

    # قضیه چبیشف
    def chebyshev(self, k):
        if not isinstance(k, nmbs.Integral) or k <= 0:
            # k باید عدد باشد و مثبت و غیر صفر
            raise ValueError('k should be number and >0')
        l = self.loc()
        v = self.variance()
        s = math.sqrt(v)
        return float(l - (k * s)), float(l + (k * s))

    # نرمال سازی داده ها
    # normed xᵢ = (xᵢ - μ)/σ
    def norm(self):
        x_s = self.nums
        l = self.loc()
        s = self.scale()
        if s == 0:
            return x_s - l
        return (x_s - l) / s

    # میانگین
    # μ = ∑(xᵢ)/n
    def loc(self):
        return self.nums.mean()

    # واریانس
    # σ² = ∑(xᵢ - μ)²/n
    def variance(self):
        nums = self.nums
        l = self.loc()
        return ((nums - l) ** 2).mean()

    # انحراف معیار
    # σ = √(variance)
    def scale(self):
        return math.sqrt(self.variance())


class Plots:
    @staticmethod
    # نمودار مقایسه دو تابع مدل و واقعی
    def comparison_plot(f_true, f_pred):
        x = np.linspace(1, 10, 100)
        plt.plot(x, f_true(x), label='truth')
        plt.plot(x, f_pred(x), label='guess')
        plt.legend()
        plt.show()
        return True

    @staticmethod
    # ساخت و نمایش نمودار ساده
    def show_plot(x, y):
        plt.plot(x, y)
        plt.show()
        return True

    @staticmethod
    # نمایش نمودار نقطه ای ساده
    def show_scatter(x, y):
        plt.scatter(x, y)
        plt.show()
        return True


class Matrix:
    def __init__(self, M):
        self.M = np.asarray(M)

    # تشخیص نوع ماتریس براساس ماتریس های خاص
    # rectangular, identity, row, column, scalar, square, diagonal
    def detect_matrix(self):
        M = self.M
        M = np.asarray(M)
        if Settings(M).is_empty():
            return None, M.shape
        if M.ndim == 0 or M.ndim == 1:
            M = M.reshape(1, -1)

        if M.ndim > 2:
            raise ValueError(
                'The matrix must ultimately be 2-dimensional.'
            )

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
            return 'null', o

        if n_columns == n_rows:

            for row in range(n_rows):
                if not dgnl:
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
        return typ, o

    # بررسی تساوی ماتریس ها
    def matrix_equal(self, B):
        A = self.M
        B = np.asarray(B)
        if A.shape == B.shape:
            return np.array_equal(A, B)
        return False


class Metrics:
    @staticmethod
    # تابع خطا
    # mse = ∑(yᵢ - ŷᵢ)²/n
    def mse(y_true_array, y_pred_array):
        y_pred_array, y_true_array = (
            Settings(y_pred_array).x, Settings(y_true_array).x
        )
        if y_pred_array.size == y_true_array.size and y_pred_array.size != 0:
            return (
                ((y_true_array - y_pred_array) ** 2).sum()
                / len(y_true_array)
            )
        return None

    @staticmethod
    # جذر تابع خطا
    # rme = √mse
    def rmse(mse_value):
        if mse_value is None:
            return None
        return math.sqrt(mse_value)


class LinearRegression:
    def __init__(self, dataframe, x_name, y_name):
        self.df, self.x_name, self.y_name = dataframe, x_name, y_name
        self.t0, self.t1 = None, None
        self.coefficients()

    # ساخت مدل رگرسیون خطی با فرمول بسته بر اساس ستون های دیتافریم
    # y = θ₀ + θ₁x
    # θ₁ = Σ(xᵢ - x̄)(yᵢ - ȳ)/Σ(xᵢ - x̄)²
    # θ₀ = ȳ - θ₁x̄
    def coefficients(self):
        df, x_name, y_name = self.df, self.x_name, self.y_name
        meanx, meany = Stats(df[x_name]).loc(), Stats(df[y_name]).loc()
        x, y = df[x_name], df[y_name]
        t1 = (sigma(x, y, lambda a: (a - meanx),
                    lambda b: (b - meany), operator='*')
              / sigma(x, f=lambda c: (c - meanx) ** 2))
        self.t0 = meany - t1 * meanx
        self.t1 = t1
        return self

    def predict(self, x):
        return self.t0 + self.t1 * np.asarray(x)

    def model_mse(self, x, y_true):
        x, y_true = np.asarray(x), np.asarray(y_true)
        y_pred = self.predict(x)
        return Metrics.mse(y_true, y_pred)


class DataHandler:
    @staticmethod
    # ساخت سریع دیتاست نمونه از سن و قد
    def make_sample_dataset(n, seed=313):
        np.random.seed(seed)
        noise = np.random.normal(0, 10, n)
        height = np.round(np.random.normal(167, 10, n) + noise, 1)
        age = np.random.randint(18, 60, n)
        data = {
            "age": age,
            "height": height,
        }
        df = pd.DataFrame(data)
        df.to_csv("sample_dataset.csv", encoding='utf-8-sig', index=False)
        return df

    @staticmethod
    # خواندن فایل سی اس وی یا درصورت نبود ساخت فایل توسط تابع داده شده
    # و مقادیر لازم آن
    def read_or_make_csv(address, make_dataset_function, *args):
        try:
            return pd.read_csv(address, encoding='utf-8-sig', index_col=False)
        except FileNotFoundError:
            make_dataset_function(*args).to_csv(
                address, encoding='utf-8-sig', index=False
            )
            return pd.read_csv(
                address, encoding='utf-8-sig', index_col=False
            )


if __name__ == "__main__":
    print(" Hello dear! Thanks for using! ;D\n This file can only be used as a library, not for running! :D")

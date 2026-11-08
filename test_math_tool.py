import os
import unittest

import matplotlib
import numpy as np
import pandas as pd
import sympy as sp

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from math_tool import (
    Settings, Stats, Matrix, Metrics, LinearRegression,
    DataHandler, Plots, derivative, sigma,
)


# تست کلاس Settings
class TestSettings(unittest.TestCase):
    # تست لیست خالی روی متد is_empty
    def test_is_empty_list(self):
        # Arrange
        value = []
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertTrue(result)

    # تست آرایه خالی روی متد is_empty
    def test_is_empty_array(self):
        # Arrange
        value = np.array([])
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertTrue(result)

    # تست آرایه دارای None روی متد is_empty
    def test_is_empty_none_array(self):
        # Arrange
        value = np.array([None])
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertTrue(result)

    # تست لیست دارای None روی متد is_empty
    def test_is_empty_none_list(self):
        # Arrange
        value = [None]
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertTrue(result)

    # تست آرایه ترکیبی دارای None روی متد is_empty
    def test_is_empty_none_array_3d(self):
        # Arrange
        value = np.array(
            [([None, (None, None), None]), [None, [None], [None, [None]]]],
            dtype=object,
        )
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertTrue(result)

    # تست لیست ترکیبی دارای None روی متد is_empty
    def test_is_empty_none_list_3d(self):
        # Arrange
        value = [
            [
                None,
                (None, np.array([None, None], dtype=object)),
                None,
                [None],
                [None, [None]],
                pd.Series([[None, None], [None, None]]),
            ]
        ]
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertTrue(result)

    # تست خروجی فالس متد is_empty
    def test_is_empty_false(self):
        # Arrange
        value = [
            [
                None,
                (None, np.array([None, 1], dtype=object)),
                None,
                [None],
                [None, [None]],
            ]
        ]
        # Act
        result = Settings(value).is_empty()
        # Assert
        self.assertFalse(result)

    # تست متد is_iter
    def test_is_iter(self):
        # Arrange
        value = [1, 2]
        # Act
        result = Settings.is_iter(value)
        # Assert
        self.assertTrue(result)

    # تست خروجی فالس متد is_iter
    def test_is_iter_false(self):
        # Arrange
        value = 12
        # Act
        result = Settings.is_iter(value)
        # Assert
        self.assertFalse(result)

    # تست کامل متد flat_and_array
    def test_flat_and_array(self):
        # Arrange
        value = [
            1, 2,
            [
                [3, [4, (5, 6)]],
                'sardari',
                np.array(
                    [[8, 9], [10, pd.Series([11, 12, "kiarash"])]],
                    dtype=object,
                ),
            ],
        ]
        # Act
        result = Settings(value).flat_and_array()
        # Assert
        out = np.array(
            [1, 2, 3, 4, 5, 6, 'sardari', 8, 9, 10, 11, 12, 'kiarash'],
            dtype=object,
        )
        self.assertEqual(result.tolist(), out.tolist())


# تست کلاس Stats
class TestStats(unittest.TestCase):
    # تست متد چبیشف
    def test_chebyshev(self):
        # Arrange
        value = pd.Series(
            [1, 10, 43, -11, 12, 57, 0],
            [2, 3, 12, -2, -32, 10, 0],
        )
        # Act
        s = Stats(value)
        result = s.chebyshev(3)
        # Assert
        self.assertAlmostEqual(result[0], -52.71, places=2)
        self.assertAlmostEqual(result[1], 84.71, places=2)

    # تست ساده متد norm
    def test_norm(self):
        # Arrange
        value = [1, 2, 3, 4, 5]
        scale = 2 ** 0.5
        expected = (np.array(value) - 3) / scale
        # Act
        result = Stats(value).norm()
        # Assert
        self.assertEqual(result.tolist(), expected.tolist())

    # تست تساوی خروجی دیتاهای شیفت شده در متد norm
    def test_norm_shifted(self):
        # Arrange
        value = [1, 2, 3, 4, 5]
        shifted_value = np.array(value) + 100
        scale = 2 ** 0.5
        expected = (np.array(value) - 3) / scale
        # Act
        shifted_result = Stats(shifted_value).norm()
        # Assert
        self.assertEqual(expected.tolist(), shifted_result.tolist())

    # تست تک عنصر روی متد norm
    def test_norm_single_element(self):
        # Arrange, Act
        result = Stats([42]).norm()
        # Assert
        self.assertEqual(result.tolist(), [0.0])

    # تست ورودی لیست متد loc
    def test_loc_list(self):
        # Arrange
        value = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Act
        result = float(Stats(value).loc())
        expected = float(5)
        # Assert
        self.assertEqual(result, expected)

    # تست ورودی آرایه متد loc
    def test_loc_array(self):
        # Arrange
        value = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected = float(5)
        # Act
        result = float(Stats(value).loc())
        # Assert
        self.assertEqual(result, expected)

    # تست ورودی سریز متد loc
    def test_loc_series(self):
        # Arrange
        value = pd.Series([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = float(5)
        # Act
        result = float(Stats(value).loc())
        # Assert
        self.assertEqual(result, expected)

    # تست ورودی ترکیبی متد loc
    def test_loc_complex(self):
        # Arrange
        value = [[1, np.array([2]), pd.Series([3])], [(4, 5), 6], [[7], 8, 9]]
        expected = float(5)
        # Act
        result = float(Stats(value).loc())
        # Assert
        self.assertEqual(result, expected)

    # تست ورودی ترکیبی متد variance
    def test_variance(self):
        # Arrange
        value = [[1, np.array([2]), pd.Series([3])], [(4, 5), 6], [[7], 8, 9]]
        # Act
        result = Stats(value).variance()
        expected = 6.6667
        # Assert
        self.assertAlmostEqual(result, expected, places=3)

    # تست ورودی ترکیبی متد scale
    def test_scale(self):
        # Arrange
        value = [[1, np.array([2]), pd.Series([3])], [(4, 5), 6], [[7], 8, 9]]
        expected = 6.6667 ** 0.5
        # Act
        result = Stats(value).scale()
        # Assert
        self.assertAlmostEqual(result, expected, places=3)


# تست کلاس ماتریس
class TestMatrix(unittest.TestCase):
    # تست تشخیص ماتریس مستطیلی متد detect_matrix
    def test_detector_rectangular(self):
        # Arrange
        value = [[1, 4, 7], [2, 0, -3]]
        expected = 'rectangular'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (2, 3))

    # تست تشخیص ماتریس همانی متد detect_matrix
    def test_detector_identity(self):
        # Arrange
        value = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        expected = 'identity'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (3, 3))

    # تست تشخیص ماتریس سطری متد detect_matrix
    def test_detector_row(self):
        # Arrange
        value = [[1, 4, 7, 2, 0, -3]]
        expected = 'row'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (1, 6))

    # تست تشخیص ماتریس ستونی متد detect_matrix
    def test_detector_column(self):
        # Arrange
        value = [[1], [2], [-5]]
        expected = 'column'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (3, 1))

    # تست تشخیص ماتریس اسکالر متد detect_matrix
    def test_detector_scalar(self):
        # Arrange
        value = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        expected = 'scalar'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (3, 3))

    # تست تشخیص ماتریس مربعی متد detect_matrix
    def test_detector_square(self):
        # Arrange
        value = [[6, 4, 1], [0, 7, 5], [1, 2, 3]]
        expected = 'square'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (3, 3))

    # تست تشخیص ماتریس قطری متد detect_matrix
    def test_detector_diagonal(self):
        # Arrange
        value = [[0, 0, 0], [0, -3, 0], [0, 0, 4]]
        expected = 'diagonal'
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (3, 3))

    # تست تشخیص ماتریس None متد detect_matrix
    def test_detector_none(self):
        # Arrange
        value = [[None, None, None], [None, None, None], [None, None, None]]
        expected = None
        # Act
        result = Matrix(value).detect_matrix()
        # Assert
        self.assertEqual(result[0], expected)
        self.assertEqual(result[1], (3, 3))

    # تست تساوی دو ماتریس لیست تو در تو و آرایه متد matrix_equal
    def test_matrix_equal(self):
        # Arrange
        A = [[1, 2], [4, 3], [2, 0]]
        B = np.array([[1, 2], [4, 3], [2, 0]])
        # Act
        result = Matrix(A).matrix_equal(B)
        # Assert
        self.assertTrue(result)

    # تست نامساوی بودن دو ماتریس لیست تو در تو و آرایه متد matrix_equal
    def test_matrix_equal_false(self):
        # Arrange
        A = [[1, 1], [4, 3], [2, 0]]
        B = np.array([[1, 2], [4, 3], [2, 0]])
        # Act
        result = Matrix(A).matrix_equal(B)
        # Assert
        self.assertFalse(result)


# تست کلاس Metrics
class TestMetrics(unittest.TestCase):
    # تست ورودی ترکیبی روی متد mse
    def test_mse(self):
        # Arrange
        pred = [[1, np.array([2]), pd.Series([3])], [(4, 5), 6], [[7], 8, 9]]
        true = [[1, np.array([2]), pd.Series([3])], [(4, 5), 6], [[7], 8, 1]]
        expected = 7.111
        # Act
        result = Metrics.mse(true, pred)
        # Assert
        self.assertAlmostEqual(result, expected, places=2)

    # تست متد rmse
    def test_rmse(self):
        # Arrange
        mse = 7.111
        expected = mse ** 0.5
        # Act
        result = Metrics.rmse(mse)
        # Assert
        self.assertAlmostEqual(result, expected, places=2)


# تست کلاس LinearRegression
class TestLinearRegression(unittest.TestCase):
    # تست متد coefficients
    def test_coefficients(self):
        # Arrange
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [7, 4, 1, -2, -5],
        })
        # Act
        model = LinearRegression(df, 'x', 'y')
        # Assert
        self.assertAlmostEqual(float(model.t1), -3.0, places=10)
        self.assertAlmostEqual(float(model.t0), 10.0, places=10)

    # تست متد predict
    def test_predict(self):
        # Arrange
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [7, 4, 1, -2, -5],
        })
        # Act
        model = LinearRegression(df, 'x', 'y')
        result = model.predict(10)
        # Assert
        self.assertAlmostEqual(float(result), -20, places=10)

    # تست متد model_mse
    def test_model_mse(self):
        # Arrange
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [7, 4, 1, -2, -5],
        })
        # Act
        model = LinearRegression(df, 'x', 'y')
        result = model.model_mse([1, 2, 3, 4, 5], [7, 4, 1, -2, -5])
        # Assert
        self.assertAlmostEqual(result, 0.0, places=10)


# تست تابع derivative
class TestDerivative(unittest.TestCase):
    # تست مشتق معمولی متد derivative
    def test_derivative(self):
        # Arrange
        x = sp.Symbol('x')
        f = 2 * x ** 3 + 3 * x - 6
        expected = sp.diff(f, x)
        # Act
        result = derivative(f, x)
        # Assert
        self.assertEqual(result, expected)

    # تست مشتق جزئی متد derivative
    def test_partial_derivative(self):
        # Arrange
        a = sp.Symbol('a')
        b = sp.Symbol('b')
        c = sp.Symbol('c')
        f = 2 * a ** 3 + 3 * b ** 2 - 6 * c ** 4 + 10 * b - 1
        expected1 = sp.diff(f, a)
        expected2 = sp.diff(f, b)
        expected3 = sp.diff(f, c)
        # Act
        result1 = derivative(f, a)
        result2 = derivative(f, b)
        result3 = derivative(f, c)
        # Assert
        self.assertEqual(result1, expected1)
        self.assertEqual(result2, expected2)
        self.assertEqual(result3, expected3)


# تست تابع sigma
class TestSigma(unittest.TestCase):
    # تست با تعیین بازه و تابع f
    def test_sigma_range_f(self):
        # Arrange
        i, n = 0, 10
        # Act
        result = sigma(i, n, lambda x: 2 * x)
        # Assert
        self.assertEqual(result, 110)

    # تست با تعیین بازه و تابع g
    def test_sigma_range_g(self):
        # Arrange
        i, n = 0, 10
        # Act
        result = sigma(i, n, g=lambda x: 2 * x)
        # Assert
        self.assertEqual(result, 110)

    # تست با دادن لیست i به تابع f
    def test_sigma_1_list_i_f(self):
        # Arrange
        i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(i, f=lambda x: 2 * x)
        # Assert
        self.assertEqual(result, 110)

    # تست با دادن لیست i به تابع g
    def test_sigma_1_list_i_g(self):
        # Arrange
        i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(i, g=lambda x: 2 * x)
        # Assert
        self.assertEqual(result, 110)

    # تست با دادن لیست n به تابع f
    def test_sigma_1_list_n_f(self):
        # Arrange
        n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(n=n, f=lambda x: 2 * x)
        # Assert
        self.assertEqual(result, 110)

    # تست با دادن لیست n به تابع g
    def test_sigma_1_list_n_g(self):
        # Arrange
        n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(n=n, g=lambda x: 2 * x)
        # Assert
        self.assertEqual(result, 110)

    # تست با دادن لیست i و اپراتور به تابع f
    def test_sigma_op_1_list_i_f(self):
        # Arrange
        i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(i=i, f=lambda x: 2 * x, operator='-')
        # Assert
        self.assertEqual(result, 0)

    # تست با دادن لیست i و اپراتور به تابع g
    def test_sigma_op_1_list_i_g(self):
        # Arrange
        i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(i=i, g=lambda x: 2 * x, operator='-')
        # Assert
        self.assertEqual(result, 0)

    # تست با دادن لیست n و اپراتور به تابع f
    def test_sigma_op_1_list_n_f(self):
        # Arrange
        n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(n=n, f=lambda x: 2 * x, operator='-')
        # Assert
        self.assertEqual(result, 0)

    # تست با دادن لیست n و اپراتور به تابع g
    def test_sigma_op_1_list_n_g(self):
        # Arrange
        n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Act
        result = sigma(n=n, g=lambda x: 2 * x, operator='-')
        # Assert
        self.assertEqual(result, 0)

    # تست با دادن لیست i و اپراتور به توابع f و g
    def test_sigma_op_1_list_i_f_g(self):
        # Arrange
        i = [1, 2, 3, 4, 5]
        # Act
        result = sigma(
            i=i, f=lambda x: 2 * x, g=lambda x: x - 1, operator='-'
        )
        # Assert
        self.assertEqual(result, 20)

    # تست با دادن لیست n و اپراتور به توابع f و g
    def test_sigma_op_1_list_n_f_g(self):
        # Arrange
        n = [1, 2, 3, 4, 5]
        # Act
        result = sigma(
            n=n, f=lambda x: 2 * x, g=lambda x: x - 1, operator='-'
        )
        # Assert
        self.assertEqual(result, 20)

    # تست با دادن 2 لیست و اپراتور به تابع f
    def test_sigma_op_2_list_f(self):
        # Arrange
        i = [1, 2, 3, 4, 5]
        n = [6, 7, 8, 9, 10]
        # Act
        result = sigma(i, n, f=lambda x: 2 * x, operator=' - ')
        # Assert
        self.assertEqual(result, -50)

    # تست با دادن 2 لیست و اپراتور به تابع g
    def test_sigma_op_2_list_g(self):
        # Arrange
        i = [1, 2, 3, 4, 5]
        n = [6, 7, 8, 9, 10]
        # Act
        result = sigma(i, n, g=lambda x: 2 * x, operator='-  ')
        # Assert
        self.assertEqual(result, -50)

    # تست با دادن 2 لیست و اپراتور به توابع f و g
    def test_sigma_op_2_list_f_g(self):
        # Arrange
        i = [1, 2, 3, 4, 5]
        n = [6, 7, 8, 9, 10]
        # Act
        result = sigma(
            i, n, lambda x: 2 * x, lambda x: x - 1, operator='  -'
        )
        # Assert
        self.assertEqual(result, -5)


# تست اجرا شدن متد های کلاس Plots
class TestPlots(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    # تست ساده ی اجرا شدن متد comparison_plot
    def test_comparison_plot(self):
        # Arrange
        f = lambda x: 2 * x
        g = lambda x: 3 * x + 3
        # Act
        result = Plots.comparison_plot(f, g)
        # Assert
        self.assertTrue(result)

    # تست ساده ی اجرا شدن متد show_plot
    def test_show_plot(self):
        # Arrange
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        # Act
        result = Plots.show_plot(x, y)
        # Assert
        self.assertTrue(result)

    # تست ساده ی اجرا شدن متد show_scatter
    def test_show_scatter(self):
        # Arrange
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        # Act
        result = Plots.show_scatter(x, y)
        # Assert
        self.assertTrue(result)


# تست کلاس DataHandler
class TestDataHandler(unittest.TestCase):
    # حذف فایل در صورت وجود
    def reset_files(self):
        address = "sample_dataset.csv"
        if os.path.exists(address):
            os.remove(address)

    # حذف فایل درصورت وجود ابتدای اجرای هر متد تست
    def setUp(self):
        self.reset_files()

    # حذف در صورت وجود فایل بعد اتمام هر متد تست
    def tearDown(self):
        self.reset_files()

    # تست ساخت فایل متد make_sample_dataset
    def test_make_sample_dataset_making_file_test(self):
        # Arrange
        address = "sample_dataset.csv"
        # Act
        DataHandler.make_sample_dataset(10)
        exist = os.path.exists(address)
        # Assert
        self.assertTrue(exist)

    # تست مساوی بودن خروجی متد با محتوای فایل ساخته شده متد
    # make_sample_dataset
    def test_make_sample_dataset_df_and_file_equality(self):
        # Arrange
        address = "sample_dataset.csv"
        result = (DataHandler.make_sample_dataset(10))
        # Act
        df = pd.read_csv(address, encoding='utf-8-sig', index_col=False)
        # Assert
        self.assertEqual(list(result.columns), list(df.columns))
        self.assertEqual(result.values.tolist(), df.values.tolist())

    # تست متد read_or_make_csv
    def test_read_or_make_csv(self):
        # Arrange
        address = "sample_dataset.csv"
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        # Act
        f = lambda: df
        result = DataHandler.read_or_make_csv(address, f)
        # Assert
        self.assertEqual(list(result.columns), list(df.columns))
        self.assertEqual(result.values.tolist(), df.values.tolist())


if __name__ == "__main__":
    unittest.main()
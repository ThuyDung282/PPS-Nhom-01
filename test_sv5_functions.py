
import unittest
import math
import warnings

from isolate_roots import isolate_roots
from bisection import bisection
from false_position import false_position
from error_estimation import bisection_steps, false_position_error


# ==================================================
# 1. KHAI BAO CAC HAM SO THEO DE BAI
# ==================================================

def f1(x):
    # Ham so thu nhat: x^2 - 5x - 3
    return x**2 - 5*x - 3


def f2(x):
    # Ham so thu hai: 5x^3 - 20x + 3
    return 5*x**3 - 20*x + 3


# Nghiem gan dung tham chieu cua f1
ROOT1 = (5 - math.sqrt(37)) / 2

# Ba nghiem gan dung cua f2
ROOTS2 = [-2.071157, 0.150858, 1.920299]

# Ba khoang phan ly nghiem cua f2 theo de bai
BRACKETS2 = [
    (-2.1, -2.0),
    (0.1, 0.2),
    (1.9, 2.0)
]

# Cac muc sai so can kiem thu
EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


class TestSV5Functions(unittest.TestCase):

    # ==================================================
    # 2. KIEM THU HAM ISOLATE_ROOTS - P1
    # ==================================================

    def test_f1_doi_dau(self):
        # Kiem tra f1 co doi dau tren (-1, 0)
        self.assertLess(f1(-1) * f1(0), 0)

    def test_f2_tim_du_ba_nghiem(self):
        # Quet khoang [-3, 3] de tim cac khoang doi dau
        intervals = isolate_roots(f2, -3, 3, 0.1)

        # Moi nghiem tham chieu phai nam trong mot khoang tim duoc
        for root in ROOTS2:
            found = any(a <= root <= b for a, b in intervals)
            self.assertTrue(
                found,
                f"Khong tim thay khoang chua nghiem {root}"
            )

    def test_isolate_roots_sai_dieu_kien(self):
        # a >= b phai bao loi
        with self.assertRaises(ValueError):
            isolate_roots(f2, 3, -3, 0.1)

        # Buoc quet khong duong phai bao loi
        with self.assertRaises(ValueError):
            isolate_roots(f2, -3, 3, 0)

    # ==================================================
    # 3. KIEM THU PHUONG PHAP CHIA DOI - P2
    # ==================================================

    def test_bisection_f1(self):
        # Tim nghiem f1 tren khoang (-1, 0)
        x, k, err = bisection(
            f1, -1, 0, eps=1e-10, n_max=100
        )

        # Nghiem gan dung phai gan nghiem tham chieu
        self.assertAlmostEqual(x, ROOT1, delta=1e-8)

        # So buoc lap khong am va sai so khong am
        self.assertGreaterEqual(k, 0)
        self.assertGreaterEqual(err, 0)

        # Kiem tra sai so dat yeu cau
        self.assertLessEqual(err, 1e-10)

    def test_bisection_ba_nghiem_f2(self):
        # Kiem tra chia doi tren ca ba khoang phan ly
        for (a, b), root in zip(BRACKETS2, ROOTS2):
            with self.subTest(khoang=(a, b)):
                x, k, err = bisection(
                    f2, a, b, eps=1e-10, n_max=100
                )

                self.assertAlmostEqual(x, root, delta=2e-5)
                self.assertLessEqual(err, 1e-10)

    def test_bisection_khoang_khong_doi_dau(self):
        # Khoang [1, 2] co hai dau ham cung duong
        with self.assertRaises(ValueError):
            bisection(f1, 1, 2, eps=1e-5, n_max=50)

    def test_bisection_dau_mut_la_nghiem(self):
        # f(x) = x - 1 co nghiem tai dau mut a = 1
        x, k, err = bisection(
            lambda x: x - 1, 1, 2
        )
        self.assertEqual(x, 1.0)
        self.assertEqual(k, 0)
        self.assertEqual(err, 0.0)

    # ==================================================
    # 4. KIEM THU PHUONG PHAP DAY CUNG - P3
    # ==================================================

    def test_false_position_f1(self):
        # Tim nghiem f1 bang phuong phap day cung
        x, k, err = false_position(
            f1, -1, 0, eps=1e-10, n_max=1000
        )

        self.assertAlmostEqual(x, ROOT1, delta=1e-8)
        self.assertGreaterEqual(k, 1)
        self.assertGreaterEqual(err, 0)

    def test_false_position_ba_nghiem_f2(self):
        # Kiem tra day cung tren ca ba khoang phan ly
        for (a, b), root in zip(BRACKETS2, ROOTS2):
            with self.subTest(khoang=(a, b)):
                x, k, err = false_position(
                    f2, a, b, eps=1e-10, n_max=10000
                )

                self.assertAlmostEqual(x, root, delta=2e-5)
                self.assertGreaterEqual(k, 1)
                self.assertGreaterEqual(err, 0)

    def test_false_position_sai_dieu_kien(self):
        # Khoang khong doi dau phai bao loi
        with self.assertRaises(ValueError):
            false_position(f1, 1, 2)

        # eps phai duong
        with self.assertRaises(ValueError):
            false_position(f1, -1, 0, eps=0)

        # n_max phai la so nguyen duong
        with self.assertRaises(ValueError):
            false_position(f1, -1, 0, n_max=0)

    def test_false_position_dau_mut_la_nghiem(self):
        # Kiem tra truong hop nghiem nam tai dau mut
        x, k, err = false_position(
            lambda x: x - 1, 1, 2
        )

        self.assertEqual(x, 1.0)
        self.assertEqual(k, 0)
        self.assertEqual(err, 0.0)

    # ==================================================
    # 5. KIEM THU DANH GIA SAI SO - P4
    # ==================================================

    def test_bisection_steps(self):
        # So buoc ly thuyet phai du lon de dat eps
        for eps in EPS_LIST:
            n = bisection_steps(-1, 0, eps)

            self.assertGreaterEqual(n, 0)

            # Theo cong thuc trong code:
            # sai so ly thuyet = (b-a)/2^(n+1)
            theoretical_error = (0 - (-1)) / (2**(n + 1))
            self.assertLessEqual(
                theoretical_error, eps * (1 + 1e-12)
            )

    def test_false_position_error(self):
        # Thu tinh can tren sai so day cung
        err = false_position_error(
            x_k=1.5,
            x_prev=1.4,
            m1=2,
            M1=6
        )

        # ((6 - 2) / 2) * |1.5 - 1.4| = 0.2
        self.assertAlmostEqual(err, 0.2, places=12)

    def test_false_position_error_sai_can(self):
        # m1 phai lon hon 0
        with self.assertRaises(ValueError):
            false_position_error(1.5, 1.4, 0, 6)

        # M1 phai lon hon hoac bang m1
        with self.assertRaises(ValueError):
            false_position_error(1.5, 1.4, 3, 2)

    # ==================================================
    # 6. THUC NGHIEM SO SANH SO BUOC LAP
    # ==================================================

    def test_so_sanh_so_buoc_lap(self):
        # Cac truong hop theo yeu cau de bai
        cases = [
            ("f1", f1, -1, 0),
            ("f2a", f2, -2.1, -2.0),
            ("f2b", f2, 0.1, 0.2),
            ("f2c", f2, 1.9, 2.0)
        ]

        print("\nSO SANH SO BUOC LAP")
        print(
            f"{'Ham':<6}{'eps':<12}"
            f"{'Chia doi':<12}{'Day cung':<12}"
        )

        for name, f, a, b in cases:
            for eps in EPS_LIST:
                with self.subTest(ham=name, eps=eps):
                    # Bo qua canh bao khi het so buoc lap toi da
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        _, kb, _ = bisection(
                            f, a, b, eps=eps, n_max=1000
                        )

                        _, kf, _ = false_position(
                            f, a, b, eps=eps, n_max=10000
                        )

                    print(
                        f"{name:<6}{eps:<12.0e}"
                        f"{kb:<12}{kf:<12}"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import math

from isolate_roots import isolate_roots
from bisection import bisection
from false_position import false_position


# Hai hàm số theo đề bài
def f1(x):
    return x**2 - 5*x - 3


def f2(x):
    return 5*x**3 - 20*x + 3

# Nghiệm tham chiếu
ROOT1 = (5 - math.sqrt(37)) / 2

ROOTS2 = [
    -2.071157,
    0.150858,
    1.920299
]

# Các khoảng phân ly nghiệm của f2
BRACKETS2 = [
    (-2.1, -2.0),
    (0.1, 0.2),
    (1.9, 2.0)
]

EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

class TestFunctionsFromAssignment(unittest.TestCase):

    # P1: Kiểm tra hàm f1 trên khoảng (-1, 0)
    def test_f1_doi_dau(self):
        self.assertLess(f1(-1) * f1(0), 0)

    def test_f1_chia_doi(self):
        x, k, err = bisection(
            f1, -1, 0, eps=1e-10, n_max=100
        )

        self.assertAlmostEqual(x, ROOT1, delta=1e-8)
        self.assertGreaterEqual(k, 1)
        self.assertGreaterEqual(err, 0)

    def test_f1_day_cung(self):
        x, k, err = false_position(
            f1, -1, 0, eps=1e-10, n_max=1000
        )

        self.assertAlmostEqual(x, ROOT1, delta=1e-8)
        self.assertGreaterEqual(k, 1)
        self.assertGreaterEqual(err, 0)

    # P1: Kiểm tra f2 có đủ ba khoảng phân ly nghiệm
    def test_f2_tim_du_ba_khoang(self):
        intervals = isolate_roots(f2, -3, 3, 0.1)

        for root in ROOTS2:
            found = any(
                a <= root <= b for a, b in intervals
            )
            self.assertTrue(
                found,
                f"Không tìm được khoảng chứa nghiệm {root}"
            )

    # P2 và P3: Kiểm tra hai phương pháp trên cả ba nghiệm f2
    def test_f2_chia_doi_va_day_cung(self):
        for (a, b), root in zip(BRACKETS2, ROOTS2):
            with self.subTest(khoang=(a, b)):
                self.assertLess(f2(a) * f2(b), 0)

                xb, _, _ = bisection(
                    f2, a, b, eps=1e-10, n_max=100
                )

                xf, _, _ = false_position(
                    f2, a, b, eps=1e-10, n_max=1000
                )

                self.assertAlmostEqual(
                    xb, root, delta=2e-5
                )
                self.assertAlmostEqual(
                    xf, root, delta=2e-5
                )

    # Thực nghiệm: so sánh số bước lặp theo đề
    def test_bang_so_sanh_so_buoc(self):
        print("\n\nSO SANH CHIA DOI VA DAY CUNG")
        print(
            f"{'Ham':<5} {'Khoang':<18} {'eps':<12} "
            f"{'k chia doi':<12} {'k day cung':<12}"
        )

        cases = [
            ("f1", f1, -1, 0),
            ("f2a", f2, -2.1, -2.0),
            ("f2b", f2, 0.1, 0.2),
            ("f2c", f2, 1.9, 2.0)
        ]

        for name, f, a, b in cases:
            for eps in EPS_LIST:
                with self.subTest(
                    ham=name, eps=eps, khoang=(a, b)
                ):
                    with __import__("warnings").catch_warnings():
                        __import__("warnings").simplefilter("ignore")

                        _, kb, _ = bisection(
                            f, a, b, eps=eps, n_max=1000
                        )

                        _, kf, _ = false_position(
                            f, a, b, eps=eps, n_max=10000
                        )

                    print(
                        f"{name:<5} [{a:>5.1f},{b:>5.1f}] "
                        f"{eps:<12.0e} {kb:<12} {kf:<12}"
                    )

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


import pytest
import math

from isolate_roots import isolate_roots
from bisection import bisection
from false_position import false_position
from bisection_steps import bisection_steps
from false_position_error import false_position_error


# ============================================================
# 2. KHAI BAO HAI PHUONG TRINH TRONG DE BAI
# ============================================================

# Phuong trinh 1: f(x) = 2^x - 5x - 3
def f1(x):
    return 2**x - 5*x - 3


# Phuong trinh 2: f(x) = 5x^3 - 20x + 3
def f2(x):
    return 5*x**3 - 20*x + 3


# Nghiem tham khao cua f1
ROOT1 = -0.453996

# Ba nghiem tham khao cua f2
ROOTS2 = [
    -2.071157,
    0.150858,
    1.920299
]

# Ba khoang phan ly nghiem cua f2
BRACKETS2 = [
    (-2.1, -2.0),
    (0.1, 0.2),
    (1.9, 2.0)
]

EPS = 1e-8
N_MAX = 10000


# ============================================================
# 3. KIEM THU P1 - TIM KHOANG PHAN LY NGHIEM
# ============================================================

class TestP1:

    # TC01: Tim khoang phan ly nghiem cua f1
    def test_f1_isolate(self):
        intervals = isolate_roots(f1, -1, 0, 0.1)

        assert len(intervals) == 1

        for a, b in intervals:
            assert f1(a) * f1(b) <= 0

    # TC02: Tim du 3 khoang phan ly nghiem cua f2
    def test_f2_isolate(self):
        intervals = isolate_roots(f2, -3, 3, 0.1)

        assert len(intervals) == 3

        for a, b in intervals:
            assert f2(a) * f2(b) <= 0

    # TC03: Kiem tra buoc quet bang 0
    def test_step_zero(self):
        with pytest.raises(ValueError):
            isolate_roots(f1, -1, 0, 0)

    # TC04: Kiem tra buoc quet am
    def test_step_negative(self):
        with pytest.raises(ValueError):
            isolate_roots(f2, -3, 3, -0.1)

    # TC05: Kiem tra a bang b
    def test_equal_interval(self):
        with pytest.raises(ValueError):
            isolate_roots(f1, 1, 1, 0.1)

    # TC06: Kiem tra a lon hon b
    def test_reverse_interval(self):
        with pytest.raises(ValueError):
            isolate_roots(f2, 3, -3, 0.1)


# ============================================================
# 4. KIEM THU P2 - PHUONG PHAP CHIA DOI
# ============================================================

class TestP2:

    # TC07: Tim nghiem f1
    def test_bisection_f1(self):
        x, k, err = bisection(
            f1, -1, 0, EPS, N_MAX
        )

        assert abs(x - ROOT1) < 1e-5
        assert abs(f1(x)) < 1e-6
        assert err <= EPS

    # TC08, TC09, TC10: Tim ba nghiem f2
    @pytest.mark.parametrize(
        "a,b,root",
        [
            (-2.1, -2.0, -2.071157),
            (0.1, 0.2, 0.150858),
            (1.9, 2.0, 1.920299)
        ]
    )
    def test_bisection_f2(self, a, b, root):
        x, k, err = bisection(
            f2, a, b, EPS, N_MAX
        )

        assert abs(x - root) < 1e-5
        assert abs(f2(x)) < 1e-6
        assert err <= EPS

    # TC11: Khoang khong phan ly nghiem
    def test_no_root_interval(self):
        with pytest.raises(ValueError):
            bisection(f1, 1, 2)

    # TC12: a lon hon b
    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            bisection(f1, 0, -1)

    # TC13: eps bang 0
    def test_eps_zero(self):
        with pytest.raises(ValueError):
            bisection(f1, -1, 0, 0)

    # TC14: eps am
    def test_eps_negative(self):
        with pytest.raises(ValueError):
            bisection(f1, -1, 0, -1e-5)

    # TC15: n_max am
    def test_nmax_negative(self):
        with pytest.raises(ValueError):
            bisection(f1, -1, 0, EPS, -1)

    # TC16: Het so buoc lap
    def test_max_iterations(self):
        with pytest.warns(UserWarning):
            x, k, err = bisection(
                f1, -1, 0, 1e-30, 0
            )

        assert err > 1e-30


# ============================================================
# 5. KIEM THU P3 - PHUONG PHAP DAY CUNG
# ============================================================

class TestP3:

    # TC17: Tim nghiem f1
    def test_false_position_f1(self):
        x, k, err = false_position(
            f1, -1, 0, EPS, N_MAX
        )

        assert abs(x - ROOT1) < 1e-5
        assert abs(f1(x)) < 1e-6
        assert err < EPS

    # TC18, TC19, TC20: Tim ba nghiem f2
    @pytest.mark.parametrize(
        "a,b,root",
        [
            (-2.1, -2.0, -2.071157),
            (0.1, 0.2, 0.150858),
            (1.9, 2.0, 1.920299)
        ]
    )
    def test_false_position_f2(self, a, b, root):
        x, k, err = false_position(
            f2, a, b, EPS, N_MAX
        )

        assert abs(x - root) < 1e-5
        assert abs(f2(x)) < 1e-6
        assert err < EPS

    # TC21: Ham dau vao khong phai ham so
    def test_invalid_function(self):
        with pytest.raises(TypeError):
            false_position(123, -1, 0, EPS, N_MAX)

    # TC22: Khoang a >= b
    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            false_position(f1, 0, -1, EPS, N_MAX)

    # TC23: eps bang 0
    def test_eps_zero(self):
        with pytest.raises(ValueError):
            false_position(f1, -1, 0, 0, N_MAX)

    # TC24: eps am
    def test_eps_negative(self):
        with pytest.raises(ValueError):
            false_position(f1, -1, 0, -EPS, N_MAX)

    # TC25: n_max bang 0
    def test_nmax_zero(self):
        with pytest.raises(ValueError):
            false_position(f1, -1, 0, EPS, 0)

    # TC26: n_max am
    def test_nmax_negative(self):
        with pytest.raises(ValueError):
            false_position(f1, -1, 0, EPS, -1)

    # TC27: n_max la so thuc
    def test_nmax_float(self):
        with pytest.raises(ValueError):
            false_position(f1, -1, 0, EPS, 10.5)

    # TC28: n_max la bool
    def test_nmax_bool(self):
        with pytest.raises(ValueError):
            false_position(f1, -1, 0, EPS, True)

    # TC29: Khoang khong doi dau
    def test_no_sign_change(self):
        with pytest.raises(ValueError):
            false_position(f1, 1, 2, EPS, N_MAX)

    # TC30: Het so buoc lap
    def test_max_iterations(self):
        with pytest.warns(RuntimeWarning):
            x, k, err = false_position(
                f1, -1, 0, 1e-30, 1
            )

        assert k == 1
        assert err > 1e-30


# ============================================================
# 6. KIEM THU P4 - DANH GIA SAI SO
# ============================================================

class TestP4:

    # TC31: Kiem tra so buoc lap ly thuyet cua f1
    def test_bisection_steps_f1(self):
        n = bisection_steps(-1, 0, EPS)

        expected = max(
            0,
            math.ceil(math.log2(1 / EPS)) - 1
        )

        assert n == expected

    # TC32: Kiem tra so buoc lap ly thuyet cua f2
    def test_bisection_steps_f2(self):
        for a, b in BRACKETS2:
            n = bisection_steps(a, b, EPS)

            expected = max(
                0,
                math.ceil(math.log2((b-a) / EPS)) - 1
            )

            assert n == expected

    # TC33: Khoang khong hop le
    def test_steps_invalid_interval(self):
        with pytest.raises(ValueError):
            bisection_steps(0, -1, EPS)

    # TC34: Sai so bang 0
    def test_steps_eps_zero(self):
        with pytest.raises(ValueError):
            bisection_steps(-1, 0, 0)

    # TC35: Kiem tra cong thuc sai so day cung
    def test_false_position_error(self):
        x_k = -0.45
        x_prev = -0.46

        # Can dao ham cua |f1'(x)| tren [-1, 0]
        m1 = 5 - math.log(2)
        M1 = 5 - math.log(2) / 2

        err = false_position_error(
            x_k, x_prev, m1, M1
        )

        expected = (
            (M1 - m1) / m1
        ) * abs(x_k - x_prev)

        assert math.isclose(err, expected)

    # TC36: m1 bang 0
    def test_m1_zero(self):
        with pytest.raises(ValueError):
            false_position_error(-0.45, -0.46, 0, 5)

    # TC37: m1 am
    def test_m1_negative(self):
        with pytest.raises(ValueError):
            false_position_error(-0.45, -0.46, -1, 5)

    # TC38: M1 nho hon m1
    def test_M1_less_than_m1(self):
        with pytest.raises(ValueError):
            false_position_error(-0.45, -0.46, 5, 2)


# ============================================================
# 7. KIEM THU CHEO P1 - P2
# P1 tim khoang, P2 giai nghiem
# ============================================================

class TestCrossP1P2:

    # TC39: Ket hop hai thanh vien voi f1
    def test_f1(self):
        intervals = isolate_roots(f1, -1, 0, 0.1)

        for a, b in intervals:
            x, k, err = bisection(
                f1, a, b, EPS, N_MAX
            )

            assert abs(f1(x)) < 1e-6
            assert abs(x - ROOT1) < 1e-5

    # TC40: Ket hop hai thanh vien voi f2
    def test_f2(self):
        intervals = isolate_roots(f2, -3, 3, 0.1)

        assert len(intervals) == 3

        for (a, b), root in zip(intervals, ROOTS2):
            x, k, err = bisection(
                f2, a, b, EPS, N_MAX
            )

            assert abs(f2(x)) < 1e-6
            assert abs(x - root) < 1e-5


# ============================================================
# 8. KIEM THU CHEO P1 - P3
# P1 tim khoang, P3 giai nghiem
# ============================================================

class TestCrossP1P3:

    # TC41: Ket hop hai thanh vien voi f1
    def test_f1(self):
        intervals = isolate_roots(f1, -1, 0, 0.1)

        for a, b in intervals:
            x, k, err = false_position(
                f1, a, b, EPS, N_MAX
            )

            assert abs(f1(x)) < 1e-6
            assert abs(x - ROOT1) < 1e-5

    # TC42: Ket hop hai thanh vien voi f2
    def test_f2(self):
        intervals = isolate_roots(f2, -3, 3, 0.1)

        assert len(intervals) == 3

        for (a, b), root in zip(intervals, ROOTS2):
            x, k, err = false_position(
                f2, a, b, EPS, N_MAX
            )

            assert abs(f2(x)) < 1e-6
            assert abs(x - root) < 1e-5


# ============================================================
# 9. KIEM THU CHEO P2 - P3
# SO SANH HAI PHUONG PHAP
# ============================================================

class TestCrossP2P3:

    # TC43: So sanh nghiem f1
    def test_compare_f1(self):
        xb, kb, eb = bisection(
            f1, -1, 0, EPS, N_MAX
        )

        xf, kf, ef = false_position(
            f1, -1, 0, EPS, N_MAX
        )

        assert abs(xb - xf) < 1e-6

    # TC44, TC45, TC46: So sanh ba nghiem f2
    @pytest.mark.parametrize(
        "a,b",
        BRACKETS2
    )
    def test_compare_f2(self, a, b):
        xb, kb, eb = bisection(
            f2, a, b, EPS, N_MAX
        )

        xf, kf, ef = false_position(
            f2, a, b, EPS, N_MAX
        )

        assert abs(xb - xf) < 1e-6


# ============================================================
# 10. KIEM THU CHEO P2 - P4
# ============================================================

class TestCrossP2P4:

    # TC47: So sanh so buoc lap ly thuyet va thuc te
    def test_f1_steps(self):
        n_theory = bisection_steps(-1, 0, EPS)

        x, k, err = bisection(
            f1, -1, 0, EPS, N_MAX
        )

        assert k <= n_theory
        assert err <= EPS


# ============================================================
# 11. CHAY BO KIEM THU
# ============================================================

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])

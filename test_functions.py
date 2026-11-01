import math
import warnings

import pytest

# P1 – Tìm khoảng phân ly nghiệm
from isolate_roots import isolate_roots

# P2 – Phương pháp chia đôi
from bisection import bisection

# P3 – Phương pháp dây cung
from false_position import false_position

# P4 – Đánh giá sai số
from error_estimation import ( bisection_steps,false_position_error)

# =========================
# P1 – isolate_roots
# =========================

def test_isolate_roots_finds_sign_change_intervals():
    f = lambda x: x**2 - 2
    intervals = isolate_roots(f, 0, 2, 0.25)

    assert len(intervals) == 1
    left, right = intervals[0]
    assert left < right
    assert f(left) * f(right) < 0


def test_isolate_roots_finds_root_on_grid_point():
    f = lambda x: x
    intervals = isolate_roots(f, -1, 1, 0.5)

    assert (0.0, 0.0) in intervals


def test_isolate_roots_finds_root_at_left_endpoint():
    intervals = isolate_roots(lambda x: x, 0, 2, 0.5)
    assert (0.0, 0.0) in intervals


def test_isolate_roots_finds_root_at_right_endpoint():
    intervals = isolate_roots(lambda x: x - 2, 0, 2, 0.5)
    assert (2.0, 2.0) in intervals


def test_isolate_roots_returns_empty_when_no_root_is_bracketed():
    intervals = isolate_roots(lambda x: x**2 + 1, -2, 2, 0.25)
    assert intervals == []


@pytest.mark.parametrize(
    "a,b,step",
    [
        (1, 1, 0.1),
        (2, 1, 0.1),
        (0, 1, 0),
        (0, 1, -0.1),
    ],
)
def test_isolate_roots_rejects_invalid_interval_or_step(a, b, step):
    with pytest.raises(ValueError):
        isolate_roots(lambda x: x, a, b, step)


# =========================
# P2 – bisection
# =========================

def test_bisection_approximates_sqrt_2():
    x, k, err = bisection(lambda t: t**2 - 2, 1, 2, eps=1e-8, n_max=100)

    assert math.isclose(x, math.sqrt(2), abs_tol=2e-8)
    assert k > 0
    assert err <= 1e-8


def test_bisection_returns_left_endpoint_if_it_is_exact_root():
    x, k, err = bisection(lambda t: t - 1, 1, 2)
    assert (x, k, err) == (1.0, 0, 0.0)


def test_bisection_returns_right_endpoint_if_it_is_exact_root():
    x, k, err = bisection(lambda t: t - 2, 1, 2)
    assert (x, k, err) == (2.0, 0, 0.0)


def test_bisection_stops_when_midpoint_is_exact_root():
    x, k, err = bisection(lambda t: t - 1.5, 1, 2, eps=1e-12)
    assert x == 1.5
    assert k == 1
    assert err == 0.0


def test_bisection_warns_when_max_iterations_are_exhausted():
    with pytest.warns(UserWarning):
        bisection(lambda t: t**2 - 2, 1, 2, eps=1e-15, n_max=0)


@pytest.mark.parametrize(
    "a,b,eps,n_max",
    [
        (2, 1, 1e-5, 10),
        (1, 2, 0, 10),
        (1, 2, -1e-5, 10),
        (1, 2, 1e-5, -1),
    ],
)
def test_bisection_rejects_invalid_arguments(a, b, eps, n_max):
    with pytest.raises(ValueError):
        bisection(lambda t: t**2 - 2, a, b, eps, n_max)


def test_bisection_rejects_interval_without_sign_change():
    with pytest.raises(ValueError):
        bisection(lambda t: t**2 + 1, -1, 1)


# =========================
# P2 – bisection_steps
# =========================

def test_bisection_steps_matches_theoretical_formula():
    # ceil(log2((1 - 0) / 0.25)) - 1 = 1
    assert bisection_steps(0, 1, 0.25) == 1


def test_bisection_steps_never_returns_negative_number():
    assert bisection_steps(0, 1, 2) == 0


@pytest.mark.parametrize(
    "a,b,eps",
    [
        (1, 1, 0.1),
        (2, 1, 0.1),
        (0, 1, 0),
        (0, 1, -0.1),
    ],
)
def test_bisection_steps_rejects_invalid_arguments(a, b, eps):
    with pytest.raises(ValueError):
        bisection_steps(a, b, eps)

# =========================
# P3 – false_position
# =========================

def test_false_position_approximates_sqrt_2():
    x, k, err = false_position(
        lambda t: t**2 - 2, 1, 2, eps=1e-8, n_max=100
    )

    assert math.isclose(x, math.sqrt(2), abs_tol=1e-7)
    assert 1 <= k <= 100
    assert err < 1e-8


@pytest.mark.xfail(strict=True, reason="Lỗi hiện tại của P3: cập nhật fa làm mẫu số bằng 0 ở lần lặp sau.")
def test_false_position_works_when_left_endpoint_value_is_positive():
    # Nghiệm nằm trong (1, 2), f(1) > 0 và f(2) < 0.
    x, k, err = false_position(
        lambda t: 1.5 - t, 1, 2, eps=1e-10, n_max=20
    )

    assert math.isclose(x, 1.5, abs_tol=1e-10)
    assert k >= 1
    assert err < 1e-10


def test_false_position_returns_exact_endpoint_root():
    x, k, err = false_position(
        lambda x: x - 1,
        1, 2,
        1e-5, 50
    )

    assert x == 1.0
    assert k == 0
    assert err == 0.0


def test_false_position_rejects_interval_without_sign_change():
    with pytest.raises(ValueError):
        false_position(
            lambda x: x**2 + 1,
            -1, 1,
            1e-5, 50
        )

@pytest.mark.parametrize(
    "a,b,eps,n_max",
    [
        (2, 1, 1e-5, 10),
        (1, 2, 0, 10),
        (1, 2, -1e-5, 10),
        (1, 2, 1e-5, 0),
        (1, 2, 1e-5, 1.5),
    ],
)
def test_false_position_rejects_invalid_arguments(a, b, eps, n_max):
    with pytest.raises(ValueError):
        false_position(lambda t: t**2 - 2, a, b, eps, n_max)


def test_false_position_rejects_non_callable_function():
    with pytest.raises(TypeError):
        false_position(123, 1, 2)


def test_false_position_warns_when_max_iterations_are_exhausted():
    with pytest.warns(RuntimeWarning):
        false_position(lambda t: t**2 - 2, 1, 2, eps=1e-15, n_max=1)


# =========================
# P4 – false_position_error
# =========================

def test_false_position_error_matches_formula():
    # ((M1 - m1) / m1) * |x_k - x_prev|
    result = false_position_error(1.4, 1.3, 2, 5)
    assert math.isclose(result, 0.15, rel_tol=1e-12)


def test_false_position_error_is_zero_when_two_iterates_are_equal():
    assert false_position_error(1.5, 1.5, 2, 5) == 0.0


def test_false_position_error_rejects_nonpositive_m1():
    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 0, 5)


def test_false_position_error_rejects_M1_smaller_than_m1():
    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 5, 2)


# =========================
# KIỂM THỬ CHÉO – so sánh P2 và P3
# =========================

@pytest.mark.parametrize(
    "f,a,b,expected",
    [
        (lambda x: x**2 - 2, 1, 2, math.sqrt(2)),
        (lambda x: x**3 - x - 2, 1, 2, 1.5213797068045676),
        pytest.param(
            lambda x: math.cos(x) - x, 0, 1, 0.7390851332151607,
            marks=pytest.mark.xfail(
                strict=True,
                reason="P3 hiện phát sinh ZeroDivisionError do chọn/cập nhật mốc sai.",
            ),
        ),
    ],
)
def test_bisection_and_false_position_agree_on_same_root(f, a, b, expected):
    xb, _, eb = bisection(f, a, b, eps=1e-8, n_max=100)
    xf, _, ef = false_position(f, a, b, eps=1e-8, n_max=200)

    assert math.isclose(xb, expected, abs_tol=2e-7)
    assert math.isclose(xf, expected, abs_tol=2e-6)
    assert abs(f(xb)) < 1e-6
    assert abs(f(xf)) < 1e-5
    assert eb <= 1e-8
    assert ef < 1e-8

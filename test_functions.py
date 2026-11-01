

import math
import pytest

# =====================================================
# IMPORT CÁC HÀM CẦN KIỂM THỬ
# =====================================================

# P1 - Tìm khoảng phân ly nghiệm
from isolate_roots import isolate_roots

# P2 - Phương pháp chia đôi
from bisection import bisection

# P3 - Phương pháp dây cung
from false_position import false_position

# P4 - Đánh giá sai số
from error_estimation import (
    bisection_steps,
    false_position_error
)


# =====================================================
# P1 - KIỂM THỬ HÀM isolate_roots
# =====================================================

def test_isolate_roots_finds_sign_change_intervals():
    # Kiểm tra khả năng tìm khoảng phân ly nghiệm
    f = lambda x: x**2 - 2

    intervals = isolate_roots(f, 0, 2, 0.25)

    # Phải tìm được một khoảng chứa nghiệm
    assert len(intervals) == 1

    a, b = intervals[0]

    # Hai đầu mút phải có giá trị hàm trái dấu
    assert a < b
    assert f(a) * f(b) < 0


def test_isolate_roots_finds_root_on_grid_point():
    # Kiểm tra nghiệm nằm đúng tại điểm lưới
    f = lambda x: x

    intervals = isolate_roots(f, -1, 1, 0.5)

    assert (0.0, 0.0) in intervals


def test_isolate_roots_finds_root_at_left_endpoint():
    # Kiểm tra nghiệm tại đầu trái
    intervals = isolate_roots(lambda x: x, 0, 2, 0.5)

    assert (0.0, 0.0) in intervals


def test_isolate_roots_finds_root_at_right_endpoint():
    # Kiểm tra nghiệm tại đầu phải
    intervals = isolate_roots(lambda x: x - 2, 0, 2, 0.5)

    assert (2.0, 2.0) in intervals


def test_isolate_roots_returns_empty_when_no_root_is_bracketed():
    # Hàm luôn dương, không có khoảng đổi dấu
    intervals = isolate_roots(
        lambda x: x**2 + 1, -2, 2, 0.25
    )

    assert intervals == []


def test_isolate_roots_finds_all_three_cubic_roots():
    # Phương trình theo đề bài:
    # 5x^3 - 20x + 3 = 0
    f = lambda x: 5*x**3 - 20*x + 3

    # Tìm khoảng phân ly nghiệm trên [-3, 3]
    intervals = isolate_roots(f, -3, 3, 0.1)

    # P1 phải tìm đủ ba khoảng phân ly nghiệm
    assert len(intervals) == 3

    # Ba nghiệm gần đúng theo đề bài
    expected_roots = [
        -2.071157,
        0.150858,
        1.920299
    ]

    # Kiểm tra từng nghiệm nằm trong một khoảng tìm được
    for root in expected_roots:
        assert any(
            a <= root <= b for a, b in intervals
        )

    # Kiểm tra ba khoảng theo thứ tự tăng dần
    for i in range(len(intervals) - 1):
        assert intervals[i][1] < intervals[i + 1][0]

    # Kiểm tra điều kiện đổi dấu
    for a, b in intervals:
        assert f(a) * f(b) < 0


@pytest.mark.parametrize(
    "a,b,step",
    [
        (1, 1, 0.1),
        (2, 1, 0.1),
        (0, 1, 0),
        (0, 1, -0.1),
    ],
)
def test_isolate_roots_rejects_invalid_arguments(a, b, step):
    # Kiểm tra dữ liệu đầu vào không hợp lệ
    with pytest.raises(ValueError):
        isolate_roots(lambda x: x, a, b, step)


# =====================================================
# P2 - KIỂM THỬ PHƯƠNG PHÁP CHIA ĐÔI
# =====================================================

def test_bisection_solves_first_equation():
    # Phương trình theo đề bài:
    # 2^x - 5x - 3 = 0
    f = lambda x: 2**x - 5*x - 3

    # Nghiệm thuộc khoảng (-1, 0)
    x, k, err = bisection(
        f, -1, 0,
        eps=1e-8,
        n_max=200
    )

    # Nghiệm tham chiếu x* = -0.453996
    assert math.isclose(x, -0.453996, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert abs(f(x)) < 1e-5

    # Kiểm tra số bước lặp và sai số
    assert k > 0
    assert err < 1e-8


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (-2.1, -2.0, -2.071157),
        (0.1, 0.2, 0.150858),
        (1.9, 2.0, 1.920299),
    ],
)
def test_bisection_solves_three_cubic_roots(a, b, expected):
    # Phương trình bậc ba theo đề bài
    f = lambda x: 5*x**3 - 20*x + 3

    # Tìm nghiệm bằng phương pháp chia đôi
    x, k, err = bisection(
        f, a, b,
        eps=1e-8,
        n_max=200
    )

    assert math.isclose(x, expected, abs_tol=2e-6)
    assert abs(f(x)) < 1e-5
    assert k > 0
    assert err < 1e-8


def test_bisection_returns_left_endpoint_if_exact_root():
    # Kiểm tra nghiệm tại đầu trái
    x, k, err = bisection(lambda x: x - 1, 1, 2)

    assert (x, k, err) == (1.0, 0, 0.0)


def test_bisection_returns_right_endpoint_if_exact_root():
    # Kiểm tra nghiệm tại đầu phải
    x, k, err = bisection(lambda x: x - 2, 1, 2)

    assert (x, k, err) == (2.0, 0, 0.0)


def test_bisection_stops_when_midpoint_is_exact_root():
    # Trung điểm x = 1.5 là nghiệm chính xác
    x, k, err = bisection(
        lambda x: x - 1.5,
        1, 2,
        eps=1e-12
    )

    assert x == 1.5
    assert k == 1
    assert err == 0.0


def test_bisection_rejects_interval_without_sign_change():
    # Ca biên: f(a)*f(b) > 0
    with pytest.raises(ValueError):
        bisection(lambda x: x**2 + 1, -1, 1)


def test_bisection_warns_when_max_iterations_are_exhausted():
    # Ca biên: hết số bước lặp mà chưa đạt sai số
    with pytest.warns(UserWarning):
        bisection(
            lambda x: x**2 - 2,
            1, 2,
            eps=1e-15,
            n_max=0
        )


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
        bisection(lambda x: x**2 - 2, a, b, eps, n_max)


# =====================================================
# P3 - KIỂM THỬ PHƯƠNG PHÁP DÂY CUNG
# =====================================================

def test_false_position_solves_first_equation():
    # Phương trình theo đề bài:
    # 2^x - 5x - 3 = 0
    f = lambda x: 2**x - 5*x - 3

    # Khoảng tìm nghiệm (-1, 0)
    x, k, err = false_position(
        f, -1, 0,
        eps=1e-8,
        n_max=200
    )

    assert math.isclose(x, -0.453996, abs_tol=2e-6)
    assert abs(f(x)) < 1e-5
    assert k > 0
    assert err < 1e-8


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (-2.1, -2.0, -2.071157),
        (0.1, 0.2, 0.150858),
        (1.9, 2.0, 1.920299),
    ],
)
def test_false_position_solves_three_cubic_roots(a, b, expected):
    # Phương trình bậc ba theo đề bài
    f = lambda x: 5*x**3 - 20*x + 3

    # Tìm nghiệm bằng phương pháp dây cung
    x, k, err = false_position(
        f, a, b,
        eps=1e-8,
        n_max=200
    )

    assert math.isclose(x, expected, abs_tol=2e-6)
    assert abs(f(x)) < 1e-5
    assert k > 0
    assert err < 1e-8


def test_false_position_returns_exact_endpoint_root():
    # Kiểm tra nghiệm tại đầu trái
    x, k, err = false_position(
        lambda x: x - 1,
        1, 2,
        1e-5, 50
    )

    assert x == 1.0
    assert k == 0
    assert err == 0.0


def test_false_position_rejects_interval_without_sign_change():
    # Ca biên: f(a)*f(b) > 0
    with pytest.raises(ValueError):
        false_position(
            lambda x: x**2 + 1,
            -1, 1,
            1e-5, 50
        )


def test_false_position_rejects_non_callable_function():
    # Hàm đầu vào phải là callable
    with pytest.raises(TypeError):
        false_position(123, 1, 2)


def test_false_position_warns_when_max_iterations_are_exhausted():
    # Ca biên: hết n_max bước mà chưa đạt eps
    with pytest.warns(RuntimeWarning):
        false_position(
            lambda x: x**2 - 2,
            1, 2,
            eps=1e-15,
            n_max=1
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
        false_position(
            lambda x: x**2 - 2,
            a, b, eps, n_max
        )


# =====================================================
# P4 - KIỂM THỬ ĐÁNH GIÁ SAI SỐ
# =====================================================

def test_bisection_steps_matches_theoretical_formula():
    # Kiểm tra công thức tính số bước chia đôi
    assert bisection_steps(0, 1, 0.25) == 1


def test_bisection_steps_never_returns_negative_number():
    # Số bước lặp lý thuyết không được âm
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


def test_false_position_error_matches_formula():
    # Kiểm tra công thức đánh giá sai số dây cung
    result = false_position_error(1.4, 1.3, 2, 5)

    assert math.isclose(result, 0.15, rel_tol=1e-12)


def test_false_position_error_is_zero_when_iterates_are_equal():
    # Hai nghiệm liên tiếp bằng nhau thì sai số bằng 0
    assert false_position_error(1.5, 1.5, 2, 5) == 0.0


def test_false_position_error_rejects_nonpositive_m1():
    # m1 phải lớn hơn 0
    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 0, 5)


def test_false_position_error_rejects_M1_smaller_than_m1():
    # M1 không được nhỏ hơn m1
    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 5, 2)


# =====================================================
# KIỂM THỬ CHÉO - SO SÁNH P2 VÀ P3
# =====================================================

@pytest.mark.parametrize(
    "f,a,b,expected",
    [
        # Phương trình thứ nhất theo đề bài
        (
            lambda x: 2**x - 5*x - 3,
            -1, 0,
            -0.453996
        ),

        # Phương trình bậc ba - nghiệm thứ nhất
        (
            lambda x: 5*x**3 - 20*x + 3,
            -2.1, -2.0,
            -2.071157
        ),

        # Phương trình bậc ba - nghiệm thứ hai
        (
            lambda x: 5*x**3 - 20*x + 3,
            0.1, 0.2,
            0.150858
        ),

        # Phương trình bậc ba - nghiệm thứ ba
        (
            lambda x: 5*x**3 - 20*x + 3,
            1.9, 2.0,
            1.920299
        ),
    ],
)
def test_bisection_and_false_position_agree_on_same_root(
    f, a, b, expected
):
    # Tìm nghiệm bằng phương pháp chia đôi
    xb, kb, eb = bisection(
        f, a, b,
        eps=1e-8,
        n_max=200
    )

    # Tìm nghiệm bằng phương pháp dây cung
    xf, kf, ef = false_position(
        f, a, b,
        eps=1e-8,
        n_max=200
    )

    # Kiểm tra nghiệm P2 gần với nghiệm tham chiếu
    assert math.isclose(xb, expected, abs_tol=2e-6)

    # Kiểm tra nghiệm P3 gần với nghiệm tham chiếu
    assert math.isclose(xf, expected, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại hai nghiệm gần bằng 0
    assert abs(f(xb)) < 1e-5
    assert abs(f(xf)) < 1e-5

    # Kiểm tra sai số của hai phương pháp
    assert eb < 1e-8
    assert ef < 1e-8

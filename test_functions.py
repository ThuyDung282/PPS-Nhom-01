
import math
import pytest

# =====================================================
# IMPORT CÁC HÀM CẦN KIỂM THỬ
# =====================================================

# P1: Tìm khoảng phân ly nghiệm
from isolate_roots import isolate_roots

# P2: Phương pháp chia đôi
from bisection import bisection

# P3: Phương pháp dây cung
from false_position import false_position

# P4: Đánh giá sai số
from error_estimation import (
    bisection_steps,
    false_position_error
)


# =====================================================
# KHAI BÁO CÁC PHƯƠNG TRÌNH THEO ĐỀ BÀI
# =====================================================

# Phương trình thứ nhất:
# 2^x - 5x - 3 = 0
def f1(x):
    return 2**x - 5*x - 3


# Phương trình thứ hai:
# 5x^3 - 20x + 3 = 0
def f2(x):
    return 5*x**3 - 20*x + 3


# Ba nghiệm gần đúng của phương trình thứ hai
# Mỗi phần tử gồm: (đầu trái, đầu phải, nghiệm tham chiếu)
CUBIC_ROOTS = [
    (-2.1, -2.0, -2.071157),
    (0.1, 0.2, 0.150858),
    (1.9, 2.0, 1.920299),
]


# =====================================================
# P1 - KIỂM THỬ HÀM isolate_roots
# =====================================================

def test_isolate_roots_finds_first_equation():
    """
    Kiểm tra tìm khoảng phân ly nghiệm của phương trình
    2^x - 5x - 3 = 0 trên đoạn [-1, 0].
    """

    # Tìm khoảng phân ly nghiệm với bước chia 0.1
    intervals = isolate_roots(f1, -1, 0, 0.1)

    # Phương trình có một nghiệm trên đoạn khảo sát
    assert len(intervals) == 1

    # Lấy hai đầu mút của khoảng tìm được
    a, b = intervals[0]

    # Kiểm tra thứ tự hai đầu mút
    assert a < b

    # Kiểm tra điều kiện đổi dấu
    assert f1(a) * f1(b) < 0

    # Kiểm tra nghiệm tham chiếu nằm trong khoảng
    assert a <= -0.453996 <= b


def test_isolate_roots_finds_all_three_cubic_roots():
    """
    Kiểm tra tìm đủ ba khoảng phân ly nghiệm của phương trình
    5x^3 - 20x + 3 = 0 trên đoạn [-3, 3].
    """

    # Tìm các khoảng phân ly nghiệm với bước chia 0.1
    intervals = isolate_roots(f2, -3, 3, 0.1)

    # Phải tìm được đúng ba khoảng
    assert len(intervals) == 3

    # Ba nghiệm gần đúng theo đề bài phải nằm trong các khoảng
    expected_roots = [
        -2.071157,
        0.150858,
        1.920299
    ]

    for root in expected_roots:
        assert any(
            a <= root <= b for a, b in intervals
        )

    # Kiểm tra các khoảng được sắp xếp tăng dần
    for i in range(len(intervals) - 1):
        assert intervals[i][1] < intervals[i + 1][0]

    # Kiểm tra hai đầu mút mỗi khoảng có giá trị trái dấu
    for a, b in intervals:
        assert f2(a) * f2(b) < 0


def test_isolate_roots_returns_empty_for_interval_without_root():
    """
    Kiểm tra trường hợp phương trình thứ nhất không có nghiệm
    trong đoạn khảo sát [1, 2].
    """

    intervals = isolate_roots(f1, 1, 2, 0.1)

    # Kết quả mong đợi là danh sách rỗng
    assert intervals == []


@pytest.mark.parametrize(
    "a, b, step",
    [
        (1, 1, 0.1),    # Hai đầu mút bằng nhau
        (2, 1, 0.1),    # Đầu trái lớn hơn đầu phải
        (0, 1, 0),      # Bước chia bằng 0
        (0, 1, -0.1),   # Bước chia âm
    ],
)
def test_isolate_roots_rejects_invalid_arguments(a, b, step):
    """
    Kiểm tra hàm isolate_roots báo lỗi khi nhận
    các tham số đầu vào không hợp lệ.
    """

    with pytest.raises(ValueError):
        isolate_roots(f1, a, b, step)


# =====================================================
# P2 - KIỂM THỬ PHƯƠNG PHÁP CHIA ĐÔI
# =====================================================

def test_bisection_solves_first_equation():
    """
    Kiểm tra phương pháp chia đôi với phương trình
    2^x - 5x - 3 = 0 trên đoạn [-1, 0].
    """

    # Tìm nghiệm gần đúng với eps = 1e-8
    x, k, err = bisection(
        f1, -1, 0,
        eps=1e-8,
        n_max=200
    )

    # So sánh với nghiệm tham chiếu
    assert math.isclose(x, -0.453996, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert abs(f1(x)) < 1e-5

    # Kiểm tra số bước lặp và sai số
    assert k > 0
    assert err < 1e-8


@pytest.mark.parametrize(
    "a, b, expected",
    CUBIC_ROOTS,
)
def test_bisection_solves_three_cubic_roots(a, b, expected):
    """
    Kiểm tra phương pháp chia đôi tìm đủ ba nghiệm
    của phương trình 5x^3 - 20x + 3 = 0.
    """

    # Tìm nghiệm trong khoảng tương ứng
    x, k, err = bisection(
        f2, a, b,
        eps=1e-8,
        n_max=200
    )

    # Kiểm tra nghiệm gần với nghiệm tham chiếu
    assert math.isclose(x, expected, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert abs(f2(x)) < 1e-5

    # Kiểm tra số bước lặp và sai số
    assert k > 0
    assert err < 1e-8


def test_bisection_returns_exact_left_endpoint_root():
    """
    Kiểm tra trường hợp nghiệm nằm tại đầu trái.
    Tạo hàm từ phương trình thứ nhất để có nghiệm chính xác
    tại x = -1.
    """

    # Tạo hàm g sao cho g(-1) = 0
    g = lambda x: f1(x) - f1(-1)

    x, k, err = bisection(g, -1, 0)

    # Kiểm tra chương trình trả về nghiệm ngay tại đầu trái
    assert x == -1.0
    assert k == 0
    assert err == 0.0


def test_bisection_returns_exact_right_endpoint_root():
    """
    Kiểm tra trường hợp nghiệm nằm tại đầu phải.
    Tạo hàm từ phương trình thứ nhất để có nghiệm chính xác
    tại x = 0.
    """

    # Tạo hàm g sao cho g(0) = 0
    g = lambda x: f1(x) - f1(0)

    x, k, err = bisection(g, -1, 0)

    # Kiểm tra chương trình trả về nghiệm tại đầu phải
    assert x == 0.0
    assert k == 0
    assert err == 0.0


def test_bisection_rejects_interval_without_sign_change():
    """
    Kiểm tra trường hợp khoảng khảo sát không thỏa mãn
    điều kiện hai đầu mút trái dấu.
    """

    with pytest.raises(ValueError):
        bisection(f1, 1, 2)


def test_bisection_warns_when_max_iterations_are_exhausted():
    """
    Kiểm tra cảnh báo khi phương pháp chia đôi hết số bước lặp
    mà chưa đạt sai số yêu cầu.
    """

    with pytest.warns(UserWarning):
        bisection(
            f1, -1, 0,
            eps=1e-15,
            n_max=1
        )


@pytest.mark.parametrize(
    "a, b, eps, n_max",
    [
        (2, 1, 1e-5, 10),    # Khoảng không hợp lệ
        (1, 2, 0, 10),       # Sai số bằng 0
        (1, 2, -1e-5, 10),   # Sai số âm
        (1, 2, 1e-5, -1),    # Số bước lặp âm
    ],
)
def test_bisection_rejects_invalid_arguments(a, b, eps, n_max):
    """
    Kiểm tra phương pháp chia đôi từ chối các tham số
    đầu vào không hợp lệ.
    """

    with pytest.raises(ValueError):
        bisection(f1, a, b, eps, n_max)


# =====================================================
# P3 - KIỂM THỬ PHƯƠNG PHÁP DÂY CUNG
# =====================================================

def test_false_position_solves_first_equation():
    """
    Kiểm tra phương pháp dây cung với phương trình
    2^x - 5x - 3 = 0 trên đoạn [-1, 0].
    """

    # Tìm nghiệm gần đúng
    x, k, err = false_position(
        f1, -1, 0,
        eps=1e-8,
        n_max=200
    )

    # So sánh với nghiệm tham chiếu
    assert math.isclose(x, -0.453996, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert abs(f1(x)) < 1e-5

    # Kiểm tra số bước lặp và sai số
    assert k > 0
    assert err < 1e-8


@pytest.mark.parametrize(
    "a, b, expected",
    CUBIC_ROOTS,
)
def test_false_position_solves_three_cubic_roots(a, b, expected):
    """
    Kiểm tra phương pháp dây cung tìm ba nghiệm
    của phương trình 5x^3 - 20x + 3 = 0.
    """

    # Tìm nghiệm trong khoảng tương ứng
    x, k, err = false_position(
        f2, a, b,
        eps=1e-8,
        n_max=200
    )

    # Kiểm tra nghiệm gần với nghiệm tham chiếu
    assert math.isclose(x, expected, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert abs(f2(x)) < 1e-5

    # Kiểm tra số bước lặp và sai số
    assert k > 0
    assert err < 1e-8


def test_false_position_returns_exact_left_endpoint_root():
    """
    Kiểm tra trường hợp nghiệm nằm tại đầu trái
    bằng cách tạo hàm từ phương trình thứ nhất.
    """

    # Tạo hàm g có nghiệm chính xác tại x = -1
    g = lambda x: f1(x) - f1(-1)

    x, k, err = false_position(
        g, -1, 0,
        eps=1e-5,
        n_max=50
    )

    # Kiểm tra kết quả trả về
    assert x == -1.0
    assert k == 0
    assert err == 0.0


def test_false_position_returns_exact_right_endpoint_root():
    """
    Kiểm tra trường hợp nghiệm nằm tại đầu phải
    bằng cách tạo hàm từ phương trình thứ nhất.
    """

    # Tạo hàm g có nghiệm chính xác tại x = 0
    g = lambda x: f1(x) - f1(0)

    x, k, err = false_position(
        g, -1, 0,
        eps=1e-5,
        n_max=50
    )

    # Kiểm tra kết quả trả về
    assert x == 0.0
    assert k == 0
    assert err == 0.0


def test_false_position_rejects_interval_without_sign_change():
    """
    Kiểm tra trường hợp khoảng khảo sát không có
    hai đầu mút trái dấu.
    """

    with pytest.raises(ValueError):
        false_position(
            f1, 1, 2,
            1e-5, 50
        )


def test_false_position_rejects_non_callable_function():
    """
    Kiểm tra tham số hàm đầu vào phải là một hàm số.
    """

    with pytest.raises(TypeError):
        false_position(123, -1, 0)


def test_false_position_warns_when_max_iterations_are_exhausted():
    """
    Kiểm tra cảnh báo khi phương pháp dây cung hết
    số bước lặp tối đa mà chưa đạt sai số yêu cầu.
    """

    with pytest.warns(RuntimeWarning):
        false_position(
            f1, -1, 0,
            eps=1e-15,
            n_max=1
        )


@pytest.mark.parametrize(
    "a, b, eps, n_max",
    [
        (2, 1, 1e-5, 10),    # Đầu trái lớn hơn đầu phải
        (1, 2, 0, 10),       # Sai số bằng 0
        (1, 2, -1e-5, 10),   # Sai số âm
        (1, 2, 1e-5, 0),     # Số bước lặp bằng 0
        (1, 2, 1e-5, 1.5),   # Số bước lặp không nguyên
    ],
)
def test_false_position_rejects_invalid_arguments(
    a, b, eps, n_max
):
    """
    Kiểm tra phương pháp dây cung từ chối các tham số
    đầu vào không hợp lệ.
    """

    with pytest.raises(ValueError):
        false_position(f1, a, b, eps, n_max)


# =====================================================
# P4 - KIỂM THỬ ĐÁNH GIÁ SAI SỐ
# =====================================================

def test_bisection_steps_matches_theoretical_formula():
    """
    Kiểm tra hàm bisection_steps tính đúng số bước lặp
    theo công thức lý thuyết của phương pháp chia đôi.
    """

    assert bisection_steps(0, 1, 0.25) == 1


def test_bisection_steps_never_returns_negative_number():
    """
    Kiểm tra số bước lặp không âm khi sai số cho phép
    lớn hơn độ dài đoạn khảo sát.
    """

    assert bisection_steps(0, 1, 2) == 0


@pytest.mark.parametrize(
    "a, b, eps",
    [
        (1, 1, 0.1),    # Hai đầu mút bằng nhau
        (2, 1, 0.1),    # Đầu trái lớn hơn đầu phải
        (0, 1, 0),      # Sai số bằng 0
        (0, 1, -0.1),   # Sai số âm
    ],
)
def test_bisection_steps_rejects_invalid_arguments(a, b, eps):
    """
    Kiểm tra hàm bisection_steps từ chối dữ liệu
    đầu vào không hợp lệ.
    """

    with pytest.raises(ValueError):
        bisection_steps(a, b, eps)


def test_false_position_error_matches_formula():
    """
    Kiểm tra kết quả của hàm false_position_error
    với một bộ tham số cụ thể.
    """

    result = false_position_error(1.4, 1.3, 2, 5)

    # So sánh kết quả với giá trị mong đợi
    assert math.isclose(result, 0.15, rel_tol=1e-12)


def test_false_position_error_is_zero_when_iterates_are_equal():
    """
    Kiểm tra sai số bằng 0 khi hai giá trị lặp liên tiếp
    bằng nhau.
    """

    assert false_position_error(1.5, 1.5, 2, 5) == 0.0


def test_false_position_error_rejects_nonpositive_m1():
    """
    Kiểm tra hàm từ chối trường hợp m1 không dương.
    """

    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 0, 5)


def test_false_position_error_rejects_M1_smaller_than_m1():
    """
    Kiểm tra hàm từ chối trường hợp M1 nhỏ hơn m1.
    """

    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 5, 2)


# =====================================================
# KIỂM THỬ CHÉO - SO SÁNH P2 VÀ P3
# =====================================================

@pytest.mark.parametrize(
    "f, a, b, expected",
    [
        # Phương trình thứ nhất
        (f1, -1, 0, -0.453996),

        # Phương trình thứ hai - nghiệm thứ nhất
        (f2, -2.1, -2.0, -2.071157),

        # Phương trình thứ hai - nghiệm thứ hai
        (f2, 0.1, 0.2, 0.150858),

        # Phương trình thứ hai - nghiệm thứ ba
        (f2, 1.9, 2.0, 1.920299),
    ],
)
def test_bisection_and_false_position_agree_on_same_root(
    f, a, b, expected
):
    """
    So sánh kết quả tìm nghiệm của phương pháp chia đôi
    và phương pháp dây cung trên cùng phương trình,
    cùng khoảng khảo sát.
    """

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

    # Kiểm tra nghiệm của phương pháp chia đôi
    assert math.isclose(xb, expected, abs_tol=2e-6)

    # Kiểm tra nghiệm của phương pháp dây cung
    assert math.isclose(xf, expected, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại hai nghiệm gần bằng 0
    assert abs(f(xb)) < 1e-5
    assert abs(f(xf)) < 1e-5

    # Kiểm tra sai số của hai phương pháp
    assert eb < 1e-8
    assert ef < 1e-8

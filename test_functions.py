
import math
import pytest

# =====================================================
# IMPORT CÁC HÀM CẦN KIỂM THỬ
# =====================================================

# P1: Import hàm tìm các khoảng phân ly nghiệm
from isolate_roots import isolate_roots

# P2: Import hàm giải phương trình bằng phương pháp chia đôi
from bisection import bisection

# P3: Import hàm giải phương trình bằng phương pháp dây cung
from false_position import false_position

# P4: Import các hàm dùng để tính toán và đánh giá sai số
from error_estimation import (bisection_steps, false_position_error)

# =====================================================
# KHAI BÁO CÁC PHƯƠNG TRÌNH THEO ĐỀ BÀI
# =====================================================

# Phương trình 1: f1(x) = 2^x - 5x - 3
def f1(x):
    return 2**x - 5*x - 3


# Phương trình 2: f2(x) = 5x^3 - 20x + 3
def f2(x):
    return 5*x**3 - 20*x + 3


# Khai báo ba khoảng chứa ba nghiệm của phương trình thứ hai
# Mỗi phần tử gồm ba giá trị:
# - a: đầu trái của khoảng
# - b: đầu phải của khoảng
# - expected: nghiệm gần đúng đã biết để đối chiếu kết quả
CUBIC_ROOTS = [
    (-2.1, -2.0, -2.071157),
    (0.1, 0.2, 0.150858),
    (1.9, 2.0, 1.920299),
]


# =====================================================
# P1 - KIỂM THỬ HÀM ISOLATE_ROOTS
# =====================================================

def test_isolate_roots_finds_first_equation():
    """Kiểm tra hàm isolate_roots tìm được nghiệm của phương trình 1."""

    # Tìm các khoảng phân ly nghiệm của f1 trên đoạn [-1, 0]
    # Bước quét giữa hai điểm liên tiếp là 0.1
    intervals = isolate_roots(f1, -1, 0, 0.1)

    # Kiểm tra kết quả có đúng một khoảng phân ly nghiệm
    assert len(intervals) == 1

    # Lấy hai đầu mút của khoảng tìm được
    a, b = intervals[0]

    # Kiểm tra khoảng có thứ tự đúng: a phải nhỏ hơn b
    assert a < b

    # Kiểm tra f1 đổi dấu tại hai đầu mút
    # Đây là điều kiện để xác định khoảng phân ly nghiệm
    assert f1(a) * f1(b) < 0

    # Kiểm tra khoảng tìm được có chứa nghiệm gần đúng -0.453996
    assert a <= -0.453996 <= b


def test_isolate_roots_finds_all_three_cubic_roots():
    """Kiểm tra hàm isolate_roots tìm đủ ba nghiệm của phương trình 2."""

    # Tìm các khoảng phân ly nghiệm của f2 trên đoạn [-3, 3]
    intervals = isolate_roots(f2, -3, 3, 0.1)

    # Phương trình có ba nghiệm trong đoạn đang xét
    # Vì vậy kết quả phải có đúng ba khoảng
    assert len(intervals) == 3

    # Khai báo ba nghiệm gần đúng đã biết của phương trình
    expected_roots = [
        -2.071157,
        0.150858,
        1.920299
    ]

    # Kiểm tra từng nghiệm tham chiếu có nằm trong một khoảng tìm được
    # any() trả về True nếu có ít nhất một khoảng chứa nghiệm đó
    for root in expected_roots:
        assert any(a <= root <= b for a, b in intervals)

    # Kiểm tra các khoảng được sắp xếp theo thứ tự tăng dần
    # Đồng thời đảm bảo hai khoảng liên tiếp không chồng lấn
    for i in range(len(intervals) - 1):
        assert intervals[i][1] < intervals[i + 1][0]

    # Kiểm tra điều kiện đổi dấu trên từng khoảng
    for a, b in intervals:
        assert f2(a) * f2(b) < 0


def test_isolate_roots_returns_empty_for_interval_without_root():
    """Kiểm tra hàm trả về danh sách rỗng khi đoạn không chứa nghiệm."""

    # Tìm khoảng phân ly nghiệm của f1 trên đoạn [1, 2]
    intervals = isolate_roots(f1, 1, 2, 0.1)

    # Đoạn này không chứa nghiệm nên kết quả phải là danh sách rỗng
    assert intervals == []


# parametrize cho phép chạy cùng một hàm kiểm thử với nhiều bộ dữ liệu
# Mỗi bộ dữ liệu gồm: a, b và step
@pytest.mark.parametrize(
    "a, b, step",
    [
        (1, 1, 0.1),    # Hai đầu mút bằng nhau
        (2, 1, 0.1),    # Đầu trái lớn hơn đầu phải
        (0, 1, 0),      # Bước quét bằng 0
        (0, 1, -0.1),   # Bước quét âm
    ],
)
def test_isolate_roots_rejects_invalid_arguments(a, b, step):
    """Kiểm tra hàm xử lý các tham số đầu vào không hợp lệ."""

    # pytest.raises kiểm tra xem đoạn lệnh có phát sinh ValueError hay không
    with pytest.raises(ValueError):
        isolate_roots(f1, a, b, step)


# =====================================================
# P2 - KIỂM THỬ PHƯƠNG PHÁP CHIA ĐÔI
# =====================================================

def test_bisection_solves_first_equation():
    """Kiểm tra phương pháp chia đôi giải phương trình 1."""

    # Tìm nghiệm của f1 trên đoạn [-1, 0]
    # eps: sai số cho phép
    # n_max: số bước lặp tối đa
    # Hàm trả về:
    # x: nghiệm gần đúng
    # k: số bước lặp đã thực hiện
    # err: sai số ước lượng
    x, k, err = bisection(
        f1, -1, 0,
        eps=1e-8,
        n_max=200
    )

    # Kiểm tra nghiệm tìm được gần với nghiệm tham chiếu -0.453996
    # abs_tol là sai số tuyệt đối cho phép khi so sánh
    assert math.isclose(x, -0.453996, abs_tol=2e-6)

    # Kiểm tra giá trị tuyệt đối của f1 tại nghiệm đủ nhỏ
    assert abs(f1(x)) < 1e-5

    # Kiểm tra thuật toán đã thực hiện ít nhất một bước lặp
    assert k > 0

    # Kiểm tra sai số ước lượng nhỏ hơn eps
    assert err < 1e-8


# Chạy cùng một bài kiểm thử với ba khoảng chứa ba nghiệm khác nhau
# a, b là hai đầu mút; expected là nghiệm tham chiếu
@pytest.mark.parametrize(
    "a, b, expected",
    CUBIC_ROOTS,
)
def test_bisection_solves_three_cubic_roots(a, b, expected):
    """Kiểm tra phương pháp chia đôi tìm được cả ba nghiệm của phương trình 2."""

    # Tìm nghiệm trong khoảng [a, b]
    x, k, err = bisection(
        f2, a, b,
        eps=1e-8,
        n_max=200
    )

    # So sánh nghiệm tìm được với nghiệm tham chiếu
    assert math.isclose(x, expected, abs_tol=2e-6)

    # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert abs(f2(x)) < 1e-5

    # Kiểm tra số bước lặp và sai số
    assert k > 0
    assert err < 1e-8


def test_bisection_returns_exact_left_endpoint_root():
    """Kiểm tra trường hợp nghiệm nằm chính xác tại đầu trái của khoảng."""

    # Tạo hàm g bằng cách trừ đi f1(-1)
    # Khi x = -1 thì g(-1) = 0
    # Do đó -1 là nghiệm chính xác của hàm g
    g = lambda x: f1(x) - f1(-1)

    # Gọi phương pháp chia đôi trên đoạn [-1, 0]
    x, k, err = bisection(
        g, -1, 0,
        eps=1e-5,
        n_max=50
    )

    # Kiểm tra thuật toán trả về ngay nghiệm ở đầu trái
    assert x == -1.0

    # Không cần lặp vì đã tìm thấy nghiệm chính xác
    assert k == 0

    # Sai số bằng 0 vì nghiệm đã chính xác
    assert err == 0.0


def test_bisection_returns_exact_right_endpoint_root():
    """Kiểm tra trường hợp nghiệm nằm chính xác tại đầu phải của khoảng."""

    # Tạo hàm g sao cho g(0) = 0
    # Do đó x = 0 là nghiệm chính xác
    g = lambda x: f1(x) - f1(0)

    # Gọi phương pháp chia đôi trên đoạn [-1, 0]
    x, k, err = bisection(
        g, -1, 0,
        eps=1e-5,
        n_max=50
    )

    # Kiểm tra thuật toán trả về nghiệm ở đầu phải
    assert x == 0.0

    # Không cần thực hiện bước lặp nào
    assert k == 0

    # Sai số bằng 0
    assert err == 0.0


def test_bisection_rejects_interval_without_sign_change():
    """Kiểm tra phương pháp chia đôi từ chối khoảng không có đổi dấu."""

    # Trên đoạn [1, 2], f1 không đổi dấu
    # Vì vậy thuật toán phải phát sinh ValueError
    with pytest.raises(ValueError):
        bisection(
            f1, 1, 2,
            eps=1e-5,
            n_max=50
        )


def test_bisection_warns_when_max_iterations_are_exhausted():
    """Kiểm tra cảnh báo khi phương pháp chia đôi hết số bước lặp."""

    # Chọn eps rất nhỏ để yêu cầu độ chính xác cao
    # Nhưng chỉ cho phép một bước lặp
    # Vì vậy thuật toán chưa thể đạt độ chính xác yêu cầu
    with pytest.warns(UserWarning):
        bisection(
            f1, -1, 0,
            eps=1e-15,
            n_max=1
        )


# Kiểm tra nhiều trường hợp tham số đầu vào không hợp lệ
@pytest.mark.parametrize(
    "a, b, eps, n_max",
    [
        (2, 1, 1e-5, 10),    # a lớn hơn b
        (1, 2, 0, 10),       # eps bằng 0
        (1, 2, -1e-5, 10),   # eps âm
        (1, 2, 1e-5, -1),    # Số bước lặp âm
    ],
)
def test_bisection_rejects_invalid_arguments(a, b, eps, n_max):
    """Kiểm tra phương pháp chia đôi từ chối các tham số không hợp lệ."""

    # Mỗi bộ dữ liệu trên phải khiến hàm phát sinh ValueError
    with pytest.raises(ValueError):
        bisection(f1, a, b, eps, n_max)


# =====================================================
# P3 - KIỂM THỬ PHƯƠNG PHÁP DÂY CUNG
# =====================================================

def test_false_position_solves_first_equation():
    """Kiểm tra phương pháp dây cung giải phương trình 1."""

    # Tìm nghiệm của f1 trên đoạn [-1, 0]
    # eps là sai số cho phép, n_max là số bước lặp tối đa
    # Kết quả gồm nghiệm gần đúng, số bước lặp và sai số
    x, k, err = false_position(
        f1, -1, 0,
        eps=1e-8,
        n_max=200
    )

    # Kiểm tra nghiệm tìm được gần với nghiệm tham chiếu
    assert math.isclose(x, -0.453996, abs_tol=2e-6)

    # Kiểm tra f1(x) gần bằng 0
    assert abs(f1(x)) < 1e-5

    # Kiểm tra thuật toán đã thực hiện ít nhất một bước lặp
    assert k > 0

    # Kiểm tra sai số nhỏ hơn eps
    assert err < 1e-8


# Kiểm tra phương pháp dây cung trên ba khoảng chứa ba nghiệm
@pytest.mark.parametrize(
    "a, b, expected",
    CUBIC_ROOTS,
)
def test_false_position_solves_three_cubic_roots(a, b, expected):
    """Kiểm tra phương pháp dây cung tìm được ba nghiệm của phương trình 2."""

    # Tìm nghiệm trong khoảng [a, b]
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
    """Kiểm tra trường hợp nghiệm chính xác tại đầu trái."""

    # Tạo hàm g có nghiệm tại x = -1
    g = lambda x: f1(x) - f1(-1)

    # Gọi phương pháp dây cung
    x, k, err = false_position(
        g, -1, 0,
        eps=1e-5,
        n_max=50
    )

    # Kiểm tra nghiệm trả về đúng bằng đầu trái
    assert x == -1.0

    # Thuật toán trả về ngay nên số bước lặp bằng 0
    assert k == 0

    # Sai số bằng 0
    assert err == 0.0


def test_false_position_returns_exact_right_endpoint_root():
    """Kiểm tra trường hợp nghiệm chính xác tại đầu phải."""

    # Tạo hàm g có nghiệm tại x = 0
    g = lambda x: f1(x) - f1(0)

    # Gọi phương pháp dây cung
    x, k, err = false_position(
        g, -1, 0,
        eps=1e-5,
        n_max=50
    )

    # Kiểm tra nghiệm trả về đúng bằng đầu phải
    assert x == 0.0

    # Thuật toán không cần lặp
    assert k == 0

    # Sai số bằng 0
    assert err == 0.0


def test_false_position_rejects_interval_without_sign_change():
    """Kiểm tra phương pháp dây cung từ chối khoảng không có đổi dấu."""

    # Đoạn [1, 2] không có điều kiện đổi dấu của f1
    # Hàm phải phát sinh ValueError
    with pytest.raises(ValueError):
        false_position(
            f1, 1, 2,
            eps=1e-5,
            n_max=50
        )


def test_false_position_rejects_non_callable_function():
    """Kiểm tra đầu vào của phương pháp dây cung phải là một hàm."""

    # Truyền số 123 thay vì một hàm số
    # Vì 123 không thể gọi như một hàm nên phải phát sinh TypeError
    with pytest.raises(TypeError):
        false_position(
            123, -1, 0,
            eps=1e-5,
            n_max=50
        )


def test_false_position_warns_when_max_iterations_are_exhausted():
    """Kiểm tra cảnh báo khi phương pháp dây cung hết số bước lặp."""

    # Yêu cầu sai số rất nhỏ nhưng chỉ cho phép một bước lặp
    # Kiểm tra thuật toán phát sinh cảnh báo RuntimeWarning
    with pytest.warns(RuntimeWarning):
        false_position(
            f1, -1, 0,
            eps=1e-15,
            n_max=1
        )


# Kiểm tra các trường hợp tham số không hợp lệ của phương pháp dây cung
@pytest.mark.parametrize(
    "a, b, eps, n_max",
    [
        (2, 1, 1e-5, 10),    # a lớn hơn b
        (1, 2, 0, 10),       # eps bằng 0
        (1, 2, -1e-5, 10),   # eps âm
        (1, 2, 1e-5, 0),     # Số bước lặp bằng 0
        (1, 2, 1e-5, 1.5),   # Số bước lặp không phải số nguyên
    ],
)
def test_false_position_rejects_invalid_arguments(a, b, eps, n_max):
    """Kiểm tra phương pháp dây cung từ chối tham số không hợp lệ."""

    # Mỗi bộ tham số trên phải phát sinh ValueError
    with pytest.raises(ValueError):
        false_position(f1, a, b, eps, n_max)


# =====================================================
# P4 - KIỂM THỬ ĐÁNH GIÁ SAI SỐ
# =====================================================

def test_bisection_steps_matches_theoretical_formula():
    """Kiểm tra hàm tính số bước lặp lý thuyết của phương pháp chia đôi."""

    # Tính số bước lặp cần thiết với:
    # - Khoảng ban đầu [0, 1]
    # - Sai số cho phép eps = 0.25
    # Kết quả theo công thức được kỳ vọng bằng 1
    assert bisection_steps(0, 1, 0.25) == 1


def test_bisection_steps_never_returns_negative_number():
    """Kiểm tra số bước lặp tính được không bao giờ âm."""

    # Độ dài đoạn ban đầu bằng 1
    # Sai số cho phép bằng 2, lớn hơn độ dài đoạn
    # Do đó không cần thực hiện bước lặp nào
    assert bisection_steps(0, 1, 2) == 0


# Kiểm tra các tham số không hợp lệ của hàm bisection_steps
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
    """Kiểm tra hàm bisection_steps từ chối tham số không hợp lệ."""

    # Các trường hợp trên phải phát sinh ValueError
    with pytest.raises(ValueError):
        bisection_steps(a, b, eps)


def test_false_position_error_matches_formula():
    """Kiểm tra kết quả tính sai số của phương pháp dây cung theo công thức."""

    # Tính sai số với hai giá trị lặp liên tiếp là 1.4 và 1.3
    # Hai hằng số sử dụng trong công thức là m1 = 2 và M1 = 5
    result = false_position_error(1.4, 1.3, 2, 5)

    # Kiểm tra kết quả tính được bằng 0.15
    # rel_tol là sai số tương đối cho phép khi so sánh số thực
    assert math.isclose(result, 0.15, rel_tol=1e-12)


def test_false_position_error_is_zero_when_iterates_are_equal():
    """Kiểm tra sai số bằng 0 khi hai giá trị lặp liên tiếp bằng nhau."""

    # Hai giá trị lặp đều bằng 1.5 nên độ chênh lệch bằng 0
    # Vì vậy sai số tính được phải bằng 0
    assert false_position_error(1.5, 1.5, 2, 5) == 0.0


def test_false_position_error_rejects_nonpositive_m1():
    """Kiểm tra hàm từ chối trường hợp m1 không dương."""

    # m1 = 0 không thỏa mãn điều kiện của công thức sai số
    # Do đó hàm phải phát sinh ValueError
    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 0, 5)


def test_false_position_error_rejects_M1_smaller_than_m1():
    """Kiểm tra hàm từ chối trường hợp M1 nhỏ hơn m1."""

    # Trường hợp này m1 = 5 và M1 = 2
    # M1 nhỏ hơn m1 nên dữ liệu không hợp lệ
    # Hàm phải phát sinh ValueError
    with pytest.raises(ValueError):
        false_position_error(1.4, 1.3, 5, 2)

import math
import warnings
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

# 1. TÌM KHOẢNG PHÂN LY NGHIỆM
def isolate_roots(f, a, b, step) -> list[tuple[float, float]]:
    """
    Thuật toán tìm các khoảng cách ly (phân ly) nghiệm bằng phương pháp quét dấu (Incremental Search).
    Require (Đầu vào):
        f: Hàm số f(x) liên tục cần tìm khoảng cách ly nghiệm (Callable[[float], float]).
        a: Biên trái của đoạn xét [a, b] (float).
        b: Biên phải của đoạn xét [a, b] (float), với a < b.
        step: Bước quét h > 0 (float).
    Ensure (Đầu ra): Danh sách các tuple (x_i, x_{i+1}) chứa các khoảng cách ly nghiệm thoả mãn f(x_i) * f(x_{i+1}) <= 0.
    Xử lý tình huống biên: Báo lỗi ValueError nếu a >= b hoặc step <= 0.
    """
    # Kiểm tra điều kiện đầu vào
    if a >= b:
        raise ValueError("Lỗi biên: Biên trái 'a' phải nhỏ hơn biên phải 'b'.")
    if step <= 0:
        raise ValueError("Lỗi biên: Bước quét 'step' phải là số thực dương (> 0).")
    intervals = []
    x_curr = float(a)
    while x_curr < b:
        x_next = min(x_curr + step, float(b)) # Đảm bảo không vượt quá biên phải b
        y_curr = f(x_curr)
        y_next = f(x_next)
        if y_curr * y_next < 0: # Kiểm tra điều kiện đổi dấu
            intervals.append((x_curr, x_next))
        elif y_curr == 0:
            intervals.append((x_curr, x_curr))
        x_curr = x_next
    if f(b) == 0: # Kiểm tra b để tránh bỏ sót nghiệm ở biên phải
        val_b = float(b)
        if not intervals or intervals[-1] != (val_b, val_b):
            intervals.append((val_b, val_b))
    return intervals # Trả về danh sách các khoảng cách ly nghiệm
def check_isolate_roots(f, a, b, step=0.01) -> bool:
    """
    Kiểm tra (a, b) có phải là khoảng phân ly nghiệm của f(x) hay không.
    Điều kiện:
      1. f(a) * f(b) < 0 (f(x) có nghiệm trong khoảng)
      2. f'(x) không đổi dấu trên (a, b) (hàm số đơn điệu, đảm bảo nghiệm duy nhất)
    """
    fa, fb = f(a), f(b)
    # Kiểm tra điều kiện đổi dấu
    if fa * fb >= 0:
        print(f"[-] LOẠI ({a}, {b}): f({a}) * f({b}) >= 0 (Không đảm bảo có nghiệm).")
        return False
    # Kiểm tra điều kiện nghiệm duy nhất (f'(x) không đổi dấu)
    h = 1e-5
    xs = np.arange(a, b, step)
    df_vals = [(f(x + h) - f(x - h)) / (2 * h) for x in xs] # Đạo hàm bằng ct sai phân trung tâm
    # Kiểm tra xem đạo hàm có luôn dương hoặc luôn âm trên toàn khoảng không
    df_duong = all(d > 0 for d in df_vals)
    df_am = all(d < 0 for d in df_vals)
    if df_duong or df_am:
        print(f"[+] HỢP LỆ: ({a}, {b}) là khoảng phân ly nghiệm.")
        print(f"    - Có nghiệm: f({a}) = {fa:.6f}, f({b}) = {fb:.6f} (Trái dấu)")
        print(f"    - Nghiệm duy nhất: f'(x) không đổi dấu trên ({a}, {b}).")
        return True
    else:
        print(f"[!] CẢNH BÁO: f(x) đổi dấu trên ({a}, {b}) nhưng f'(x) không đơn điệu (có thể có >1 nghiệm).")
        return False

# 2. PHƯƠNG PHÁP CHIA ĐÔI & ĐÁNH GIÁ SAI SỐ
# Đánh giá sai số (số bước lặp tối thiểu)
def bisection_steps(a, b, eps) -> int:
    """
    Tính số bước lặp lý thuyết tối thiểu của phương pháp chia đôi để đạt sai số eps:
    Thuật toán: Đánh giá số bước lặp Chia đôi (Bisection Step Estimation)
    Công thức sai số: e_n = (b - a) / 2^(n + 1) <= eps
    => 2^(n + 1) >= (b - a) / eps => n >= log2((b - a) / eps) - 1
    Require (Đầu vào):
        a (float): Biên trái khoảng cách ly nghiệm (a < b)
        b (float): Biên phải khoảng cách ly nghiệm
        eps (float): Độ chính xác yêu cầu (eps > 0)
    Ensure (Đầu ra):
        int: Số bước lặp tối thiểu n = max(0, ceil(log2((b - a) / eps)) - 1)
    """
    # Kiểm tra điều kiện đầu vào
    if a >= b:
        raise ValueError("Khoảng cách ly không hợp lệ: yêu cầu b > a")
    if eps <= 0:
        raise ValueError("Sai số eps phải lớn hơn 0")
    n = np.ceil(np.log2((b - a) / eps)) - 1  # Số bước lặp tối thiểu, làm tròn lên np.ceil
    return max(0, int(n)) # Trả kết quả, n < 0 thì trả về 0
# Phương pháp chia đôi
def bisection(f, a, b, eps, n_max) -> tuple[float, int, float]:
    """
    Giải phương trình f(x) = 0 bằng phương pháp chia đôi (theo Algorithm 4 trong bài giảng)
    Require (Đầu vào):
        f     : hàm số liên tục, nhận một số thực, trả về một số thực
        a, b  : hai đầu khoảng phân ly nghiệm, a < b và f(a)f(b) < 0
        eps   : sai số cho trước, eps > 0
        n_max : số bước lặp tối đa cho trước, n_max >= 0
    Ensure (Đầu ra):
        Trả về bộ (x, k, err):
        x   : nghiệm gần đúng ở bước lặp cuối cùng
        k   : số bước lặp đã thực hiện (biến i trong thuật toán)
        err : sai số tại bước lặp cuối, tính bằng (b - a)/2 của khoảng hiện tại
    Điều kiện dừng: err <= eps, hoặc k đạt n_max + 1, hoặc f(a)f(x) = 0
    Lưu ý: Nếu a >= b, eps <= 0, n_max < 0, hoặc f(a)f(b) > 0 thì (a; b) không phải khoảng phân ly nghiệm
           Nếu hết n_max bước mà chưa đạt sai số eps thì cảnh báo người dùng
    Algorithm: Phương pháp chia đôi (Algorithm 4 trong giáo trình).
    """
    # Kiểm tra điều kiện đầu vào
    if not a < b:
        raise ValueError("Cần a < b.")
    if eps <= 0:
        raise ValueError("Sai số eps phải dương.")
    if n_max < 0:
        raise ValueError("n_max phải không âm.")
    fa = f(a)
    fb = f(b)
    # Trường hợp đầu mút là nghiệm
    if fa == 0: return float(a), 0, 0.0
    if fb == 0: return float(b), 0, 0.0
    if fa * fb > 0: # Kiểm tra điều kiện đổi dấu
        raise ValueError("f(a)f(b) > 0: (a; b) không phải khoảng phân ly nghiệm.")
    i = 0
    err = (b - a) / 2 # dòng 1 - Thuật toán 4
    while err > eps and i < n_max + 1: # dòng 2 - Thuật toán 4
        x = (a + b) / 2 # dòng 3 - Thuật toán 4
        fx = f(x) # dòng 4 - Thuật toán 4
        if fa * fx == 0: # dòng 5 - Thuật toán 4
            return float(x), i + 1, 0.0 # dòng 6 - Thuật toán 4
        elif fa * fx > 0: # dòng 7 - Thuật toán 4
            a = x # dòng 8 - Thuật toán 4
            fa = fx
            err = (b - a) / 2 # dòng 9 - Thuật toán 4
        else: # dòng 10 - Thuật toán 4
            b = x # dòng 11 - Thuật toán 4
            fb = fx
            err = (b - a) / 2 # dòng 12 - Thuật toán 4
        i = i + 1 # dòng 13,14 - Thuật toán 4: end if và gán i <- i + 1
    # dòng 15 - Thuật toán 4: end while
    if err > eps:
        warnings.warn(f"Hết n_max={n_max} bước nhưng chưa đạt sai số eps={eps} (err={err}).", UserWarning)
    x = (a + b) / 2
    err = (b - a) / 2
    return float(x), i, float(err) # dòng 16 - Thuật toán 4

# 3. PHƯƠNG PHÁP DÂY CUNG & ĐÁNH GIÁ SAI SỐ
# Đánh giá sai số
def false_position_error(x_k, x_prev, m1, M1) -> float:
    """
    Thuật toán: Đánh giá sai số Dây cung (False Position Error)
    Công thức sai số: |x_k - x_star| <= ((M1 - m1) / m1) * |x_k - x_prev|
    Require(Đầu vào):
        x_k (float): Giá trị nghiệm xấp xỉ ở bước lặp thứ k
        x_prev (float): Giá trị nghiệm xấp xỉ ở bước lặp thứ k-1
        m1 (float): Cận dưới của |f'(x)| trên khoảng cách ly (m1 > 0)
        M1 (float): Cận trên của |f'(x)| trên khoảng cách ly (M1 >= m1)
    Ensure (Đầu ra):
        float: Giá trị cận trên sai số tuyệt đối tại bước lặp k
    """
    # Kiểm tra điều kiện đầu vào m1, M1
    if m1 <= 0:
        raise ValueError("Cận m1 phải lớn hơn 0")
    if M1 < m1:
        raise ValueError("Cận M1 phải lớn hơn hoặc bằng m1")
    err = ((M1 - m1) / m1) * abs(x_k - x_prev)
    return float(err)
# Phương pháp dây cung
def false_position(f, a, b, eps, n_max) -> tuple[float, int, float]:
    """
    Tìm nghiệm gần đúng của f(x) = 0 bằng phương pháp dây cung.
    Require:
        f: Hàm số liên tục trên [a, b].
        a, b: Hai đầu mút, a < b.
        eps: Sai số cho phép, eps > 0.
        n_max: Số bước lặp tối đa, n_max là số nguyên dương.
        Hai đầu mút trái dấu, trừ khi một đầu mút là nghiệm.
    Ensure:
        Trả về (xn, i, err):
            xn: Nghiệm gần đúng.
            i: Số bước lặp đã thực hiện.
            err: Sai số giữa hai lần lặp liên tiếp.
        Cảnh báo nếu hết n_max bước mà chưa đạt eps.
    Algorithm: Phương pháp dây cung (Algorithm 6 trong giáo trình).
    """
    # Kiểm tra điều kiện đầu vào
    if not callable(f):
        raise TypeError("f phải là một hàm số.")
    if a >= b:
        raise ValueError("Phải có a < b.")
    if eps <= 0:
        raise ValueError("eps phải lớn hơn 0.")
    if not isinstance(n_max, int) or n_max < 1:
        raise ValueError("n_max phải là số nguyên dương.")
    fa = f(a)
    fb = f(b)
    # Trường hợp đầu mút là nghiệm
    if fa == 0: return float(a), 0, 0.0
    if fb == 0: return float(b), 0, 0.0
    if fa * fb > 0: # Kiểm tra điều kiện đổi dấu
        raise ValueError("Khoảng [a, b] không thỏa mãn f(a)*f(b) < 0.")
    # Chọn điểm xuất phát
    if fa < 0: # dòng 3,6 - Thuật toán 6
        x0 = a
    else:
        x0 = b
    x_prev = x0
    for i in range(1, n_max + 1): # dòng 1 - Thuật toán 6
        fx_prev = f(x_prev)
        if fa < 0: # dòng 2 - Thuật toán 6
            # dòng 4 - Thuật toán 6: Tính xn <- xn-1 - f(xn-1)*(xn-1 - b) / (f(xn-1) - f(b))
            mau_so = fx_prev - fb
            if mau_so == 0: raise ZeroDivisionError("Mẫu số bằng 0")
            xn = x_prev - fx_prev * (x_prev - b) / mau_so
        else: # dòng 5 - Thuật toán 6
            # dòng 7 - Thuật toán 6: Tính xn <- xn-1 - f(xn-1)*(xn-1 - a) / (f(xn-1) - f(a))
            mau_so = fx_prev - fa
            if mau_so == 0: raise ZeroDivisionError("Mẫu số bằng 0")
            xn = x_prev - fx_prev * (x_prev - a) / mau_so
        # dòng 8 - Thuật toán 6: end if
        err = abs(xn - x_prev) # dòng 9 - Thuật toán 6
        if err < eps: # dòng 10,11 - Thuật toán 6
            return float(xn), i, float(err)
        # dòng 12 - Thuật toán 6: end if
        x_prev = xn
        # dòng 13 - Thuật toán 6: end for
    warnings.warn("Chưa đạt sai số yêu cầu sau số bước lặp tối đa", RuntimeWarning)
    return float(xn), i, float(err)

# 4. BỘ KIỂM THỬ
print ("BỘ KIỂM THỬ:")
def f1(x): return 2**x - 5*x - 3
def f2(x): return 5*x**3 - 20*x + 3
# Danh sách các khoảng chứa nghiệm và nghiệm tham chiếu gần đúng của f2
CUBIC_ROOTS = [(-2.1, -2.0, -2.071157),(0.1, 0.2, 0.150858),(1.9, 2.0, 1.920299)]

# KIỂM THỬ TÌM KHOẢNG PHÂN LY NGHIỆM
def test_khoang_phan_ly_f1():
    intervals = isolate_roots(f1, -1, 0, 0.1)
    assert len(intervals) == 1
    a, b = intervals[0]
    assert a < b
    assert f1(a) * f1(b) < 0
    assert a <= -0.453996 <= b
def test_khoang_phan_ly_f2():
    intervals = isolate_roots(f2, -3, 3, 0.1)
    assert len(intervals) == 3
    expected_roots = [-2.071157,0.150858,1.920299]
    for root in expected_roots: # Kiểm tra nghiệm có thuộc khoảng phân ly
        assert any(a <= root <= b for a, b in intervals)
    for i in range(len(intervals) - 1): # Kiểm tra các khoảng phân ly không chồng lấn nhau
        assert intervals[i][1] < intervals[i + 1][0]
    for a, b in intervals: # Kiểm tra điều kiện nghiệm
        assert f2(a) * f2(b) < 0
def test_f1_khong_co_nghiem(): # Kiểm tra f1 không có nghiệm trong (1,2)
    intervals = isolate_roots(f1, 1, 2, 0.1)
    assert intervals == []
def test_f2_khong_co_nghiem(): # Kiểm tra f1 không có nghiệm trong (3,4)
    assert isolate_roots(f2, 3, 4, 0.1) == []

@pytest.mark.parametrize("function", [f1, f2])
@pytest.mark.parametrize(
    "a, b, step",
    [
        (1, 1, 0.1),
        (2, 1, 0.1),
        (0, 1, 0),
        (0, 1, -0.1),
    ],
)
def test_dau_vao_nghiem(function, a, b, step): # Kiểm tra đầu vào không hợp lệ f1, f2
    with pytest.raises(ValueError):
        isolate_roots(function, a, b, step)

# KIỂM THỬ PHƯƠNG PHÁP CHIA ĐÔI
def test_bi_f1(): # Kiểm tra pp chia đôi với f1
    x, k, err = bisection(f1, -1, 0, eps=1e-8, n_max=200)
    assert math.isclose(x, -0.453996, abs_tol=2e-6) # So sánh nghiệm gần đúng với nghiệm tham chiếu
    assert abs(f1(x)) < 1e-5  # Kiểm tra giá trị hàm tại nghiệm gần bằng 0
    assert k > 0
    assert err < 1e-8
@pytest.mark.parametrize("a, b, expected",CUBIC_ROOTS,)
def test_bi_f2(a, b, expected): # Kiểm tra pp chia đôi với f2
    x, k, err = bisection(f2, a, b, eps=1e-8, n_max=200)
    assert math.isclose(x, expected, abs_tol=2e-6)
    assert abs(f2(x)) < 1e-5
    assert k > 0
    assert err < 1e-8
def test_bi_nghiem_tai_a_f1(): # Kiểm tra nghiệm tại a với f1
    g = lambda x: f1(x) - f1(-1) # Tạo hàm g có nghiệm chính xác tại x = -1
    x, k, err = bisection(g, -1, 0, eps=1e-8, n_max=200)
    assert x == -1.0
    assert k == 0
    assert err == 0.0
def test_bi_nghiem_tai_b_f1(): # Kiểm tra nghiệm tại b với f1
    g = lambda x: f1(x) - f1(0) # Tạo hàm g có nghiệm chính xác tại x = 0
    x, k, err = bisection(g, -1, 0, eps=1e-8, n_max=200)
    assert x == 0.0
    assert k == 0
    assert err == 0.0
def test_bi_nghiem_tai_a_f2(): # Kiểm tra nghiệm tại a với f2
    a, b = -2.1, -2.0
    g = lambda x: f2(x) - f2(a) # Tạo hàm g có nghiệm chính xác tại x = a
    x, k, err = bisection(g, a, b, eps=1e-5, n_max=50)
    assert x == a
    assert k == 0
    assert err == 0.0
def test_bi_nghiem_tai_b_f2(): # Kiểm tra nghiệm tại b với f2
    a, b = -2.1, -2.0
    g = lambda x: f2(x) - f2(b) # Tạo hàm g có nghiệm chính xác tại x = a
    x, k, err = bisection(g, a, b, eps=1e-5, n_max=50)
    assert x == b
    assert k == 0
    assert err == 0.0

@pytest.mark.parametrize("function", [f1, f2])
def test_bi_khong_trai_dau(function): # Kiểm tra điều kiện trái dấu
    with pytest.raises(ValueError):
        bisection(function, 3, 4, eps=1e-5, n_max=50)

@pytest.mark.parametrize("function", [f1, f2])
@pytest.mark.parametrize(
    "a, b, eps, n_max",
    [
        (2, 1, 1e-5, 10),
        (1, 2, 0, 10),
        (1, 2, -1e-5, 10),
        (1, 2, 1e-5, -1),
    ],
)
def test_bi_dau_vao(function, a, b, eps, n_max): # Kiểm tra đầu vào không hợp lệ
    with pytest.raises(ValueError):
        bisection(function, a, b, eps, n_max)
def test_bi_warn_het_buoc_lap_f1(): # Kiểm tra cảnh báo khi hết n_max chưa đạt sai số
    with pytest.warns(UserWarning):
        bisection(f1, -1, 0, eps=1e-15, n_max=1)

# KIỂM THỬ PHƯƠNG PHÁP DÂY CUNG
def test_fp_f1(): # Kiểm tra pp dây cung với f1 trên đoạn [-1, 0]
    x, k, err = false_position(f1, -1, 0, eps=1e-8, n_max=200)
    assert math.isclose(x, -0.453996, abs_tol=2e-6)
    assert abs(f1(x)) < 1e-5
    assert k > 0
    assert err < 1e-8
@pytest.mark.parametrize("a, b, expected", CUBIC_ROOTS,)
def test_fp_f2(a, b, expected): # Kiểm tra pp dây cung với f2
    x, k, err = false_position(f2, a, b, eps=1e-8, n_max=200)
    assert math.isclose(x, expected, abs_tol=2e-6)
    assert abs(f2(x)) < 1e-5
    assert k > 0
    assert err < 1e-8
def test_fp_nghiem_tai_a_f1(): # Kiểm tra nghiệm tại a với f1
    g = lambda x: f1(x) - f1(-1) # Tạo hàm g có nghiệm chính xác tại x = -1
    x, k, err = false_position(g, -1, 0, eps=1e-5, n_max=50)
    assert x == -1.0
    assert k == 0
    assert err == 0.0
def test_fp_nghiem_tai_b_f1(): # Kiểm tra nghiệm tại b với f1
    g = lambda x: f1(x) - f1(0) # Tạo hàm g có nghiệm chính xác tại x = 0
    x, k, err = false_position(g, -1, 0, eps=1e-5, n_max=50)
    assert x == 0.0
    assert k == 0
    assert err == 0.0
def test_fp_nghiem_tai_a_f2(): # Kiểm tra nghiệm tại a với f2
    a, b = -2.1, -2.0
    g = lambda x: f2(x) - f2(a) # Tạo hàm g có nghiệm chính xác tại x = a
    x, k, err = false_position(g, a, b, eps=1e-5, n_max=50)
    assert x == a
    assert k == 0
    assert err == 0.0
def test_fp_nghiem_tai_b_f2(): # Kiểm tra nghiệm tại b với f2
    a, b = -2.1, -2.0
    g = lambda x: f2(x) - f2(b) # Tạo hàm g có nghiệm chính xác tại x = b
    x, k, err = false_position(g, a, b, eps=1e-5, n_max=50)
    assert x == b
    assert k == 0
    assert err == 0.0

@pytest.mark.parametrize("function", [f1, f2])
def test_fp_khoang_khong_thoa_man(function): # Kiểm tra trường hợp khoảng khảo sát không thỏa mãn điều kiện trái dấu
    with pytest.raises(ValueError): false_position(function, 3, 4, 1e-5, 50)
def test_fp_kp_ham_so(): # Kiểm tra f kp hàm số
    with pytest.raises(TypeError): false_position(123, -1, 0, 1e-5, 50)
def test_fp_warn_het_buoc_lap_f1(): # Kiểm tra cảnh báo khi dây cung hết bước lặp mà chưa đạt sai số
    with pytest.warns(RuntimeWarning): false_position(f1, -1, 0, eps=1e-15, n_max=1)

@pytest.mark.parametrize("function", [f1, f2])
@pytest.mark.parametrize(
    "a, b, eps, n_max",
    [
        (2, 1, 1e-5, 10),
        (1, 2, 0, 10),
        (1, 2, -1e-5, 10),
        (1, 2, 1e-5, 0),
        (1, 2, 1e-5, 1.5),
    ],
)
def test_fp_dau_vao(function, a, b, eps, n_max): # Kiểm tra đầu vào không hợp lệ
    with pytest.raises(ValueError): false_position(function, a, b, eps, n_max)

# KIỂM THỬ ĐÁNH GIÁ SAI SỐ
def test_bis_buoc_lap(): # Kiểm tra tính số bước lặp pp chia đôi
    assert bisection_steps(0, 1, 0.25) == 1
def test_bis_buoc_lap_0(): # Kiểm tra số bước lặp không âm khi sai số lớn hơn khoảng khảo sát
    assert bisection_steps(0, 1, 2) == 0
@pytest.mark.parametrize(
    "a, b, eps",
    [
        (1, 1, 0.1),
        (2, 1, 0.1),
        (0, 1, 0),
        (0, 1, -0.1),
    ],
)
def test_bis_dau_vao(a, b, eps): # Kiểm tra đầu vào không hợp lệ
    with pytest.raises(ValueError): bisection_steps(a, b, eps)
def test_fp_error(): # Kiểm tra kết quả của hàm false_position_error với x_k, x_prev, m1, M1
    result = false_position_error(1.4, 1.3, 2, 5)
    assert math.isclose(result, 0.15, rel_tol=1e-12)
def test_fp_error_0(): # Kiểm tra sai số bằng 0 khi hai giá trị lặp liên tiếp bằng nhau
    assert false_position_error(1.5, 1.5, 2, 5) == 0.0
def test_m1_khong_duong(): # Kiểm tra hàm từ chối m1 không dương
    with pytest.raises(ValueError): false_position_error(1.4, 1.3, 0, 5)
def test_m1_nho_hon_M1(): # Kiểm tra cảnh báo trường hợp M1 nhỏ hơn m1
    with pytest.raises(ValueError): false_position_error(1.4, 1.3, 5, 2)

!pip install ipytest
import ipytest
# Chạy lại tất cả các bài kiểm thử để kiểm tra xem lỗi đã được khắc phục hoàn toàn chưa
ipytest.run("-v")

print("\nTìm nghiệm kiểm thử cho các hàm số f(x) = 2^x - 5x - 3 và f2(x) = 5x^3 - 20x + 3")
if __name__ == "__main__":
    def f(x): return 2**x - 5*x - 3
    def f2(x): return 5*x**3 - 20*x + 3
    print("TÌM NGHIỆM CHO HÀM f1(x) = 2^x - 5x - 3 ")
    intervals_f = isolate_roots(f, -1.0, 0.0, step=0.1)
    if not intervals_f:
        print("Không tìm thấy khoảng chứa nghiệm nào cho f(x).")
    else:
        for idx, interval in enumerate(intervals_f, 1):
            a, b = interval
            x, k, err = bisection(f, a, b, eps=1e-8, n_max=50)
            print(f"  Khoảng {idx}: [{a:.2f}, {b:.2f}] -> Nghiệm xấp xỉ: {x:.6f} (sau {k} bước lặp, sai số {err:.2e})")
    print("TÌM NGHIỆM CHO HÀM f2(x) = 5x^3 - 20x + 3 ")
    intervals_f2 = isolate_roots(f2, -3.0, 3.0, step=0.1)
    if not intervals_f2:
        print("Không tìm thấy khoảng chứa nghiệm nào cho f2(x).")
    else:
        for idx, interval in enumerate(intervals_f2, 1):
            a, b = interval
            x, k, err = bisection(f2, a, b, eps=1e-8, n_max=50)
            print(f"  Khoảng {idx}: [{a:.2f}, {b:.2f}] -> Nghiệm xấp xỉ: {x:.6f} (sau {k} bước lặp, sai số {err:.2e})")

# 5. BÀI ỨNG DỤNG
# Đánh giá sai số (số bước lặp tối thiểu): def bisection_steps(a, b, eps)
# Phương pháp chia đôi, có bảng tính
def bisection_history(f, a, b, eps=1e-5, n_max=50):
    history = []
    for i in range(0, n_max + 1):
        x = (a + b) / 2.0
        fx = f(x)
        err = (b - a) / 2.0
        history.append({
            "n": i,
            "a": round(a, 6),
            "b": round(b, 6),
            "x": round(x, 6),
            "f(x)": round(fx, 6),
            "Sai số": round(err, 6)
        })
        if err <= eps or fx == 0:
            return x, i, err, pd.DataFrame(history)
        if f(a) * fx < 0:
            b = x
        else:
            a = x
    return float(x), i, float(err), pd.DataFrame(history)
# Phương pháp dây cung, có bảng tính
def false_position_history(f, a, b, eps=1e-5, n_max=50):
    fa, fb = f(a), f(b)
    h = 1e-5
    d2f_a = (f(a + h) - 2 * fa + f(a - h)) / (h**2)
    d2f_b = (f(b + h) - 2 * fb + f(b - h)) / (h**2)
    # Xác định điểm Fourier x0 và mút cố định d
    if fa * d2f_a > 0:
        x0 = b
        x = a
        print(f"[+] Điểm Fourier là x = a = {a}")
        print(f"    - f({a}) = {fa:.6f}")
        print(f"    - f''({a}) ≈ {d2f_a:.6f} => f(a)*f''(a) > 0 \n")
    elif fb * d2f_b > 0:
        x0 = a
        x = b
        print(f"[+] Điểm Fourier là x = b = {b}")
        print(f"    - f({b}) = {fb:.6f}")
        print(f"    - f''({b}) ≈ {d2f_b:.6f} => f(b)*f''(b) > 0 \n")
    else:
        raise ValueError(f"Không tìm thấy điểm Fourier trên khoảng [{a}, {b}].")
    # Khởi tạo bảng tính với bước 0 là điểm Fourier x0
    history = [{
        "n": 0,
        "x_n": round(x0, 6),
        "f(x_n)": round(f(x0), 6),
        "err": "-"
    }]
    x_prev = x0
    fx = f(x)
    for i in range(1, n_max + 1):
        fx_prev = f(x_prev)
        mau_so = fx_prev - fx
        if mau_so == 0:
            raise ZeroDivisionError("Mẫu số bằng 0")
        xn = x_prev - fx_prev * (x_prev - x) / mau_so
        err = abs(xn - x_prev)
        history.append({
            "n": i,
            "x_n": round(xn, 6),
            "f(x_n)": round(f(xn), 6),
            "err": round(err, 6)
        })
        if err < eps:
            return xn, i, err, pd.DataFrame(history)
        x_prev = xn
    return float(xn), i, float(err), pd.DataFrame(history)

print ("\n5. BÀI TOÁN ỨNG DỤNG")
"""
Bài 1  - BÀI GIẢNG: Ví dụ 3.1, giải f(x) = 2^x - 5x - 3 = 0 trên (-1; 0).
  Phần 1 - Chia đôi:
    a) Chứng tỏ (-1; 0) là khoảng phân ly nghiệm.
    Tìm nghiệm gần đúng với n_max = 6 bước lặp và đánh giá sai số.
    b) Muốn sai số eps = 10^-6 thì cần tối thiểu bao nhiêu bước lặp?
  Phần 2 - Dây cung: tìm nghiệm gần đúng với n = 4 bước lặp và đánh giá sai số bước 4.
"""
print("BÀI 1: GIẢI PHƯƠNG TRÌNH f(x) = 2^x - 5x - 3 = 0 TRÊN (-1, 0)")
def f1(x): return 2**x - 5*x - 3
print("Kiểm tra khoảng phân ly nghiệm")
check_isolate_roots(f1, -1.0, 0.0)
# Phương pháp chia đôi
print("\nPhương pháp Chia đôi (n_max = 6)")
x1, k1, err1, df1 = bisection_history(f1, -1.0, 0.0, eps=1e-6, n_max=6)
print(df1.to_string(index=False)) # In ra bảng lịch sử các bước lặp
print(f"-> Nghiệm gần đúng tại bước {k1}: x = {x1:.6f}")
print(f"-> Sai số tại bước {k1}: err = {err1:.6f}")
n1_min = bisection_steps(-1.0, 0.0, eps=1e-6)
print(f"-> Số bước lặp tối thiểu để đạt eps = 1e-6: {n1_min} bước lặp.")
# Phương pháp dây cung
print("\nPhương pháp Dây cung (n_max = 4)")
x1_fp, k1_fp, err1_fp, df1_fp = false_position_history(f1, -1.0, 0.0, eps=1e-6, n_max=4)
print(df1_fp.to_string(index=False))
print(f"-> Nghiệm gần đúng tại bước {k1_fp}: x = {x1_fp:.6f}")
print(f"-> Sai số tại bước {k1_fp}: err = {err1_fp:.6f}")

"""
Bài 2 - (Bài tập Chương 3, Bài tập 3.10):
Tìm nghiệm dương nhỏ nhất của phương trình x^3 - 3,23x^2 - 5,64x + 9,84 = 0 bằng phương pháp chia đôi và dây cung.
"""
print("\nBÀI 2: GIẢI PHƯƠNG TRÌNH f(x) = x^3 - 3,23x^2 - 5,64x + 9,84 = 0 ")
def f2_app(x): return x**3 - 3.23*(x**2) - 5.64*x + 9.84
# Quét khoảng phân ly nghiệm dương
intervals_pos = isolate_roots(f2_app, 0.0, 5.0, step=0.5)
print(f"Các khoảng chứa nghiệm dương tìm được: {intervals_pos}")
a2, b2 = intervals_pos[0] # Khoảng nghiệm dương nhỏ nhất [1.0, 1.5] hoặc [1.0, 2.0]
a2, b2 = 1.1, 1.5
print(f"Chọn khoảng phân ly cho nghiệm dương nhỏ nhất: [{a2}, {b2}]")
print("Kiểm tra khoảng phân ly nghiệm")
check_isolate_roots(f2_app, a2, b2)
# Phương pháp chia đôi
print("\nPhương pháp chia đôi (eps = 1e-5)")
x2_bi, k2_bi, err2_bi, df2_bi = bisection_history(f2_app, a2, b2, eps=1e-5, n_max=50)
print(df2_bi.to_string(index=False))
print(f"-> Nghiệm gần đúng tại bước {k2_bi}: x = {x2_bi:.6f}")
print(f"-> Sai số tại bước {k2_bi}: err = {err2_bi:.6f}")
n2_min = bisection_steps(a2, b2, eps=1e-5)
print(f"-> Số bước lặp tối thiểu để đạt eps = 1e-5: {n2_min} bước lặp.")
# Phương pháp dây cung
print("\nPhương pháp dây cung (eps = 1e-5)")
x2_fp, k2_fp, err2_fp, df2_fp = false_position_history(f2_app, a2, b2, eps=1e-5, n_max=50)
print(df2_fp.to_string(index=False))
print(f"-> Nghiệm gần đúng tại bước {k2_fp}: x = {x2_fp:.6f} ")
print(f"-> Sai số tại bước {k2_fp}: err = {err2_fp:.6f}")

"""
Bài 3. Một người vay P = 500 triệu đồng, trả đều A = 12 triệu đồng mỗi tháng trong N = 60 tháng.
Lãi suất tháng r thỏa mãn f(r) = A*[(1-(1+r)^-N)/r]-P = 0. Tìm r với sai số 1e-8 bằng hai phương pháp.
Khảo sát r khi A thay đổi từ 9 đến 15 triệu đồng, nhận xét điều gì xảy ra khi A*N<=P """
print("\nBÀI 3: BÀI TOÁN LÃI SUẤT TÀI CHÍNH (P = 500 tr, N = 60 tháng, eps = 1e-8)")
def make_f3(A, P=500.0, N=60):
    def f(r):
        if r == 0:
            return A * N - P
        return A * ((1.0 - (1.0 + r)**(-N)) / r) - P
    return f
f3 = make_f3(A=12.0)
# Kiểm tra khoảng phân ly r trong [0.001, 0.03] (0.1%/tháng - 3%/tháng)
print("Kiểm tra khoảng phân ly nghiệm r (với A = 12 triệu)")
check_isolate_roots(f3, 0.001, 0.03)
# Giải tìm r bằng 2 phương pháp với eps = 1e-8
print("\nPhương pháp chia đôi")
r_bi, k3_bi, err3_bi, df3_bi = bisection_history(f3, 0.001, 0.03, eps=1e-8, n_max=50)
print(df3_bi.to_string(index=False))
print(f"-> Lãi suất tháng (Chia đôi): r = {r_bi:.6f} ({r_bi*100:.6f}%/tháng) sau {k3_bi} bước")
print("\nPhương pháp dây cung")
r_fp, k3_fp, err3_fp, df3_fp = false_position_history(f3, 0.001, 0.03, eps=1e-8, n_max=50)
print(df3_fp.to_string(index=False))
print(f"-> Lãi suất tháng (Dây cung): r = {r_fp:.6f} ({r_fp*100:.6f}%/tháng) sau {k3_fp} bước")
# Khảo sát r khi A thay đổi từ 9 đến 15 triệu
print("\nKhảo sát r khi A thay đổi từ 9 đến 15 triệu đồng")
survey_results = []
for A_val in range(9, 16):
    f_temp = make_f3(A = float(A_val))
    r_sol, _, _, _ = bisection_history(f_temp, 1e-6, 0.05, eps=1e-8, n_max=100) # Tìm khoảng nghiệm
    survey_results.append({
        "A (triệu/tháng)": A_val,
        "A * N (triệu)": A_val * 60,
        "P (triệu)": 500,
        "r (lãi suất tháng)": round(r_sol, 6),
        "r (%/tháng)": round(r_sol * 100, 6),
        "Lãi suất năm (%/năm)": round(r_sol * 12 * 100, 6)
    })
df_survey = pd.DataFrame(survey_results)
print(df_survey.to_string(index=False))
print("\nNHẬN XÉT KHI A * N <= P")
print("""• Về mặt Toán học:
  - Khi A * N = P (ví dụ: A = 8.333 triệu/tháng -> 8.333 * 60 = 500 triệu), nghiệm của phương trình f(r) = 0 là r = 0.
  - Khi A * N < P (ví dụ: A = 8 triệu/tháng -> 8 * 60 = 480 triệu < 500 triệu), phương trình f(r) = 0 cho nghiệm r < 0 (lãi suất âm).
• Về mặt Kinh tế / Thực tế:
  - A*N<=P là dấu hiệu nhận biết khoản vay không có lãi (r = 0) hoặc lãi suất bị âm (r < 0).
  - Ngân hàng / Người cho vay không thu được tiền lãi (r = 0) hoặc bị lỗ vốn (r < 0).
  - Trong thực tế hợp đồng tín dụng thương mại, không tồn tại giao dịch vay vốn với A * N <= P (trừ các chính sách trợ giá/trả góp 0% lãi suất do bên thứ ba tài trợ).
""")

warnings.filterwarnings("ignore") # Bỏ cảnh báo khi tính đạo hàm bậc 2 gần biên (d2f_a, d2f_b) trong phương pháp dây cung
# 6. THỰC NGHIỆM SO SÁNH
print("6. THỰC NGHIỆM SO SÁNH: BẢNG SAI SỐ, ĐỒ THỊ, NHẬN XÉT")
# A. CÁC HÀM CƠ SỞ THEO ĐỀ BÀI
# Số bước lặp tối thiểu: def bisection_steps(a, b, eps)
# Phương pháp chia đôi, có bảng tính: def bisection_history(f, a, b, eps=1e-5, n_max=50)
# Phương pháp dây cung, có bảng tính, không in ra điểm fourier
def false_position_2(f, a, b, eps=1e-5, n_max=50):
    fa, fb = f(a), f(b)
    h = 1e-5
    d2f_a = (f(a + h) - 2 * fa + f(a - h)) / (h**2)
    d2f_b = (f(b + h) - 2 * fb + f(b - h)) / (h**2)
    # Xác định điểm Fourier x0 và mút cố định d
    if fa * d2f_a > 0:
        x0 = b
        x = a
    elif fb * d2f_b > 0:
        x0 = a
        x = b
    else:
        raise ValueError(f"Không tìm thấy điểm Fourier trên khoảng [{a}, {b}].")
    # Khởi tạo bảng lịch sử với bước 0 là điểm Fourier x0
    history = [{
        "n": 0,
        "x_n": round(x0, 6),
        "f(x_n)": round(f(x0), 6),
        "err": "-"  # Chưa có bước lặp trước nên sai số để '-'
    }]
    x_prev = x0
    fx = f(x)
    for i in range(1, n_max + 1):
        fx_prev = f(x_prev)
        mau_so = fx_prev - fx
        if mau_so == 0:
            raise ZeroDivisionError("Mẫu số bằng 0")
        xn = x_prev - fx_prev * (x_prev - x) / mau_so
        err = abs(xn - x_prev)
        history.append({
            "n": i,
            "x_n": round(xn, 6),
            "f(x_n)": round(f(xn), 6),
            "err": round(err, 6)
        })
        if err < eps:
            return xn, i, err, pd.DataFrame(history)
        x_prev = xn
    return float(xn), i, float(err), pd.DataFrame(history)

# B. KHAI BÁO 3 BÀI TOÁN KIỂM THỬ
#def f1(x): return 2**x - 5*x - 3
#def f2_app(x): return x**3 - 3.23*(x**2) - 5.64*x + 9.84
"""
def make_f3(A, P=500.0, N=60):
    def f(r):
        if r == 0:return A * N - P
        return A * ((1.0 - (1.0 + r)**(-N)) / r) - P
    return f
f3 = make_f3(A=12.0)"""
problems = [
    {"Ten": "Bài 1: 2^x - 5x - 3 = 0", "f": f1, "a": -1.0, "b": 0.0},
    {"Ten": "Bài 2: x^3 - 3.23x^2 - 5.64x + 9.84 = 0", "f": f2_app, "a": 1.1, "b": 1.5},
    {"Ten": "Bài 3: Lãi suất tài chính", "f": f3, "a": 0.001, "b": 0.03}
]

# C. SO SÁNH SỐ BƯỚC LẶP CHO 3 BÀI TOÁN
print("\nSO SÁNH SỐ BƯỚC LẶP CHO 3 BÀI TOÁN")
for idx, bt in enumerate(problems, 1): # Duyệt qua từng bài toán
    print(f"\n{bt['Ten']} TRÊN [{bt['a']}, {bt['b']}]")
    print(f"{'eps':>10} | {'Chia đôi (TT)':>14} | {'Chia đôi (LT)':>14} | {'Dây cung (TT)':>14}")
    for p in range(2, 11):
        eps = 10**(-p)
        _, k_bi, _, _ = bisection_history(bt["f"], bt["a"], bt["b"], eps=eps, n_max=30)
        k_lt = bisection_steps(bt["a"], bt["b"], eps)
        _, k_fp, _, _ = false_position_2(bt["f"], bt["a"], bt["b"], eps=eps, n_max=25)
        print(f"{eps:10.0e} | {k_bi:14d} | {k_lt:14d} | {k_fp:14d}")
print("""\n>>> NHẬN XÉT:
1. Tính chính xác của lý thuyết: Số bước lặp chia đôi thực tế hoàn toàn khớp hoặc
   chênh lệch không quá 1 bước so với công thức đánh giá theo lý thuyết.
2. Ưu thế của Dây cung ở điều kiện thường: Đối với cả 3 bài toán trơn, Dây cung
   hội tụ cực nhanh. Khi siết sai số xuống 10^-10, Chia đôi mất 28-30 bước,
   trong khi Dây cung chỉ cần từ 5-15 bước là đạt độ chính xác tuyệt đối.
""")

# D. VẼ ĐỒ THỊ SAI SỐ TRÊN THANG LOG
print("\nVẼ ĐỒ THỊ SAI SỐ TRÊN THANG LOG")
def get_history_bi(f, a, b, n_steps=30): # Lấy danh sách các x_k của phương pháp Chia đôi
    aa, bb = a, b
    xs = []
    for _ in range(n_steps):
        x = (aa + bb) / 2.0
        xs.append(x)
        if f(aa) * f(x) > 0: aa = x
        else: bb = x
    return xs
def get_history_fp(f, a, b, n_steps=25): # Lấy danh sách các x_k của phương pháp Dây cung
    _, _, _, df = false_position_2(f, a, b, eps=1e-15, n_max=n_steps)
    return df['x_n'].tolist()[1:]  # Bỏ bước 0 (điểm xuất phát Ban đầu)
for idx, bt in enumerate(problems, 1):
    # Tìm nghiệm chính xác cao bằng Chia đôi x*
    x_star, _, _, _ = bisection_history(bt["f"], bt["a"], bt["b"], eps=1e-15, n_max=100)
    # Lấy danh sách nghiệm
    xs_bi = get_history_bi(bt["f"], bt["a"], bt["b"], 30)
    xs_fp = get_history_fp(bt["f"], bt["a"], bt["b"], 25)
    # Tính sai số so với nghiệm x_star
    sai_bi = [max(abs(x - x_star), 1e-16) for x in xs_bi]
    sai_fp = [max(abs(x - x_star), 1e-16) for x in xs_fp]
    # Đồ thị Phương pháp Chia đôi
    plt.figure(figsize=(6, 4))
    plt.semilogy(range(1, len(sai_bi) + 1), sai_bi, 'b-o', label='Chia đôi', markersize=4) # Vẽ đồ thị sai số theo thang log
    plt.xlabel("Số bước lặp k")
    plt.ylabel("|x_k - x*| (thang log)")
    plt.title(f"Bài {idx}a - Chia đôi: {bt['Ten']}")
    plt.grid(True, which="both", linestyle="--", alpha=0.6) # Hiển thị lưới
    plt.legend() # Hiển thị chú thích
    plt.tight_layout() # Tự động điều chỉnh layout để tránh tràn chữ
    plt.show()
    # Đồ thị Phương pháp Dây cung
    plt.figure(figsize=(6, 4))
    plt.semilogy(range(1, len(sai_fp) + 1), sai_fp, 'r-s', label='Dây cung', markersize=4)
    plt.xlabel("Số bước lặp k")
    plt.ylabel("|x_k - x*| (thang log)")
    plt.title(f"Bài {idx}b - Dây cung: {bt['Ten']}")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
print("""\n>>> NHẬN XÉT:
1. Phương pháp Chia đôi (Hình a): Đường biểu diễn sai số có dạng một ĐƯỜNG THẲNG
   đi xuống đều đặn. Vì sai số tuân theo công thức tuyến tính e_(k+1) = 0.5 * e_k,
   khi lấy logarit sẽ tạo ra một hàm bậc nhất theo k, thể hiện tốc độ ổn định.
2. Phương pháp Dây cung (Hình b): Đường sai số dốc xuống GẦN NHƯ THẲNG ĐỨNG
   và chạm đáy giới hạn máy tính (1e-16) chỉ sau vài bước lặp, minh chứng cho
   hiệu quả vượt trội của việc tận dụng giá trị hàm số để ép sát nghiệm.
3. Kết luận: Dây cung hội tụ nhanh hơn Chia đôi rất nhiều, đặc biệt khi hàm số trơn
   và không có điểm cực trị gần nghiệm. Tuy nhiên, trong các trường hợp hàm số
   có dốc cực đoan hoặc gần điểm cực trị, Dây cung có thể gặp vấn đề hội tụ chậm hơn Chia đôi,
   như sẽ được minh họa trong phần "hàm dây cung hội tự chậm hơn chia đôi".
""")

# E. THỰC NGHIỆM HÀM DÂY CUNG HỘI TỤ CHẬM HƠN CHIA ĐÔI
print("\nTHỰC NGHIỆM TRƯỜNG HỢP DÂY CUNG HỘI TỤ CHẬM HƠN CHIA ĐÔI")
f_slow = lambda x: x**10 - 0.1
a_s, b_s = 0.0, 1.0
print(f">>> Hàm f(x) = x^10 - 0.1 trên [{a_s}, {b_s}]")
print(f"{'eps':>10} | {'Chia đôi (Thực tế)':>20} | {'Dây cung (Thực tế)':>20}")
for p in range(2, 7):
    eps = 10**(-p)
    _, k_bi, _, _ = bisection_history(f_slow, a_s, b_s, eps=eps, n_max=100)
    _, k_fp, _, _ = false_position_2(f_slow, a_s, b_s, eps=eps, n_max=100)
    print(f"{eps:10.0e} | {k_bi:20d} | {k_fp:20d}")
print(f""">>> Nhận xét: Để đạt độ chính xác 1e-6, phương pháp Chia đôi chỉ cần {k_bi} bước,
trong khi phương pháp Dây cung mất tới {k_fp} bước (nhiều hơn gấp đôi).""")
print(""">>> Giải thích:
Phương pháp Dây cung (Giữ mút cố định Fourier):
Gặp hiện tượng 'bẫy mút cố định' đối với các hàm dốc lệch góc cực đoan.
CƠ CHẾ BỊ CHẬM:
    1. Điều kiện Fourier: Tại b = 1.0, f(1) = 0.9 và f''(1) > 0 => f(1)*f''(1) > 0.
       Do đó, mút b = 1.0 bắt buộc phải chọn làm MÚT CỐ ĐỊNH.
    2. Điểm lặp xuất phát động là x0 = a = 0.0. Các dây cung sẽ luôn kéo từ điểm
       động x_n bên trái lên mút cố định (1.0, 0.9) rất cao ở bên phải.
    3. Vì hệ số góc (độ dốc) của dây cung nối tới (1.0, 0.9) quá lớn, giao điểm của
       dây cung với trục hoành (điểm x_n+1) chỉ có thể dịch chuyển từng bước
       SIÊU NHỎ sang bên phải. Phương pháp bị mất khả năng co hẹp khoảng cách từ mút phải.
""")

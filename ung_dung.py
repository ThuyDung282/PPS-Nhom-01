"""Bài 1 - BÀI GIẢNG: Ví dụ 3.1, giải f(x) = 2^x - 5x - 3 = 0 trên (-1; 0).

  Phần 1 - Chia đôi:
    a) Chứng tỏ (-1; 0) là khoảng phân ly nghiệm. Tìm nghiệm gần đúng với n_max = 5
       bước lặp và đánh giá sai số.
    b) Muốn sai số eps = 10^-6 thì cần tối thiểu bao nhiêu bước lặp?
  Phần 2 - Dây cung: tìm nghiệm gần đúng với n = 4 bước lặp và đánh giá sai số bước 4.
  
import math
import warnings

from bisection import bisection                                       # P2
from error_estimation import bisection_steps, false_position_error    # P4
from false_position import false_position                             # P3

A, B = -1, 0  # khoảng phân ly (-1; 0)

def f(x):
    """Hàm của ví dụ: f(x) = 2^x - 5x - 3.

    Require: x là số thực.
    Ensure: trả về f(x).
    """
    return 2 ** x - 5 * x - 3

def df(x):
    """Đạo hàm f'(x) = 2^x * ln2 - 5.

    Require: x là số thực.
    Ensure: trả về f'(x).
    """
    return 2 ** x * math.log(2) - 5

def d2f(x):
    """Đạo hàm cấp hai f''(x) = 2^x * (ln2)^2.

    Require: x là số thực.
    Ensure: trả về f''(x).
    """
    return 2 ** x * math.log(2) ** 2

def lap_n_buoc(phuong_phap, n):
    """Kết quả (x, k, err) của phương pháp sau đúng n bước lặp, để lập bảng từng bước.

    Dùng eps rất nhỏ để thuật toán không dừng sớm; P2, P3 sẽ cảnh báo "chưa đạt eps"
    nên ta tắt cảnh báo (cố ý). P2 lặp n_max + 1 lần, còn P3 lặp n_max lần.

    Require: phuong_phap là bisection (P2) hoặc false_position (P3); n >= 1.
    Ensure: trả về (x, k, err) do phuong_phap trả về.
    """
    n_max = n - 1 if phuong_phap is bisection else n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return phuong_phap(f, A, B, eps=1e-15, n_max=n_max)

if __name__ == "__main__":
    # PHẦN 1: CHIA ĐÔI 
    print("PHẦN 1 - Chia đôi")
    print(f"f(a) = {f(A)}, f(b) = {f(B)}, f'(b) = {df(B):.5f}")
    # f' = 2^x ln2 - 5 là hàm tăng nên f'(b) < 0 thì f' < 0 trên cả (a; b)
    if f(A) * f(B) >= 0 or df(B) >= 0:
        raise ValueError("(a; b) không phải khoảng phân ly nghiệm.")

    print(f"{'n':>2} {'a_n':>9} {'b_n':>9} {'x_n':>10} {'f(x_n)':>12}")
    x_0 = (A + B) / 2
    print(f"{0:2d} {A:9.5f} {B:9.5f} {x_0:10.6f} {f(x_0):12.5e}")
    for n in range(1, 6):
        x_n, _, err = lap_n_buoc(bisection, n)
        # P2 trả về trung điểm x_n và err = (b_n - a_n)/2 nên a_n = x_n - err, b_n = x_n + err
        print(f"{n:2d} {x_n - err:9.5f} {x_n + err:9.5f} {x_n:10.6f} {f(x_n):12.5e}")
    print(f"x_5 = {x_n}, sai số <= (b - a)/2^(5+1) = {(B - A) / 2 ** 6}")
    print(f"eps = 1e-6: log2((b - a)/eps) - 1 = {math.log2((B - A) / 1e-6) - 1:.5f}, "
          f"n_min = {bisection_steps(A, B, 1e-6)}")

    #  PHẦN 2: DÂY CUNG 
    print("\nPHẦN 2 - Dây cung")
    # Điểm Fourier c thỏa f(c)*f''(c) > 0 đứng yên; x_0 là đầu mút còn lại
    x_0 = B if f(A) * d2f(A) > 0 else A
    xs = [x_0] + [lap_n_buoc(false_position, n)[0] for n in range(1, 5)]
    print(f"x_0 = {x_0}")
    print(f"{'n':>2} {'x_n':>11} {'f(x_n)':>13} {'|x_n - x_(n-1)|':>16}")
    for n, x_n in enumerate(xs):
        buoc = "" if n == 0 else f"{abs(x_n - xs[n - 1]):16.3e}"
        print(f"{n:2d} {x_n:11.6f} {f(x_n):13.5e} {buoc}")
    # |f'| đơn điệu trên [a; b] nên m, M nằm ở hai đầu mút
    m, M = sorted([abs(df(A)), abs(df(B))])
    print(f"m = {m:.5f}, M = {M:.5f}, (M - m)/m = {(M - m) / m:.4f}")
    print(f"Sai số bước 4: |x_4 - x_3| = {abs(xs[4] - xs[3]):.3e}, "
          f"công thức (3.22) = {false_position_error(xs[4], xs[3], m, M):.3e}")
 
"""Bài 2 - BÀI TẬP 3.10: tìm nghiệm dương nhỏ nhất của
        x^3 - 3,23x^2 - 5,64x + 9,84 = 0
bằng phương pháp chia đôi. Làm thêm: giải bằng dây cung và so sánh số bước lặp.

from bisection import bisection                 # P2
from error_estimation import bisection_steps    # P4
from false_position import false_position       # P3
from isolate_roots import isolate_roots         # P1

def f(x):
    """Hàm của bài: f(x) = x^3 - 3,23x^2 - 5,64x + 9,84.

    Require: x là số thực.
    Ensure: trả về f(x).
    """
    return x ** 3 - 3.23 * x ** 2 - 5.64 * x + 9.84

def khoang_nho_nhat(ham, a, b, step):
    """Khoảng chứa nghiệm dương nhỏ nhất của ham trên [a; b], tìm bằng isolate_roots (P1).

    P1 quét từ trái sang phải nên khoảng đổi dấu đầu tiên chứa nghiệm nhỏ nhất.
    Bỏ qua khoảng suy biến (x; x) vì đó là nghiệm trùng đúng một điểm quét.

    Require: 0 <= a < b; step > 0.
    Ensure: trả về (a1, b1) với ham(a1)*ham(b1) < 0.
    Raises: ValueError nếu không có nghiệm dương trong [a; b].
    """
    cac_khoang = [kh for kh in isolate_roots(ham, a, b, step) if kh[0] < kh[1]]
    if len(cac_khoang) == 0:  # tình huống biên: không có nghiệm
        raise ValueError("Không tìm thấy nghiệm dương trong khoảng đã quét.")
    return cac_khoang[0]

if __name__ == "__main__":
    a, b = khoang_nho_nhat(f, 0, 5, 0.5)
    print(f"Khoảng chứa nghiệm dương nhỏ nhất (P1): [{a}; {b}]")
    print(f"{'eps':>8} {'chia đôi':>12} {'k':>3} {'k lý thuyết':>12} {'dây cung':>12} {'k':>3}")
    for eps in (1e-2, 1e-5, 1e-8):
        x_cd, k_cd, _ = bisection(f, a, b, eps=eps, n_max=200)
        x_dc, k_dc, _ = false_position(f, a, b, eps=eps, n_max=200)
        print(f"{eps:8.0e} {x_cd:12.8f} {k_cd:3d} {bisection_steps(a, b, eps):12d} "
              f"{x_dc:12.8f} {k_dc:3d}")
    print(f"Thay nghiệm vào f: f({x_cd:.8f}) = {f(x_cd):.2e}")

"""Bài 3 - BÀI NGOÀI (đề của cô): lãi suất vay trả góp.

ĐỀ BÀI:
    Một người vay P = 500 triệu đồng, trả đều A = 12 triệu đồng mỗi tháng trong
    N = 60 tháng. Lãi suất tháng r thỏa mãn
        f(r) = A * (1 - (1 + r)^(-N)) / r - P = 0.
    Tìm r với sai số 10^-8 bằng cả hai phương pháp (đáp số r ≈ 1,2833 %/tháng).
    Khảo sát r khi A thay đổi từ 9 đến 15 triệu đồng; nhận xét điều gì xảy ra
    khi A * N <= P.

Cách làm: viết hàm f(r), dùng isolate_roots (P1) tìm khoảng chứa nghiệm, rồi gọi phương pháp
cần dùng. Muốn đổi phương pháp chỉ cần thay bisection (P2) bằng false_position (P3), vì hai
hàm có cùng cách gọi. 
"""
from bisection import bisection                 # P2
from false_position import false_position       # P3
from isolate_roots import isolate_roots         # P1

P = 500.0      # số tiền vay (triệu đồng)
N = 60         # số tháng
PHUONG_PHAP = [("Chia đôi", bisection), ("Dây cung", false_position)]

def f(r, A=12.0):
    """Hàm của bài toán: f(r) = A * (1 - (1 + r)^(-N)) / r - P.

    Require: r > 0 (r = 0 làm mẫu số bằng 0); A > 0.
    Ensure: trả về f(r) (float).
    """
    if r <= 0:  # điều kiện áp dụng: chia cho r nên cần r > 0
        raise ValueError("Cần r > 0.")
    return A * (1 - (1 + r) ** (-N)) / r - P

def lai_suat(A, phuong_phap, eps=1e-8, n_max=200):
    """Tìm lãi suất tháng r ứng với tiền trả A bằng phương pháp được chọn.

    f giảm theo r, f(0+) = A*N - P và f(r) -> -P < 0 khi r lớn. Nên khi A*N > P, f đổi dấu
    đúng một lần trong [1e-6; 1]. Khi A*N <= P thì f(r) < 0 với mọi r > 0 nên
    isolate_roots không tìm thấy khoảng nào.

    Require: A > 0; phuong_phap là bisection (P2) hoặc false_position (P3); eps > 0; n_max >= 1.
    Ensure: trả về (r, k, err): lãi suất tháng, số bước lặp, sai số.
    Raises: ValueError nếu A <= 0 hoặc A*N <= P (không có lãi suất dương).
    """
    if A <= 0:
        raise ValueError("A phải dương.")

    def g(r):
        return f(r, A)

    cac_khoang = isolate_roots(g, 1e-6, 1.0, 0.01)  # tách nghiệm (P1)
    if len(cac_khoang) == 0:  # tình huống biên: A*N <= P
        raise ValueError("A*N <= P: tổng tiền trả không vượt số tiền vay, không có lãi suất dương.")
    a, b = cac_khoang[0]
    return phuong_phap(g, a, b, eps=eps, n_max=n_max)

if __name__ == "__main__":
    print("A = 12:")
    for ten, phuong_phap in PHUONG_PHAP:
        r, k, err = lai_suat(12, phuong_phap)
        print(f"  {ten:9s}: r = {100 * r:.6f} %/tháng, k = {k}, err = {err:.2e}")

    print("\nKhảo sát A từ 9 đến 15:")
    print(f"{'A':>6} {'r chia đôi (%)':>15} {'k':>3} {'r dây cung (%)':>15} {'k':>3}")
    for i in range(13):
        A = 9 + 0.5 * i
        r_cd, k_cd, _ = lai_suat(A, bisection)
        r_dc, k_dc, _ = lai_suat(A, false_position)
        print(f"{A:6.1f} {100 * r_cd:15.6f} {k_cd:3d} {100 * r_dc:15.6f} {k_dc:3d}")

    print(f"\nA*N <= P (P = {P:.0f}, N = {N}, tức A <= {P / N:.3f}):")
    for A in (8.0, P / N):
        try:
            lai_suat(A, bisection)
        except ValueError as loi:
            print(f"  A = {A:.3f}: {loi}")

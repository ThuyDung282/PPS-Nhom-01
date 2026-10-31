""Bài ứng dụng 1: VÍ DỤ 3.1 (bài giảng): Cho phương trình f(x) = 2^x - 5x - 3 = 0.

  PHẦN 1 - Phương pháp chia đôi
    a) Chứng tỏ (-1; 0) là khoảng phân ly nghiệm. Tìm nghiệm gần đúng bằng phương pháp
       chia đôi với n_max = 5 bước lặp và đánh giá sai số của nghiệm ở bước lặp đó.
    b) Nếu muốn sai số eps = 10^-6 thì cần tối thiểu bao nhiêu bước lặp?

  PHẦN 2 - Phương pháp dây cung (cùng phương trình, cùng khoảng (-1; 0))
    Tìm nghiệm gần đúng bằng phương pháp dây cung với n = 4 bước lặp và
    đánh giá sai số tại bước lặp 4.

Đáp số tham khảo của đề: nghiệm x* ≈ -0,453996.

Chỉ GỌI các hàm của nhóm: P2 bisection, P3 false_position, P4 bisection_steps và
false_position_error. Không dùng numpy.linalg, scipy... (R3).

Chạy:  python ung_dung_bai1.py
"""
import math
import warnings

# Tên module theo repo của nhóm. 
from bisection import bisection                                       # P2
from error_estimation import bisection_steps, false_position_error    # P4
from false_position import false_position                             # P3

A, B = -1.0, 0.0   # khoảng phân ly (-1; 0)
EPS_NHO = 1e-15    # eps rất nhỏ để P2, P3 không dừng sớm: số bước do ta chọn bằng n_max
EPS_THAM_CHIEU = 1e-14  # độ chính xác để tính nghiệm tham chiếu x*

# Hàm của bài và các đạo hàm (dùng để kiểm tra điều kiện lý thuyết)
def f(x):
    """Hàm của Ví dụ 3.1: f(x) = 2^x - 5x - 3.

    Require: x là số thực.
    Ensure: trả về f(x). Có f(-1) = 5/2 > 0 và f(0) = -2 < 0.
    Thuật toán: không phải thuật toán bài giảng, đây là hàm của ví dụ.
    """
    return 2 ** x - 5 * x - 3

def df(x):
    """Đạo hàm f'(x) = 2^x * ln2 - 5.

    Require: x là số thực.
    Ensure: trả về f'(x).
    """
    return 2 ** x * math.log(2) - 5

def d2f(x):
    """Đạo hàm cấp hai f''(x) = 2^x * (ln2)^2 > 0 với mọi x.

    Require: x là số thực.
    Ensure: trả về f''(x).
    """
    return 2 ** x * math.log(2) ** 2

# PHẦN 1: chia đôi
def kiem_tra_dau_vao(ham, a, b, so_buoc):
    """Kiểm tra dữ liệu đầu vào (R6).

    Require: ham là hàm số; a < b; so_buoc là số nguyên.
    Ensure: không trả về gì; báo ValueError nếu so_buoc < 1 hoặc ham(a)*ham(b) >= 0.
    """
    if so_buoc < 1:
        raise ValueError("so_buoc phải >= 1.")
    if ham(a) * ham(b) >= 0:
        raise ValueError("Cần f(a)*f(b) < 0 (a; b là khoảng phân ly nghiệm).")


def khoang_phan_ly(ham, dao_ham, a, b):
    """Kiểm tra (a; b) là khoảng phân ly nghiệm: f(a)*f(b) < 0 và f' không đổi dấu.

    Ở đây f'(x) = 2^x ln2 - 5 là hàm tăng, nên chỉ cần f'(b) < 0 là f' < 0 trên cả (a; b),
    tức f đơn điệu giảm và nghiệm là duy nhất.

    Require: ham liên tục trên [a; b]; dao_ham là hàm tăng trên [a; b]; a < b.
    Ensure: trả về bộ (f_a, f_b, f'_b, la_khoang_phan_ly) với la_khoang_phan_ly là bool.
    """
    f_a, f_b, d_b = ham(a), ham(b), dao_ham(b)
    return f_a, f_b, d_b, (f_a * f_b < 0 and d_b < 0)


def chia_doi_tung_buoc(ham, a, b, so_buoc):
    """Chạy chia đôi sau n = 1, 2, ..., so_buoc bước và ghi lại khoảng (a_n; b_n) sau mỗi bước.

    Cách làm: gọi P2 với n_max = n - 1 thì vòng lặp của P2 chạy đúng n lần. P2 trả về
    x_n là trung điểm và err_n = (b_n - a_n)/2 nên a_n = x_n - err_n, b_n = x_n + err_n.
    (P2 sẽ cảnh báo vì chưa đạt eps; ta tắt cảnh báo vì đây là cố ý.)

    Require: ham(a)*ham(b) < 0; so_buoc >= 1.
    Ensure: trả về danh sách các bộ (n, a_n, b_n, x_n, err_n) với n = 1..so_buoc.
    Thuật toán: chia đôi (P2).
    """
    kiem_tra_dau_vao(ham, a, b, so_buoc)
    ket_qua = []
    for n in range(1, so_buoc + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_n, _, err_n = bisection(ham, a, b, eps=EPS_NHO, n_max=n - 1)
        ket_qua.append((n, x_n - err_n, x_n + err_n, x_n, err_n))
    return ket_qua

def bang_chia_doi(ham, a, b, n_max):
    """Bảng tính của chia đôi từ n = 0 đến n_max: (n, a_n, b_n, x_n, f(x_n)).

    Hàng n = 0 là khoảng ban đầu (a_0; b_0) = (a; b) với x_0 = (a + b)/2.

    Require: ham(a)*ham(b) < 0; n_max >= 1.
    Ensure: trả về danh sách n_max + 1 bộ (n, a_n, b_n, x_n, f(x_n)).
    Thuật toán: chia đôi (P2).
    """
    x_0 = (a + b) / 2
    bang = [(0, a, b, x_0, ham(x_0))]
    for n, a_n, b_n, x_n, _ in chia_doi_tung_buoc(ham, a, b, n_max):
        bang.append((n, a_n, b_n, x_n, ham(x_n)))
    return bang

def can_sai_so_chia_doi(a, b, n):
    """Cận sai số của chia đôi sau n bước: |x_n - x*| <= (b - a) / 2^(n+1)  (công thức 3.3).

    Require: a < b; n >= 0.
    Ensure: trả về cận sai số (float).
    """
    return (b - a) / 2 ** (n + 1)

def so_buoc_toi_thieu(a, b, eps):
    """Số bước tối thiểu để đạt sai số eps theo công thức (3.4).

    Require: a < b; eps > 0.
    Ensure: trả về bộ (gia_tri_thuc, so_buoc) với gia_tri_thuc = log2((b - a)/eps) - 1
            (chưa làm tròn) và so_buoc là số bước do P4 tính.
    Thuật toán: số bước lý thuyết (P4).
    """
    gia_tri_thuc = math.log2((b - a) / eps) - 1
    return gia_tri_thuc, bisection_steps(a, b, eps)

# PHẦN 2: dây cung
def diem_fourier(ham, d2_ham, a, b):
    """Tìm điểm Fourier của phương pháp dây cung: đầu mút c có ham(c) * ham''(c) > 0.

    Điểm này đứng yên trong mọi bước lặp; điểm xuất phát x_0 là đầu mút còn lại.

    Require: ham'' không đổi dấu trên [a; b]; ham(a)*ham(b) < 0.
    Ensure: trả về bộ (c, x_0) với c là điểm Fourier, x_0 là điểm xuất phát.
    """
    if ham(a) * d2_ham(a) > 0:
        return a, b
    return b, a

def day_cung_tung_buoc(ham, a, b, so_buoc):
    """Dãy x_0, x_1, ..., x_so_buoc của phương pháp dây cung.

    x_0 = a nếu f(a) < 0 (công thức 3.20), ngược lại x_0 = b (công thức 3.19).
    Từ x_1 trở đi gọi P3 với n_max = n thì P3 lặp đúng n bước và trả về x_n.
    (P3 sẽ cảnh báo vì chưa đạt eps; ta tắt cảnh báo vì đây là cố ý.)

    Require: ham(a)*ham(b) < 0; so_buoc >= 1.
    Ensure: trả về danh sách [x_0, x_1, ..., x_so_buoc].
    Thuật toán: dây cung (P3).
    """
    kiem_tra_dau_vao(ham, a, b, so_buoc)
    xs = [a if ham(a) < 0 else b]  # x_0
    for n in range(1, so_buoc + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_n, _, _ = false_position(ham, a, b, eps=EPS_NHO, n_max=n)
        xs.append(x_n)
    return xs
    
def sai_so_day_cung(xs, dao_ham, a, b, x_dung):
    """Đánh giá sai số của dây cung tại bước cuối n = len(xs) - 1 (cần n >= 1).

    Ba con số:
      - sai_so_that = |x_n - x*|
      - can_tho     = |x_n - x_(n-1)|  (đúng khi (M - m)/m <= 1)
      - can_p4      = (M - m)/m * |x_n - x_(n-1)|  (công thức 3.22, tính bằng P4)
    với m = min|f'|, M = max|f'| trên [a; b]. Ở đây |f'(x)| = 5 - 2^x ln2 giảm theo x
    nên m = |f'(b)| và M = |f'(a)|.

    Require: xs từ day_cung_tung_buoc; a < b; x_dung là nghiệm đúng.
    Ensure: trả về bộ (sai_so_that, can_tho, can_p4, m, M).
    Thuật toán: đánh giá sai số dây cung (P4 false_position_error).
    """
    m, M = abs(dao_ham(b)), abs(dao_ham(a))
    x_n, x_truoc = xs[-1], xs[-2]
    return (abs(x_n - x_dung), abs(x_n - x_truoc),
            false_position_error(x_n, x_truoc, m, M), m, M)

def nghiem_tham_chieu(ham, a, b):
    """Nghiệm x* tính bằng chia đôi (P2) với sai số rất nhỏ, chỉ để đối chiếu.

    Require: ham(a)*ham(b) < 0.
    Ensure: trả về x* (float).
    """
    x, _, _ = bisection(ham, a, b, eps=EPS_THAM_CHIEU, n_max=100)
    return x

# Chương trình chính
def main():
    """Chạy Ví dụ 3.1 bằng chia đôi (phần 1) và dây cung (phần 2)."""
    x_dung = nghiem_tham_chieu(f, A, B)

    # PHẦN 1: CHIA ĐÔI
    print("VÍ DỤ 3.1 - PHẦN 1: chia đôi, f(x) = 2^x - 5x - 3 trên (-1; 0)")
    f_a, f_b, d_b, ok = khoang_phan_ly(f, df, A, B)
    print(f"  a) f(-1) = {f_a}, f(0) = {f_b}  =>  f(-1)*f(0) < 0 nên có nghiệm trong (-1; 0)")
    print(f"     f'(x) = 2^x ln2 - 5 là hàm tăng, f'(0) = {d_b:.5f} < 0 nên f' < 0 trên (-1; 0):"
          " f đơn điệu, nghiệm duy nhất")
    print(f"     => (-1; 0) là khoảng phân ly nghiệm: {'đúng' if ok else 'SAI'}")
    print("     Công thức lặp: x_n = (a_n + b_n)/2")
    print(f"  {'n':>3} {'a_n':>10} {'b_n':>10} {'x_n':>11} {'f(x_n)':>13}")
    bang = bang_chia_doi(f, A, B, 5)
    for n, a_n, b_n, x_n, f_x in bang:
        print(f"  {n:3d} {a_n:10.5f} {b_n:10.5f} {x_n:11.6f} {f_x:13.6e}")
    x5 = bang[-1][3]
    print(f"     Nghiệm gần đúng tại n = 5: x_5 = {x5}")
    print(f"     Sai số: |x_5 - x*| <= (b - a)/2^(5+1) = {can_sai_so_chia_doi(A, B, 5)}"
          f"  (sai số thật: {abs(x5 - x_dung):.6f})")

    gia_tri, so_buoc = so_buoc_toi_thieu(A, B, 1e-6)
    print(f"  b) eps = 1e-6: n_min = log2((b - a)/eps) - 1 = {gia_tri:.5f}"
          f"  =>  cần {so_buoc} bước lặp (P4)")

    # PHẦN 2: DÂY CUNG 
    print("\nVÍ DỤ 3.1 - PHẦN 2: dây cung, f(x) = 2^x - 5x - 3 trên (-1; 0), n = 4")
    c, x_0 = diem_fourier(f, d2f, A, B)
    print(f"  (-1; 0) là khoảng phân ly (phần 1). f(-1) = {f(A)} > 0 và f''(x) > 0 nên "
          f"f(-1)*f''(-1) > 0: x = {c:.0f} là điểm Fourier")
    print(f"  => chọn xuất phát ban đầu x_0 = {x_0:.0f}")
    print("  Công thức lặp: x_n = x_(n-1) - f(x_(n-1)) / (f(x_(n-1)) - f(a)) * (x_(n-1) - a)"
          f"   (a = {A:.0f}, f(a) = {f(A)})")
    xs = day_cung_tung_buoc(f, A, B, 4)
    print(f"  {'n':>3} {'x_n':>12} {'f(x_n)':>14}")
    for n, x_n in enumerate(xs):
        print(f"  {n:3d} {x_n:12.6f} {f(x_n):14.6e}")
    that, tho, p4, m, M = sai_so_day_cung(xs, df, A, B, x_dung)
    print(f"  Sai số tại bước 4 (m = {m:.5f}, M = {M:.5f}, (M - m)/m = {(M - m) / m:.4f} <= 1):")
    print(f"     |x_4 - x*|                      = {that:.3e}   (sai số thật)")
    print(f"     |x_4 - x_3|                     = {tho:.3e}   (cận thô, vì (M - m)/m <= 1)")
    print(f"     (M - m)/m * |x_4 - x_3| (P4)    = {p4:.3e}   (cận theo công thức 3.22)")
    print(f"  Nghiệm tham chiếu x* = {x_dung:.8f}")

if __name__ == "__main__":
    main()

"""Bài ứng dụng 2 - BÀI TẬP: nghiệm dương nhỏ nhất của phương trình bậc ba.

ĐỀ BÀI (Bài tập Chương 3, Bài tập 3.10):
    Tìm nghiệm dương nhỏ nhất của phương trình
        x^3 - 3,23x^2 - 5,64x + 9,84 = 0
    bằng phương pháp chia đôi.
    Nhóm làm thêm: giải bằng cả dây cung, so sánh với số bước lý thuyết ở các mức
    sai số eps = 10^-2, 10^-5, 10^-8.

Chỉ GỌI các hàm của nhóm: P1 isolate_roots, P2 bisection, P3 false_position,
P4 bisection_steps. Không dùng numpy.linalg, scipy... (R3).

Chạy:  python ung_dung_bai2.py
"""
# Tên module theo file Test của nhóm.
from bisection import bisection                 # P2
from error_estimation import bisection_steps     # P4 (file error_estimation.py)
from false_position import false_position       # P3
from isolate_roots import isolate_roots         # P1

DANH_SACH_EPS = [1e-2, 1e-5, 1e-8]  # các mức sai số để so sánh
N_MAX = 200                         # số bước lặp tối đa

def f_bai2(x):
    """Hàm của Bài 2: f(x) = x^3 - 3,23x^2 - 5,64x + 9,84.

    Require: x là số thực.
    Ensure: trả về f(x).
    Thuật toán: không phải thuật toán bài giảng, đây là hàm mục tiêu của bài toán.
    """
    return x ** 3 - 3.23 * x ** 2 - 5.64 * x + 9.84

def nghiem_duong_nho_nhat(f, a, b, step):
    """Tìm khoảng chứa nghiệm dương nhỏ nhất của f trên [a; b] bằng isolate_roots (P1).

    P1 quét từ trái sang phải nên khoảng đầu tiên chứa nghiệm nhỏ nhất.
    Bỏ qua khoảng suy biến (x; x), tức là nghiệm trùng đúng một điểm quét.

    Require: 0 <= a < b; step > 0.
    Ensure: trả về (a1, b1) với f(a1)*f(b1) < 0, là khoảng chứa nghiệm dương nhỏ nhất.
    Raises: ValueError nếu không có nghiệm dương trong [a; b].
    Thuật toán: tách nghiệm bằng quét dấu (P1).
    """
    cac_khoang = [kh for kh in isolate_roots(f, a, b, step) if kh[0] < kh[1]]
    if len(cac_khoang) == 0:  # tình huống biên: không có nghiệm
        raise ValueError("Không tìm thấy nghiệm dương trong khoảng đã quét.")
    return cac_khoang[0]

def bang_so_sanh(f, a, b, danh_sach_eps, n_max):
    """Giải f(x) = 0 trên [a; b] bằng cả hai phương pháp với từng mức sai số.

    Require: f(a)*f(b) < 0; danh_sach_eps gồm các số dương; n_max >= 1.
    Ensure: trả về danh sách các bộ
        (eps, x_chia_doi, k_chia_doi, k_ly_thuyet, x_day_cung, k_day_cung).
    Thuật toán: chia đôi (P2), dây cung (P3), số bước lý thuyết (P4).
    """
    bang = []
    for eps in danh_sach_eps:
        x_cd, k_cd, _ = bisection(f, a, b, eps=eps, n_max=n_max)
        x_dc, k_dc, _ = false_position(f, a, b, eps=eps, n_max=n_max)
        bang.append((eps, x_cd, k_cd, bisection_steps(a, b, eps), x_dc, k_dc))
    return bang

def in_bang(bang):
    """In bảng so sánh. Require: bang từ bang_so_sanh. Ensure: in ra màn hình."""
    print(f"{'eps':>8} {'x chia đôi':>13} {'k':>3} {'k lý thuyết':>12} {'x dây cung':>13} {'k':>3}")
    for eps, x_cd, k_cd, k_lt, x_dc, k_dc in bang:
        print(f"{eps:8.0e} {x_cd:13.8f} {k_cd:3d} {k_lt:12d} {x_dc:13.8f} {k_dc:3d}")

def main():
    """Chạy Bài 2 và in kết quả."""
    print("Bài 2 (Bài tập 3.10): x^3 - 3,23x^2 - 5,64x + 9,84 = 0, nghiệm dương nhỏ nhất")
    a, b = nghiem_duong_nho_nhat(f_bai2, 0.0, 5.0, 0.5)
    print(f"  Khoảng chứa nghiệm dương nhỏ nhất (P1): [{a}; {b}]")
    in_bang(bang_so_sanh(f_bai2, a, b, DANH_SACH_EPS, N_MAX))
    x, _, _ = bisection(f_bai2, a, b, eps=1e-8, n_max=N_MAX)
    print(f"  Thay lại: f({x:.8f}) = {f_bai2(x):.2e}")

if __name__ == "__main__":
    main()

"""Bài ứng dụng 3 - BÀI NGOÀI (đề của cô): lãi suất vay trả góp.

ĐỀ BÀI:
    Một người vay P = 500 triệu đồng, trả đều A = 12 triệu đồng mỗi tháng trong
    N = 60 tháng. Lãi suất tháng r thỏa mãn
        f(r) = A * (1 - (1 + r)^(-N)) / r - P = 0.
    Tìm r với sai số 10^-8 bằng cả hai phương pháp (đáp số r ≈ 1,2833 %/tháng).
    Khảo sát r khi A thay đổi từ 9 đến 15 triệu đồng; nhận xét điều gì xảy ra
    khi A * N <= P.

GỌI các hàm của nhóm (không cài lại thuật toán):
    P1 isolate_roots  : tách khoảng chứa nghiệm
    P2 bisection      : phương pháp chia đôi
    P3 false_position : phương pháp dây cung
    P4 bisection_steps: số bước chia đôi theo lý thuyết
Không dùng numpy.linalg, scipy... (R3).

Chạy:  python ung_dung.py
"""
# Tên module theo file Test của nhóm. 
from bisection import bisection                 # P2
from error_estimation import bisection_steps     # P4 (file error_estimation.py)
from false_position import false_position       # P3
from isolate_roots import isolate_roots         # P1

# Số liệu của đề bài
P = 500.0      # số tiền vay (triệu đồng)
A = 12.0       # tiền trả mỗi tháng (triệu đồng)
N = 60         # số tháng
EPS = 1e-8     # sai số yêu cầu
N_MAX = 200    # số bước lặp tối đa

# 1. Mô hình hóa
def f_lai_suat(r, A, P, N):
    """Tính f(r) = A*(1 - (1+r)^(-N))/r - P.

    Ý nghĩa: A*(1 - (1+r)^(-N))/r là tổng giá trị hiện tại của N lần trả A,
    nên f(r) = 0 khi giá trị hiện tại đó đúng bằng số tiền vay P.

    Require:
        r : lãi suất tháng, r > 0 (r = 0 làm mẫu số bằng 0)
        A, P : số thực dương;  N : số nguyên dương
    Ensure:
        Trả về f(r) (float).
    Thuật toán: không phải thuật toán bài giảng, đây là hàm mục tiêu của bài toán.
    """
    if r <= 0:  # điều kiện áp dụng: chia cho r nên cần r > 0
        raise ValueError("Cần r > 0.")
    return A * (1 - (1 + r) ** (-N)) / r - P


def kiem_tra_tham_so(A, P, N):
    """Kiểm tra dữ liệu đầu vào (R6).

    Require: A, P, N là các số.
    Ensure: không trả về gì; báo ValueError nếu A <= 0, P <= 0 hoặc N không là số nguyên dương.
    """
    if A <= 0 or P <= 0:
        raise ValueError("A và P phải dương.")
    if int(N) != N or N < 1:
        raise ValueError("N phải là số nguyên dương.")

# 2. Tìm khoảng chứa nghiệm (dùng P1) và giải (dùng P2, P3)
def tim_khoang_nghiem(A, P, N, step=0.01):
    """Tìm khoảng [a; b] chứa nghiệm r > 0 của f(r) = 0 bằng isolate_roots (P1).

    Vì sao quét r trong [1e-6; 1]?
      - f giảm theo r; khi r -> 0+ thì f -> A*N - P; khi r lớn thì f -> -P < 0.
      - Nên nếu A*N > P thì f đổi dấu đúng một lần với r > 0.
      - Bắt đầu từ 1e-6 (không phải 0) để tránh chia cho 0; r = 1 là 100%/tháng, đủ lớn.
      - Nếu A*N <= P thì f(r) < 0 với mọi r > 0, không có lãi suất dương -> báo lỗi.

    Require: A, P > 0; N nguyên dương; step > 0.
    Ensure: trả về (a, b) với f(a)*f(b) < 0.
    Raises: ValueError nếu A*N <= P hoặc không tách được nghiệm.
    Thuật toán: tách nghiệm bằng quét dấu (P1).
    """
    kiem_tra_tham_so(A, P, N)
    if A * N <= P + 1e-9:  # tình huống biên: tổng tiền trả không vượt số tiền vay (+1e-9 để bù sai số làm tròn)
        raise ValueError("A*N <= P: tổng tiền trả không vượt số tiền vay, không có lãi suất dương.")

    def f(r):
        return f_lai_suat(r, A, P, N)

    cac_khoang = isolate_roots(f, 1e-6, 1.0, step)  # dùng P1
    if len(cac_khoang) == 0:
        raise ValueError("Không tách được khoảng chứa nghiệm trong [1e-6; 1].")
    return cac_khoang[0]  # f đơn điệu nên chỉ có một khoảng


def giai_lai_suat(A, P, N, eps, n_max, phuong_phap):
    """Tìm lãi suất tháng r bằng chia đôi hoặc dây cung.

    Require:
        A, P > 0; N nguyên dương; eps > 0; n_max >= 1
        phuong_phap : "chia_doi" hoặc "day_cung"
    Ensure:
        Trả về (r, k, err): nghiệm xấp xỉ, số bước lặp, sai số (do P2/P3 trả về).
    Thuật toán: chia đôi (P2) hoặc dây cung (P3).
    """
    a, b = tim_khoang_nghiem(A, P, N)  # bước 1: tách nghiệm (P1)

    def f(r):
        return f_lai_suat(r, A, P, N)

    if phuong_phap == "chia_doi":  # bước 2: giải bằng P2
        return bisection(f, a, b, eps=eps, n_max=n_max)
    if phuong_phap == "day_cung":  # bước 2: giải bằng P3
        return false_position(f, a, b, eps=eps, n_max=n_max)
    raise ValueError('phuong_phap phải là "chia_doi" hoặc "day_cung".')

# 3. Khảo sát khi A thay đổi
def khao_sat(danh_sach_A, P, N, eps, n_max):
    """Giải bằng cả hai phương pháp với từng giá trị A.

    Require: danh_sach_A gồm các số thỏa A*N > P; P, N, eps, n_max hợp lệ.
    Ensure: trả về danh sách các bộ (A, r_chia_doi, k_chia_doi, r_day_cung, k_day_cung).
    """
    ket_qua = []
    for a_i in danh_sach_A:
        r_cd, k_cd, _ = giai_lai_suat(a_i, P, N, eps, n_max, "chia_doi")
        r_dc, k_dc, _ = giai_lai_suat(a_i, P, N, eps, n_max, "day_cung")
        ket_qua.append((float(a_i), r_cd, k_cd, r_dc, k_dc))
    return ket_qua

def in_bang(ket_qua):
    """In bảng kết quả khảo sát. Require: ket_qua từ khao_sat. Ensure: in ra màn hình."""
    print(f"{'A':>6} {'A*N':>6} {'r chia đôi (%)':>15} {'k':>3} {'r dây cung (%)':>15} {'k':>3}")
    for a_i, r_cd, k_cd, r_dc, k_dc in ket_qua:
        print(f"{a_i:6.1f} {a_i * N:6.0f} {100 * r_cd:15.6f} {k_cd:3d} {100 * r_dc:15.6f} {k_dc:3d}")


# 4. Chương trình chính
def main():
    """Chạy toàn bộ bài toán ứng dụng và in kết quả."""
    # --- Phần 1: A = 12, giải bằng hai phương pháp ---
    a, b = tim_khoang_nghiem(A, P, N)
    print(f"[1] Khoảng chứa nghiệm (P1): [{a:.6f}; {b:.6f}]")
    print(f"    Số bước chia đôi theo lý thuyết (P4): {bisection_steps(a, b, EPS)}")
    for ten in ("chia_doi", "day_cung"):
        r, k, err = giai_lai_suat(A, P, N, EPS, N_MAX, ten)
        print(f"    {ten:9s}: r = {100 * r:.6f} %/tháng, k = {k}, "
              f"err = {err:.2e}, |f(r)| = {abs(f_lai_suat(r, A, P, N)):.2e}")
    print("    Đáp số của đề: r ≈ 1,2833 %/tháng")

    # --- Phần 2: khảo sát A từ 9 đến 15 ---
    danh_sach_A = [9.0 + 0.5 * i for i in range(13)]  # 9,0; 9,5; ...; 15,0
    ket_qua = khao_sat(danh_sach_A, P, N, EPS, N_MAX)
    print("\n[2] Khảo sát r khi A thay đổi từ 9 đến 15:")
    in_bang(ket_qua)

    # --- Phần 3: tình huống A*N <= P ---
    print(f"\n[3] Tình huống A*N <= P (P = {P:.0f}, N = {N}, nên A <= {P / N:.3f}):")
    for a_loi in (8.0, P / N):
        try:
            giai_lai_suat(a_loi, P, N, EPS, N_MAX, "chia_doi")
        except ValueError as loi:
            print(f"    A = {a_loi:.3f} (A*N = {a_loi * N:.1f}): báo lỗi -> {loi}")


if __name__ == "__main__":
    main()


"""Bài 1  - BÀI GIẢNG: Ví dụ 3.1, giải f(x) = 2^x - 5x - 3 = 0 trên (-1; 0).

  Phần 1 - Chia đôi:
    a) Chứng tỏ (-1; 0) là khoảng phân ly nghiệm. Tìm nghiệm gần đúng với n_max = 5
       bước lặp và đánh giá sai số.
    b) Muốn sai số eps = 10^-6 thì cần tối thiểu bao nhiêu bước lặp?
  Phần 2 - Dây cung: tìm nghiệm gần đúng với n = 4 bước lặp và đánh giá sai số bước 4.

"""
import math
import warnings

from bisection import bisection                                       # P2
from error_estimation import bisection_steps, false_position_error    # P4
from false_position import false_position                             # P3

A, B = -1, 0  # khoảng phân ly (-1; 0)

def f(x):
    """Hàm của ví dụ: f(x) = 2^x - 5x - 3.

    Require: x là số thực.
    Ensure: trả về f(x). Có f(-1) = 5/2 > 0 và f(0) = -2 < 0.
    """
    return 2 ** x - 5 * x - 3

def df(x):
    """Đạo hàm f'(x) = 2^x * ln2 - 5.

    Require: x là số thực.
    Ensure: trả về f'(x).
    """
    return 2 ** x * math.log(2) - 5

def chay_n_buoc(phuong_phap, n_max):
    """Chạy phương pháp trên (A; B) với n_max bước lặp để lấy kết quả của một bước n cụ thể.

    Dùng eps rất nhỏ để thuật toán không dừng sớm; P2, P3 sẽ cảnh báo "chưa đạt eps"
    nên ta tắt cảnh báo (cố ý). Lưu ý: P2 lặp n_max + 1 lần, còn P3 lặp n_max lần.

    Require: phuong_phap là bisection (P2) hoặc false_position (P3); n_max >= 0.
    Ensure: trả về (x, k, err) do phuong_phap trả về.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return phuong_phap(f, A, B, eps=1e-15, n_max=n_max)
      
def main():
    """Chạy Ví dụ 3.1 bằng chia đôi (phần 1) và dây cung (phần 2)."""
    #  PHẦN 1: CHIA ĐÔI 
    print("PHẦN 1 - Chia đôi")
    if f(A) * f(B) >= 0:  # tình huống biên: không phải khoảng phân ly
        raise ValueError("Cần f(a)*f(b) < 0.")
    print(f"a) f(-1) = {f(A)}, f(0) = {f(B)}: f(-1)*f(0) < 0 nên có nghiệm trong (-1; 0)")
    print(f"   f'(0) = {df(B):.4f} < 0 và f' tăng nên f' < 0 trên cả (-1; 0): nghiệm duy nhất")
    print(f"   {'n':>2} {'a_n':>9} {'b_n':>9} {'x_n':>10} {'f(x_n)':>12}")
    x_0 = (A + B) / 2
    print(f"   {0:2d} {A:9.5f} {B:9.5f} {x_0:10.6f} {f(x_0):12.5e}")
    for n in range(1, 6):
        x_n, _, err = chay_n_buoc(bisection, n - 1)  # P2 lặp (n - 1) + 1 = n lần
        # P2 trả về x_n là trung điểm và err = (b_n - a_n)/2, nên a_n = x_n - err, b_n = x_n + err
        print(f"   {n:2d} {x_n - err:9.5f} {x_n + err:9.5f} {x_n:10.6f} {f(x_n):12.5e}")
    print(f"   Nghiệm gần đúng x_5 = {x_n}")
    print(f"   Sai số: |x_5 - x*| <= (b - a)/2^(5+1) = {(B - A) / 2 ** 6}")
    n_min = math.log2((B - A) / 1e-6) - 1
    print(f"b) eps = 1e-6: n_min = log2((b - a)/eps) - 1 = {n_min:.5f}"
          f" => cần {bisection_steps(A, B, 1e-6)} bước lặp (P4)")

    # PHẦN 2: DÂY CUNG 
    print("\nPHẦN 2 - Dây cung")
    # f''(x) = 2^x (ln2)^2 > 0 và f(-1) > 0 nên f(-1)*f''(-1) > 0: -1 là điểm Fourier (đứng yên)
    print("   f(-1) > 0 và f'' > 0 nên -1 là điểm Fourier => chọn xuất phát x_0 = 0")
    xs = [B] + [chay_n_buoc(false_position, n)[0] for n in range(1, 5)]  # P3 lặp đúng n lần
    print(f"   {'n':>2} {'x_n':>11} {'f(x_n)':>13}")
    for n, x_n in enumerate(xs):
        print(f"   {n:2d} {x_n:11.6f} {f(x_n):13.5e}")
    m, M = abs(df(B)), abs(df(A))  # |f'| giảm theo x nên nhỏ nhất tại b = 0, lớn nhất tại a = -1
    print(f"   Sai số bước 4: |x_4 - x*| <= |x_4 - x_3| = {abs(xs[4] - xs[3]):.3e}"
          f"  (đúng vì (M - m)/m = {(M - m) / m:.3f} <= 1)")
    print(f"   Chặt hơn (công thức 3.22, P4): {false_position_error(xs[4], xs[3], m, M):.3e}")

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

"""Bài 3- BÀI NGOÀI (đề của cô): lãi suất vay trả góp.

ĐỀ BÀI:
    Một người vay P = 500 triệu đồng, trả đều A = 12 triệu đồng mỗi tháng trong
    N = 60 tháng. Lãi suất tháng r thỏa mãn
        f(r) = A * (1 - (1 + r)^(-N)) / r - P = 0.
    Tìm r với sai số 10^-8 bằng cả hai phương pháp (đáp số r ≈ 1,2833 %/tháng).
    Khảo sát r khi A thay đổi từ 9 đến 15 triệu đồng; nhận xét điều gì xảy ra
    khi A * N <= P.
"""
from bisection import bisection                 # P2
from false_position import false_position       # P3
from isolate_roots import isolate_roots         # P1

P = 500.0      # số tiền vay (triệu đồng)
N = 60         # số tháng
EPS = 1e-8     # sai số yêu cầu
N_MAX = 200    # số bước lặp tối đa
PHUONG_PHAP = [("Chia đôi", bisection), ("Dây cung", false_position)]

def f_lai_suat(r, A, P=P, N=N):
    """Hàm của bài toán: f(r) = A * (1 - (1 + r)^(-N)) / r - P.

    Ý nghĩa: A*(1 - (1+r)^(-N))/r là tổng giá trị hiện tại của N lần trả A,
    nên f(r) = 0 khi giá trị đó đúng bằng số tiền vay P.

    Require: r > 0 (r = 0 làm mẫu số bằng 0); A, P > 0; N nguyên dương.
    Ensure: trả về f(r) (float).
    Thuật toán: không phải thuật toán bài giảng, đây là hàm mục tiêu của bài toán.
    """
    if r <= 0:  # điều kiện áp dụng: chia cho r nên cần r > 0
        raise ValueError("Cần r > 0.")
    return A * (1 - (1 + r) ** (-N)) / r - P

def giai(A, phuong_phap):
    """Tìm lãi suất tháng r ứng với tiền trả A bằng phương pháp được chọn.

    f giảm theo r, f(0+) = A*N - P và f(r) -> -P < 0 khi r lớn. Nên khi A*N > P, f đổi dấu
    đúng một lần trong [1e-6; 1]. Khi A*N <= P thì f(r) < 0 với mọi r > 0: không có
    lãi suất dương và isolate_roots không tìm thấy khoảng nào.

    Require:
        A > 0
        phuong_phap : hàm có dạng (f, a, b, eps, n_max) -> (x, k, err), ví dụ
                      bisection (P2) hoặc false_position (P3)
    Ensure:
        Trả về (r, k, err): lãi suất tháng, số bước lặp, sai số.
    Raises:
        ValueError nếu A <= 0 hoặc A*N <= P (không có nghiệm dương).
    Thuật toán: tách nghiệm (P1), rồi chia đôi (P2) hoặc dây cung (P3).
    """
    if A <= 0:
        raise ValueError("A phải dương.")

    def f(r):
        return f_lai_suat(r, A)

    cac_khoang = isolate_roots(f, 1e-6, 1.0, 0.01)  # tách nghiệm (P1)
    if len(cac_khoang) == 0:  # tình huống biên: A*N <= P
        raise ValueError("A*N <= P: tổng tiền trả không vượt số tiền vay, không có lãi suất dương.")
    a, b = cac_khoang[0]
    return phuong_phap(f, a, b, eps=EPS, n_max=N_MAX)

def main():
    """Chạy bài toán: A = 12, khảo sát A từ 9 đến 15, và tình huống A*N <= P."""
    # --- Phần 1: A = 12, thay phương pháp là xong ---
    print("[1] A = 12:")
    for ten, phuong_phap in PHUONG_PHAP:
        r, k, err = giai(12, phuong_phap)
        print(f"    {ten:9s}: r = {100 * r:.6f} %/tháng, k = {k}, err = {err:.2e}")
    print("    Đáp số của đề: r ≈ 1,2833 %/tháng")

    # Phần 2: khảo sát A từ 9 đến 15 
    print("\n[2] Khảo sát r khi A thay đổi từ 9 đến 15:")
    print(f"{'A':>6} {'r chia đôi (%)':>15} {'k':>3} {'r dây cung (%)':>15} {'k':>3}")
    for i in range(13):
        A = 9 + 0.5 * i
        r_cd, k_cd, _ = giai(A, bisection)
        r_dc, k_dc, _ = giai(A, false_position)
        print(f"{A:6.1f} {100 * r_cd:15.6f} {k_cd:3d} {100 * r_dc:15.6f} {k_dc:3d}")

    # Phần 3: tình huống A*N <= P 
    print(f"\n[3] Tình huống A*N <= P (P = {P:.0f}, N = {N}, nên A <= {P / N:.3f}):")
    for A in (8.0, P / N):
        try:
            giai(A, bisection)
        except ValueError as loi:
            print(f"    A = {A:.3f} (A*N = {A * N:.1f}): báo lỗi -> {loi}")

if __name__ == "__main__":
    main()

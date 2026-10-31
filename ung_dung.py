"""Bài ứng dụng 1 - BÀI GIẢNG: tính căn bậc hai của 2.

ĐỀ BÀI
    Cho phương trình f(x) = x^2 - 2 = 0.
    a) Tìm nghiệm gần đúng trong khoảng phân ly (1; 2) bằng phương pháp chia đôi
       với 6 bước lặp. Đánh giá sai số của nghiệm.
    b) Nếu muốn sai số là eps = 10^-5 thì cần tối thiểu bao nhiêu bước lặp?
    Đáp số của sách: x6 = 1,4140625, sai số 0,0078125; cần ít nhất 16 bước.
    Nhóm làm thêm c): giải bằng cả dây cung, so sánh số bước lặp với chia đôi.

Chỉ GỌI các hàm của nhóm: P2 bisection, P3 false_position, P4 bisection_steps.
Không dùng numpy.linalg, scipy... (R3).

Chạy:  python ung_dung_bai1.py
"""
import math

# Tên module theo file Test của nhóm. 
from bisection import bisection                 # P2
from error_estimation import bisection_steps     # P4 (file error_estimation.py)
from false_position import false_position       # P3

A, B = 1.0, 2.0                      # khoảng phân ly cho sẵn trong sách: f(1)*f(2) = (-1)*2 < 0
DANH_SACH_EPS = [1e-2, 1e-5, 1e-8]   # các mức sai số để so sánh
N_MAX = 200                          # số bước lặp tối đa

def f_bai1(x):
    """Hàm của Bài 1: f(x) = x^2 - 2.

    Require: x là số thực.
    Ensure: trả về f(x). Nghiệm dương của f chính là căn bậc hai của 2.
    Thuật toán: không phải thuật toán bài giảng, đây là hàm mục tiêu của bài toán.
    """
    return x * x - 2

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
    """Chạy Bài 1 và in kết quả."""
    print("Bài 1 (Ví dụ 3.1): x^2 - 2 = 0 trên (1; 2), nghiệm là căn bậc hai của 2")

    # Câu a: chia đôi với eps = 0,01 thì dừng sau 6 bước (sai số 1/2^7 = 0,0078125 <= 0,01)
    x, k, err = bisection(f_bai1, A, B, eps=0.01, n_max=N_MAX)
    print(f"  a) Chia đôi: x = {x}, k = {k}, sai số = {err}"
          "  (sách: x6 = 1,4140625, sai số 0,0078125)")

    # Câu b: số bước tối thiểu để đạt eps = 1e-5 theo công thức (P4)
    print(f"  b) Số bước tối thiểu với eps = 1e-5 (P4): {bisection_steps(A, B, 1e-5)}  (sách: 16)")

    # Câu c (làm thêm): so sánh chia đôi và dây cung
    print("  c) So sánh chia đôi và dây cung:")
    in_bang(bang_so_sanh(f_bai1, A, B, DANH_SACH_EPS, N_MAX))
    print(f"  Đối chiếu: căn bậc hai của 2 = {math.sqrt(2):.8f}")

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


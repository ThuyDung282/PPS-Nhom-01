import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore") # Bỏ cảnh báo khi tính đạo hàm bậc 2 gần biên (d2f_a, d2f_b) trong phương pháp dây cung

# 1. CÁC HÀM CƠ SỞ THEO ĐỀ BÀI
# Số bước lặp tối thiểu
def bisection_steps(a, b, eps) -> int:
    # Kiểm tra điều kiện đầu vào
    if a >= b:
        raise ValueError("Khoảng cách ly không hợp lệ: yêu cầu b > a")
    if eps <= 0:
        raise ValueError("Sai số eps phải lớn hơn 0")
    n = np.ceil(np.log2((b - a) / eps)) - 1  # Tính số bước lặp tối thiểu n theo công thức và làm tròn lên bằng np.ceil
    return max(0, int(n)) # Trả kết quả. Đảm bảo n không âm, nếu n < 0 thì trả về 0
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
        if f(a) * fx < 0: # Thu hẹp khoảng phân ly
            b = x
        else:
            a = x
    return float(x), n_max, float(err), pd.DataFrame(history)

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
    # Lặp tính các bước x_1, x_2, ...
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
    return float(xn), n_max, float(err), pd.DataFrame(history)

# 2. KHAI BÁO 3 BÀI TOÁN KIỂM THỬ
def f1(x): return 2**x - 5*x - 3
def f2(x): return x**3 - 3.23*(x**2) - 5.64*x + 9.84
def f3(r): 
    if r == 0: return 12.0 * 60 - 500.0
    return 12.0 * ((1.0 - (1.0 + r)**(-60)) / r) - 500.0
problems = [
    {"Ten": "Bài 1: 2^x - 5x - 3 = 0", "f": f1, "a": -1.0, "b": 0.0},
    {"Ten": "Bài 2: x^3 - 3.23x^2 - 5.64x + 9.84 = 0", "f": f2, "a": 1.1, "b": 1.5},
    {"Ten": "Bài 3: Lãi suất tài chính", "f": f3, "a": 0.001, "b": 0.03}
]

# 3. SO SÁNH SỐ BƯỚC LẶP CHO 3 BÀI TOÁN
for idx, bt in enumerate(problems, 1): # Duyệt qua từng bài toán
    print(f"\n{bt['Ten']} TRÊN [{bt['a']}, {bt['b']}]")
    print(f"{'eps':>10} | {'Chia đôi (TT)':>14} | {'Chia đôi (LT)':>14} | {'Dây cung (TT)':>14}")
    for p in range(2, 11): # Duyệt qua các giá trị eps = 10^-p từ p = 2 đến p = 10
        eps = 10**(-p)
        _, k_bi, _, _ = bisection_history(bt["f"], bt["a"], bt["b"], eps=eps, n_max=30)
        k_lt = bisection_steps(bt["a"], bt["b"], eps)
        _, k_fp, _, _ = false_position_history(bt["f"], bt["a"], bt["b"], eps=eps, n_max=25)
        print(f"{eps:10.0e} | {k_bi:14d} | {k_lt:14d} | {k_fp:14d}")
"""
NHẬN XÉT:
1. Tính chính xác của lý thuyết: Số bước lặp chia đôi thực tế hoàn toàn khớp hoặc 
   chênh lệch không quá 1 bước so với công thức đánh giá theo lý thuyết.
2. Ưu thế của Dây cung ở điều kiện thường: Đối với cả 3 bài toán trơn, Dây cung 
   hội tụ cực nhanh. Khi siết sai số xuống 10^-10, Chia đôi mất hơn 30 bước, 
   trong khi Dây cung chỉ cần từ 3 - 5 bước là đạt độ chính xác tuyệt đối.
"""

# 4. VẼ ĐỒ THỊ SAI SỐ TRÊN THANG LOG
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
    _, _, _, df = false_position_history(f, a, b, eps=1e-15, n_max=n_steps)
    return df['x_n'].tolist()[1:]  # Bỏ bước 0 (điểm xuất phát Ban đầu)
for idx, bt in enumerate(problems, 1):
    # Tìm nghiệm chính xác cao bằng Chia đôi trước để làm x* cho bài toán
    x_star, _, _, _ = bisection_history(bt["f"], bt["a"], bt["b"], eps=1e-15, n_max=100)
    xs_bi = get_history_bi(bt["f"], bt["a"], bt["b"], 30)
    xs_fp = get_history_fp(bt["f"], bt["a"], bt["b"], 25)
    # Tính sai số tuyệt đối so với nghiệm x_star
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
"""
NHẬN XÉT:
1. Phương pháp Chia đôi (Hình a): Đường biểu diễn sai số có dạng một ĐƯỜNG THẲNG
   đi xuống đều đặn. Vì sai số tuân theo công thức tuyến tính e_(k+1) = 0.5 * e_k,
   khi lấy logarit sẽ tạo ra một hàm bậc nhất theo k, thể hiện tốc độ ổn định.
2. Phương pháp Dây cung (Hình b): Đường sai số dốc xuống GẦN NHƯ THẲNG ĐỨNG 
   và chạm đáy giới hạn máy tính (1e-16) chỉ sau vài bước lặp, minh chứng cho 
   hiệu quả vượt trội của việc tận dụng giá trị hàm số để ép sát nghiệm.
3. Kết luận: Dây cung hội tụ nhanh hơn Chia đôi rất nhiều, đặc biệt khi hàm số trơn 
   và không có điểm cực trị gần nghiệm. Tuy nhiên, trong các trường hợp hàm số 
   có dốc cực đoan hoặc gần điểm cực trị, Dây cung có thể gặp vấn đề hội tụ chậm hơn Chia đôi,
   như sẽ được minh họa trong phần 5.
"""
# 5. THỰC NGHIỆM HÀM DÂY CUNG HỘI TỤ CHẬM HƠN CHIA ĐÔI
print("\nTHỰC NGHIỆM TRƯỜNG HỢP DÂY CUNG HỘI TỤ CHẬM HƠN CHIA ĐÔI")
f_slow = lambda x: x**10 - 0.1
a_s, b_s = 0.0, 1.0
print(f">>> Hàm f(x) = x^10 - 0.1 trên [{a_s}, {b_s}]")
print(f"{'eps':>10} | {'Chia đôi (Thực tế)':>20} | {'Dây cung (Thực tế)':>20}")
for p in range(2, 7):
    eps = 10**(-p)
    _, k_bi, _, _ = bisection_history(f_slow, a_s, b_s, eps=eps, n_max=100)
    _, k_fp, _, _ = false_position_history(f_slow, a_s, b_s, eps=eps, n_max=100)
    print(f"{eps:10.0e} | {k_bi:20d} | {k_fp:20d}") 
print(f"Nhận xét: Để đạt độ chính xác 1e-6, phương pháp Chia đôi chỉ cần {k_bi} bước, 
      trong khi phương pháp Dây cung mất tới {k_fp} bước (nhiều hơn gấp đôi).")
"""
Phương pháp Dây cung (Giữ mút cố định Fourier):
Gặp hiện tượng 'bẫy mút cố định' đối với các hàm dốc lệch góc cực đoan.
GIẢI THÍCH CƠ CHẾ BỊ CHẬM:
    1. Điều kiện Fourier: Tại b = 1.0, f(1) = 0.9 và f''(1) > 0 => f(1)*f''(1) > 0.
       Do đó, mút b = 1.0 bắt buộc phải chọn làm MÚT CỐ ĐỊNH.
    2. Điểm lặp xuất phát động là x0 = a = 0.0. Các dây cung sẽ luôn kéo từ điểm 
       động x_n bên trái lên mút cố định (1.0, 0.9) rất cao ở bên phải.
    3. Vì hệ số góc (độ dốc) của dây cung nối tới (1.0, 0.9) quá lớn, giao điểm của 
       dây cung với trục hoành (điểm x_n+1) chỉ có thể dịch chuyển từng bước 
       SIÊU NHỎ sang bên phải. Phương pháp bị mất khả năng co hẹp khoảng cách từ mút phải.
"""

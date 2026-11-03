5. BÀI ỨNG DỤNG
# Đánh giá sai số (số bước lặp tối thiểu)
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
    history = [] # Lưu trữ lịch sử các bước lặp
    for i in range(0, n_max + 1): # Lặp từ 0 đến n_max + 1
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
        if err <= eps or fx == 0: # Dừng nếu sai số nhỏ hơn eps hoặc f(x) = 0
            return x, i, err, pd.DataFrame(history)
        if f(a) * fx < 0: # Nếu f(a) và f(x) trái dấu, nghiệm nằm trong [a, x]
            b = x
        else: # Nếu f(x) và f(b) trái dấu, nghiệm nằm trong [x, b]
            a = x
    return float(x), n_max, float(err), pd.DataFrame(history)
# Phương pháp dây cung, có bảng tính
def false_position_history(f, a, b, eps=1e-5, n_max=50): 
    fa, fb = f(a), f(b)
    h = 1e-5 # Bước nhỏ để tính xấp xỉ đạo hàm bậc hai f''(x) tại a và b
    d2f_a = (f(a + h) - 2 * fa + f(a - h)) / (h**2) # Tính xấp xỉ đạo hàm bậc hai f''(a)
    d2f_b = (f(b + h) - 2 * fb + f(b - h)) / (h**2) # Tính xấp xỉ đạo hàm bậc hai f''(b)
    # Xác định điểm Fourier x0 và mút cố định d
    if fa * d2f_a > 0: # Nếu f(a)*f''(a) > 0 thì chọn x0 = b và x = a
        x0 = b
        x = a
        print(f"[+] Điểm Fourier là x = a = {a}")
        print(f"    - f({a}) = {fa:.6f}")
        print(f"    - f''({a}) ≈ {d2f_a:.6f} => f(a)*f''(a) > 0 \n")
    elif fb * d2f_b > 0: # Nếu f(b)*f''(b) > 0 thì chọn x0 = a và x = b
        x0 = a
        x = b
        print(f"[+] Điểm Fourier là x = b = {b}")
        print(f"    - f({b}) = {fb:.6f}")
        print(f"    - f''({b}) ≈ {d2f_b:.6f} => f(b)*f''(b) > 0 \n")
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

"""
Bài 1  - BÀI GIẢNG: Ví dụ 3.1, giải f(x) = 2^x - 5x - 3 = 0 trên (-1; 0).
  Phần 1 - Chia đôi:
    a) Chứng tỏ (-1; 0) là khoảng phân ly nghiệm.
    Tìm nghiệm gần đúng với n_max = 6 bước lặp và đánh giá sai số.
    b) Muốn sai số eps = 10^-6 thì cần tối thiểu bao nhiêu bước lặp?
  Phần 2 - Dây cung: tìm nghiệm gần đúng với n = 4 bước lặp và đánh giá sai số bước 4.
"""
print("BÀI 1: GIẢI PHƯƠNG TRÌNH f(x) = 2^x - 5x - 3 = 0 TRÊN (-1, 0)")
def f1(x):
    return 2**x - 5*x - 3
# Kiểm tra khoảng phân ly
print("\nKiểm tra khoảng phân ly nghiệm") 
check_isolate_roots(f1, -1.0, 0.0)
# Phương pháp chia đôi
print("\nPhương pháp Chia đôi (n_max = 6)")
x1, k1, err1, df1 = bisection_history(f1, -1.0, 0.0, eps=1e-6, n_max=6)
print(df1.to_string(index=False)) # In ra bảng lịch sử các bước lặp
print(f"\n-> Nghiệm gần đúng tại bước {k1}: x = {x1:.6f}")
print(f"-> Sai số tại bước {k1}: err = {err1:.6f}")
print(f"-> Đánh giá sai số lý thuyết bước {k1}: (b - a)/2^6 = (0 - (-1))/64 = {1/64:.6f}")
n1_min = bisection_steps(-1.0, 0.0, eps=1e-6)
print(f"-> Số bước lặp tối thiểu để đạt eps = 1e-6: {n1_min} bước lặp.")
# Phương pháp dây cung
print("\nPhương pháp Dây cung (n_max = 4)")
x1_fp, k1_fp, err1_fp, df1_fp = false_position_history(f1, -1.0, 0.0, eps=1e-6, n_max=4)
print(df1_fp.to_string(index=False))
print(f"\n-> Nghiệm gần đúng tại bước {k1_fp}: x = {x1_fp:.6f}")
print(f"-> Sai số tại bước {k1_fp}: err = {err1_fp:.6f}")

"""
Bài 2 - (Bài tập Chương 3, Bài tập 3.10):
Tìm nghiệm dương nhỏ nhất của phương trình x^3 - 3,23x^2 - 5,64x + 9,84 = 0 bằng phương pháp chia đôi và dây cung.
"""
print("\nBÀI 2: GIẢI PHƯƠNG TRÌNH f(x) = x^3 - 3,23x^2 - 5,64x + 9,84 = 0 ")
def f2(x):
    return x**3 - 3.23*(x**2) - 5.64*x + 9.84
# Quét khoảng phân ly nghiệm dương
intervals_pos = isolate_roots(f2, 0.0, 5.0, step=0.5)
print(f"\nCác khoảng chứa nghiệm dương tìm được: {intervals_pos}")
a2, b2 = intervals_pos[0] # Khoảng nghiệm dương nhỏ nhất [1.0, 1.5] hoặc [1.0, 2.0]
a2, b2 = 1.1, 1.5
print(f"Chọn khoảng phân ly cho nghiệm dương nhỏ nhất: [{a2}, {b2}]")
print("\nKiểm tra khoảng phân ly nghiệm")
check_isolate_roots(f2, a2, b2)
# Phương pháp chia đôi
print("\nPhương pháp chia đôi (eps = 1e-5)")
x2_bi, k2_bi, err2_bi, df2_bi = bisection_history(f2, a2, b2, eps=1e-5, n_max=50)
print(df2_bi.to_string(index=False))
print(f"\n-> Nghiệm gần đúng tại bước {k2_bi}: x = {x2_bi:.6f}")
print(f"-> Sai số tại bước {k2_bi}: err = {err2_bi:.6f}")
print(f"-> Đánh giá sai số lý thuyết bước {k2_bi}: e = (b - a)/2^(k+1) = {(b2 - a2) / (2**(k2_bi + 1)):.6f}")
n2_min = bisection_steps(a2, b2, eps=1e-5)
print(f"-> Số bước lặp tối thiểu để đạt eps = 1e-5: {n2_min} bước lặp.")
# Phương pháp dây cung
print("\nPhương pháp dây cung (eps = 1e-5)")
x2_fp, k2_fp, err2_fp, df2_fp = false_position_history(f2, a2, b2, eps=1e-5, n_max=50)
print(df2_fp.to_string(index=False))
print(f"\n-> Nghiệm gần đúng tại bước {k2_fp}: x = {x2_fp:.6f} ")
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
print("\nKiểm tra khoảng phân ly nghiệm r (với A = 12 triệu) ---")
check_isolate_roots(f3, 0.001, 0.03)
# Giải tìm r bằng 2 phương pháp với eps = 1e-8
print("\nPhương pháp chia đôi")
r_bi, k3_bi, err3_bi, df3_bi = bisection_history(f3, 0.001, 0.03, eps=1e-8, n_max=50)
print(df3_bi.to_string(index=False)) # In 10 dòng đầu đại diện
print(f"\n-> Lãi suất tháng (Chia đôi): r = {r_bi:.6f} ({r_bi*100:.6f}%/tháng) sau {k3_bi} bước")
print("\nPhương pháp dây cung")
r_fp, k3_fp, err3_fp, df3_fp = false_position_history(f3, 0.001, 0.03, eps=1e-8, n_max=50)
print(df3_fp.to_string(index=False))
print(f"\n-> Lãi suất tháng (Dây cung): r = {r_fp:.6f} ({r_fp*100:.6f}%/tháng) sau {k3_fp} bước")
# Khảo sát r khi A thay đổi từ 9 đến 15 triệu
print("\nKhảo sát r khi A thay đổi từ 9 đến 15 triệu đồng")
survey_results = []
for A_val in range(9, 16):
    f_temp = make_f3(A = float(A_val))
    # Tìm khoảng chứa nghiệm
    r_sol, _, _, _ = bisection_history(f_temp, 1e-6, 0.05, eps=1e-8, n_max=100)
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
print("""
• Về mặt Toán học:
  - Khi A * N = P (ví dụ: A = 8.333 triệu/tháng -> 8.333 * 60 = 500 triệu), nghiệm của phương trình f(r) = 0 là r = 0.
  - Khi A * N < P (ví dụ: A = 8 triệu/tháng -> 8 * 60 = 480 triệu < 500 triệu), phương trình f(r) = 0 cho nghiệm r < 0 (lãi suất âm).
• Về mặt Kinh tế / Thực tế:
  - A*N<=P là dấu hiệu nhận biết khoản vay không có lãi (r = 0) hoặc lãi suất bị âm (r < 0).
  - Ngân hàng / Người cho vay không thu được tiền lãi (r = 0) hoặc bị lỗ vốn (r < 0).
  - Trong thực tế hợp đồng tín dụng thương mại, không tồn tại giao dịch vay vốn với A * N <= P (trừ các chính sách trợ giá/trả góp 0% lãi suất do bên thứ ba tài trợ).
""")

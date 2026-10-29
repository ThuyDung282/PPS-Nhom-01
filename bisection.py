import warnings

def bisection(f, a, b, eps=1e-5, n_max=50):
    """Giải phương trình f(x) = 0 bằng phương pháp chia đôi (theo Algorithm 4 trong bài giảng)

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

    Lưu ý: Nếu a >= b, eps <= 0, n_max < 0, hoặc f(a)f(b) > 0
            (khi đó (a; b) không phải khoảng phân ly nghiệm)

           Nếu hết n_max bước mà chưa đạt sai số eps thì cảnh báo người dùng
    """
    # Xử lý tình huống biên (R6)
    if not a < b:
        raise ValueError("Cần a < b.")
    if eps <= 0:
        raise ValueError("Sai số eps phải dương.")
    if n_max < 0:
        raise ValueError("n_max phải không âm.")

    fa = f(a)
    fb = f(b)
    if fa == 0:                      # Đầu mút a đã là nghiệm
        return float(a), 0, 0.0
    if fb == 0:                      # Đầu mút b đã là nghiệm
        return float(b), 0, 0.0
    if fa * fb > 0:
        raise ValueError(
            "f(a)f(b) > 0: (a; b) không phải khoảng phân ly nghiệm.")

    # Xây dựng thuật toán - Algorithm 4: Phương pháp chia đôi
    
    i = 0
    err = (b - a) / 2                   # dòng 1 - Thuật toán 4: err ← (b - a)/2
    x = (a + b) / 2                     # Để hàm vẫn trả kết quả hợp lệ khi eps lớn hơn nửa độ rộng khoảng ban đầu
    
    while err > eps and i < n_max + 1:  # dòng 2 - Thuật toán 4: while err > eps and i < n_max + 1
        x = (a + b) / 2                 # dòng 3 - Thuật toán 4: x ← (a + b)/2
        fx = f(x)                       # dòng 4 - Thuật toán 4: tính f(x)
        if fa * fx == 0:                # dòng 5 - Thuật toán 4: if f(a)f(x) = 0
            return x, i + 1, 0.0        # dòng 6 - Thuật toán 4: dừng, x là nghiệm
        # dòng 7 - Thuật toán 4: else if f(a)f(x) > 0
        elif fa * fx > 0: 
            a = x                       # dòng 8 - Thuật toán 4: a ← x        
            fa = fx                     # lưu f(a) mới
            err = (b - a) / 2           # dòng 9 - Thuật toán 4: err ← (b - a)/2
        # dòng 10 - Thuật toán 4: else f(a)f(x) < 0
        else:                       
            b = x                       # dòng 11 - Thuật toán 4: b ← x
            err = (b - a) / 2           # dòng 12 - Thuật toán 4: err ← (b - a)/2
        i = i + 1                       # dòng 13-14 - Thuật toán 4: end if và gán i ← i + 1
    # dòng 15 - Thuật toán 4: end while (kết thúc vòng lặp)
    # Cảnh báo khi hết n_max mà chưa đạt eps
    if err > eps:
        warnings.warn(
            f"Hết n_max={n_max} bước nhưng chưa đạt sai số eps={eps} "
            f"(err={err}).",
            UserWarning,
        )
    x = (a + b) / 2                 # trung điểm khoảng sau khi thu hẹp
    err = (b - a) / 2
    return x, i, err                     # dòng 16 - Thuật toán 4: return x, i, err

import numpy as np
# PP CHIA ĐÔI
def bisection_steps(a, b, eps) -> int:
    """
    Tính số bước lặp lý thuyết tối thiểu của phương pháp chia đôi để đạt sai số eps.

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
    # dong 1 - Kiem tra dieu kien dau vao
    if a >= b:                                 # Kiểm tra nếu biên trái >= biên phải -> Khoảng cách ly không hợp lệ
        raise ValueError("Khoảng cách ly không hợp lệ: yêu cầu b > a")  # Báo lỗi nếu khoảng không hợp lệ
    if eps <= 0:                               # Kiểm tra sai số eps phải là số dương
        raise ValueError("Sai số eps phải lớn hơn 0")   # Báo lỗi nếu eps <= 0

    # dong 2 - Tinh so buoc lap theo cong thuc
    n = np.ceil(np.log2((b - a) / eps)) - 1             # Tính n theo công thức log2 và làm tròn lên bằng np.ceil, sau đó trừ 1

    # dong 3 - Tra ve ket qua
    return max(0, int(n))                               # Đảm bảo n không bị âm (max với 0) và ép kiểu về int trước khi trả về
# PP DÂY CUNG
def false_position_error(x_k, x_prev, m1, M1) -> float:
    """
    Đánh giá cận trên sai số của phương pháp dây cung qua 2 bước lặp liên tiếp.

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
    # dong 1 - Kiem tra dieu kien m1, M1
    if m1 <= 0:                                            #Kiểm tra cận dưới đạo hàm m1 phải >0
        raise ValueError("Cận m1 phải lớn hơn 0")          #Báo lỗi nếu m1 <=0
    if M1 < m1:                                            #Kiểm tra cận trên đạo hàm M1 phải lớn hơn hoặc bằng m1
        raise ValueError("Cận M1 phải lớn hơn hoặc bằng m1")      #Báo lỗi nếu M1 < m1

    # dong 2 - Tinh sai so theo cong thuc
    err = ((M1 - m1) / m1) * abs(x_k - x_prev)                #Áp dụng công thức sai số: ((M1-m1)/m1)*|x_k-x_prev|
    #dong 3 - Tra ve ket qua
    return float(err)                                      #Ép kiểu dữ liệu về float và trả về kết quả sai số 

import math

def isolate_roots(f, a, b, step) -> list[tuple[float, float]]:
  """Thuật toán tìm các khoảng cách ly (phân ly) nghiệm bằng phương pháp quét dấu (Incremental Search).

  Require (Đầu vào):
      - f: Hàm số f(x) liên tục cần tìm khoảng cách ly nghiệm (Callable[[float],
      float]).
      - a: Biên trái của đoạn xét [a, b] (float).
      - b: Biên phải của đoạn xét [a, b] (float), với a < b.
      - step: Bước quét h > 0 (float).

  Ensure (Đầu ra):
      - Danh sách các tuple (x_i, x_{i+1}) chứa các khoảng cách ly nghiệm thoả
      mãn f(x_i) * f(x_{i+1}) <= 0.

  Xử lý tình huống biên (R6):
      - Báo lỗi ValueError nếu a >= b hoặc step <= 0.
  """
  # dong 1 - Kiem tra dieu kien bien cua tham so dau vao (R6)
  if a >= b:
    raise ValueError("Lỗi biên: Biên trái 'a' phải nhỏ hơn biên phải 'b'.")
  if step <= 0:
    raise ValueError("Lỗi biên: Bước quét 'step' phải là số thực dương (> 0).")

  # dong 2 - Khoi tao danh sach chua cac khoang cach ly nghiem
  intervals = []

  # dong 3 - Khoi tao diem quet ban dau
  x_curr = float(a)

  # dong 4 - Vong lap quet tren doan [a, b] voi buoc h = step
  while x_curr < b:
    # dong 5 - Tinh diem tiep theo (khong vuot qua ranh gioi b)
    x_next = min(x_curr + step, float(b))

    # dong 6 - Tinh gia tri ham so tai hai dau khoang con [x_curr, x_next]
    y_curr = f(x_curr)
    y_next = f(x_next)

    # dong 7 - Kiem tra dieu kien doi dau hoac nghiem dung
    if y_curr * y_next < 0:
      # dong 8 - Neu f(x) doi dau, ghi nhan khoang (x_curr, x_next)
      intervals.append((x_curr, x_next))
    elif y_curr == 0:
      # dong 9 - Neu x_curr la nghiem dung, ghi nhan khoang chot [x_curr, x_curr]
      intervals.append((x_curr, x_curr))

    # dong 10 - Chuyen sang diem quet tiep theo
    x_curr = x_next

  # dong 11 - Kiem tra rieng diem cuoi cung b de tranh bo sot nghiem o bien phai
  if f(b) == 0:
    val_b = float(b)
    if not intervals or intervals[-1] != (val_b, val_b):
      intervals.append((val_b, val_b))

  # dong 12 - Tra ve danh sach cac khoang cach ly tim duoc
  return intervals
  
  def check_isolate_roots(f, a, b, step=0.01) -> bool:
    """
    Kiểm tra (a, b) có phải là khoảng phân ly nghiệm của f(x) hay không.
    Điều kiện:
      1. f(a) * f(b) < 0 (f(x) có nghiệm trong khoảng)
      2. f'(x) không đổi dấu trên (a, b) (hàm số đơn điệu, đảm bảo nghiệm duy nhất)
    """
    # dong 13 - Tinh gia tri ham so tai hai dau mut a va b
  fa, fb = f(a), f(b)

  # dong 14 - Kiem tra dieu kien co nghiem: f(a) * f(b) < 0
  if fa * fb >= 0:
    print(f"[-] LOẠI ({a}, {b}): f({a}) * f({b}) >= 0 (Không đảm bảo có nghiệm).")
    return False

  # dong 15 - Chon buoc vi phan h = 1e-5 de tinh xap xi dao ham
  h = 1e-5

  # dong 16 - Tao mang cac diem x tren khoang (a, b) voi buoc nhay step
  xs = np.arange(a, b, step)

  # dong 17 - Tinh xap xi dao ham f'(x) tai moi diem x bang sai phan trung tam
  df_vals = [(f(x + h) - f(x - h)) / (2 * h) for x in xs]

  # dong 18 - Kiem tra dao ham f'(x) co luon duong (> 0) tren toan khoang hay khong
  is_always_positive = all(d > 0 for d in df_vals)

  # dong 19 - Kiem tra dao ham f'(x) co luon am (< 0) tren toan khoang hay khong
  is_always_negative = all(d < 0 for d in df_vals)

  # dong 20 - Neu f'(x) luon duong hoac luon am, ham so don dieu => nghiem duy nhat
  if is_always_positive or is_always_negative:
    print(f"[+] HỢP LỆ: ({a}, {b}) là khoảng phân ly nghiệm.")
    print(f"    - Có nghiệm: f({a}) = {fa:.6f}, f({b}) = {fb:.6f} (Trái dấu)")
    print(f"    - Nghiệm duy nhất: f'(x) không đổi dấu trên ({a}, {b}).")
    return True
  else:
    # dong 21 - Neu f'(x) doi dau, ham khong don dieu => canh bao co the co > 1 nghiem
    print(
        f"[!] CẢNH BÁO: f(x) đổi dấu trên ({a}, {b}) nhưng f'(x) không đơn"
        " điệu (có thể có >1 nghiệm)."
    )
    return False

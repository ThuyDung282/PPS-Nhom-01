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

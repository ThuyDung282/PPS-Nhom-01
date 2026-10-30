
import unittest
import math
import warnings

# Import các hàm xay dung
from isolate_roots import isolate_roots  #  Tìm khoảng phân ly nghiệm
from bisection import bisection          #  Phương pháp chia đôi
from false_position import false_position  #  Phương pháp dây cung


# PHẦN 1. XÂY DỰNG CÁC HÀM SỐ THEO YÊU CẦU ĐỀ BÀI

# Hàm số thứ nhất: f(x) = 2^x - 5x - 3
def f1(x):
    return 2**x - 5*x - 3


# Hàm số thứ hai: f(x) = 5x^3 - 20x + 3
def f2(x):
    return 5*x**3 - 20*x + 3


# Nghiệm chính xác của f1 nằm trong khoảng (-1, 0). Giải phương trình 2^x - 5x - 3 = 0 bằng công thức nghiệm.
ROOT1 = (5 - math.sqrt(37)) / 2


# Các nghiệm gần đúng của f2 theo đề bài. Dùng để so sánh kết quả do các phương pháp số tìm được.
ROOTS2 = [
    -2.071157,
    0.150858,
    1.920299
]


# Ba khoảng phân ly nghiệm của f2. Mỗi khoảng phải có f(a)*f(b) < 0.
BRACKETS2 = [
    (-2.1, -2.0),
    (0.1, 0.2),
    (1.9, 2.0)
]


# Các mức sai số cần khảo sát theo đề bài. Sai số càng nhỏ thì thường cần độ chính xác cao hơn.
EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


# PHẦN 2. XÂY DỰNG CÁC TEST KIỂM THỬ

class TestFunctionsFromAssignment(unittest.TestCase):

    # TEST 1: Kiểm tra f1 đổi dấu trên khoảng (-1, 0). 
    # Đây là điều kiện để áp dụng phương pháp chia đôi và phương pháp dây cung trong bài toán này.
    
    def test_f1_doi_dau(self):
        self.assertLess(f1(-1) * f1(0), 0)  # Điều kiện đổi dấu là f(a)*f(b) < 0.

    # TEST 2: Kiểm tra phương pháp chia đôi với f1.
    
    def test_f1_chia_doi(self):
        x, k, err = bisection(                 # Gọi hàm bisection để tìm nghiệm gần đúng x: nghiệm gần đúng; k: số bước lặp; err: sai số.
            f1, -1, 0, eps=1e-10, n_max=100
        )

        self.assertAlmostEqual(x, ROOT1, delta=1e-8) # Kiểm tra nghiệm tìm được gần ROOT1 trong phạm vi 1e-8.
        self.assertGreaterEqual(k, 1)                # kiểm tra số bước không âm và sai số không âm.
        self.assertGreaterEqual(err, 0)

    # --------------------------------------------------------
    # TEST 3: Kiểm tra phương pháp dây cung với f1. Tìm nghiệm trên khoảng (-1, 0), với sai số yêu cầu 1e-10.
  
    def test_f1_day_cung(self):
        x, k, err = false_position(
            f1, -1, 0, eps=1e-10, n_max=1000
        )

        self.assertAlmostEqual(x, ROOT1, delta=1e-8) # Kiểm tra nghiệm, số bước lặp và sai số trả về.
        self.assertGreaterEqual(k, 1)
        self.assertGreaterEqual(err, 0)

    
    # TEST 4: Kiểm tra SV1 có tìm được khoảng chứa cả ba nghiệm của f2 hay không.
  
    def test_f2_tim_du_ba_khoang(self):
        intervals = isolate_roots(f2, -3, 3, 0.1)    # isolate_roots quét khoảng [-3, 3] với bước 0.1.

        for root in ROOTS2:
            found = any(
                a <= root <= b for a, b in intervals               # Với mỗi nghiệm tham chiếu, kiểm tra xem có khoảng [a, b] nào chứa nghiệm đó không.
            )

            self.assertTrue(
                found,
                f"Không tìm được khoảng chứa nghiệm {root}"
            )

    # TEST 5: Kiểm tra hai phương pháp chia đôi và dây cung trên cả ba khoảng phân ly nghiệm của f2.
    # Sau đó tìm nghiệm bằng cả hai phương pháp và so sánh
    
    def test_f2_chia_doi_va_day_cung(self):
        for (a, b), root in zip(BRACKETS2, ROOTS2):
            with self.subTest(khoang=(a, b)):  #subTest giúp xác định khoảng nào gây lỗi nếu thất bại.

                # Kiểm tra khoảng có thỏa điều kiện đổi dấu.
                self.assertLess(f2(a) * f2(b), 0)

                # Tìm nghiệm bằng phương pháp chia đôi.
                xb, _, _ = bisection(
                    f2, a, b, eps=1e-10, n_max=100
                )

                # Tìm nghiệm bằng phương pháp dây cung.
                xf, _, _ = false_position(
                    f2, a, b, eps=1e-10, n_max=1000
                )

                # So sánh nghiệm tìm được với nghiệm tham chiếu, sai lệch cho phép là 2e-5.
                self.assertAlmostEqual(
                    xb, root, delta=2e-5
                )
                self.assertAlmostEqual(
                    xf, root, delta=2e-5 #sai lệch cho phép là 2e-5.
                )

    # TEST 6: Thực nghiệm so sánh số bước lặp.Chạy cả hai phương pháp trên f1 và ba khoảng của f2.
    # Với mỗi khoảng, lần lượt thử năm mức sai số. In ra bảng số bước lặp của chia đôi và dây cung.
    def test_bang_so_sanh_so_buoc(self):

        # In tiêu đề bảng kết quả thực nghiệm.
        print("\n\nSO SANH CHIA DOI VA DAY CUNG")
        print(
            f"{'Ham':<5} {'Khoang':<18} {'eps':<12} "
            f"{'k chia doi':<12} {'k day cung':<12}"
        )

        # Danh sách các trường hợp cần so sánh.
        cases = [
            ("f1", f1, -1, 0),
            ("f2a", f2, -2.1, -2.0),
            ("f2b", f2, 0.1, 0.2),
            ("f2c", f2, 1.9, 2.0)
        ]

        # Lặp qua từng hàm và từng khoảng nghiệm.
        for name, f, a, b in cases:

            # Thử lần lượt các mức sai số trong EPS_LIST.
            for eps in EPS_LIST:
                with self.subTest(
                    ham=name, eps=eps, khoang=(a, b)
                ):

                    # Tạm ẩn cảnh báo để bảng kết quả dễ đọc, không làm thay đổi thuật toán.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # Chạy phương pháp chia đôi.
                        _, kb, _ = bisection(
                            f, a, b, eps=eps, n_max=1000
                        )

                        # Chạy phương pháp dây cung.
                        _, kf, _ = false_position(
                            f, a, b, eps=eps, n_max=10000
                        )

                    # In số bước lặp của hai phương pháp.
                    print(
                        f"{name:<5} [{a:>5.1f},{b:>5.1f}] "
                        f"{eps:<12.0e} {kb:<12} {kf:<12}"
                    )

# PHẦN 3. CHẠY CÁC TEST
if __name__ == "__main__":
    # argv=['first-arg-is-ignored']: Không để unittest hiểu các tham số dòng lệnh bên ngoài là tên test cần chạy.
    # exit=False:Chạy test nhưng không đóng phiên Python sau khi hoàn tất.
    # verbosity=2: Hiển thị tên từng test và kết quả OK/FAIL.
    unittest.main(
        argv=['first-arg-is-ignored'],
        exit=False,
        verbosity=2
    )


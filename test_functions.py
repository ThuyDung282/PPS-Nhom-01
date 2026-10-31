
import unittest # Import thư viện unittest để xây dựng và thực hiện các bài kiểm thử tự động.
import math     # Import thư viện math để sử dụng các hàm toán học, ví dụ math.sqrt() tính căn bậc hai.
import warnings # Import thư viện warnings để quản lý và kiểm tra các cảnh báo phát sinh trong quá trình chạy chương trình.

# PHẦN 1. IMPORT CÁC HÀM CẦN KIỂM THỬ
from isolate_roots import isolate_roots     # Tìm các khoảng phân ly nghiệm bằng phương pháp quét dấu.
from bisection import bisection             # Tìm nghiệm gần đúng bằng phương pháp chia đôi.
from false_position import false_position   # Tìm nghiệm gần đúng bằng phương pháp dây cung.

# PHẦN 2. XÂY DỰNG DỮ LIỆU KIỂM THỬ
# Khai báo hàm số thứ nhất: f(x) = 2^x - 5x - 3.
def f1(x):
    return 2**x - 5*x - 3

# Khai báo hàm số thứ hai: f(x) = 5x^3 - 20x + 3.
def f2(x):
    return 5*x**3 - 20*x + 3

# Nghiệm tham chiếu gần đúng của hàm f1 trên khoảng (-1, 0). Giá trị này được dùng để so sánh với nghiệm do chương trình tìm được.
ROOT1 = -0.45399643850469835


# Danh sách ba nghiệm tham chiếu gần đúng của hàm f2. Dùng để kiểm tra độ chính xác của kết quả tìm nghiệm.
ROOTS2 = [
    -2.071157,
    0.150858,
    1.920299
]

# Khai báo ba khoảng phân ly nghiệm của hàm f2. Các khoảng này thỏa mãn điều kiện f(a)*f(b) < 0.
BRACKETS2 = [
    (-2.1, -2.0),
    (0.1, 0.2),
    (1.9, 2.0)
]

# Danh sách năm mức sai số cho phép cần khảo sát.Dùng để đánh giá sự thay đổi số bước lặp khi độ chính xác tăng lên.
EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

# PHẦN 3. KIỂM THỬ CÁC HÀM SỐ

class TestFunctions(unittest.TestCase):  
    # unittest.TestCase là lớp cơ sở hỗ trợ xây dựng các test case.

    def test_f1_doi_dau(self):    
        # Kiểm tra hàm f1 có đổi dấu trên khoảng [-1, 0] hay không.

        self.assertLess(f1(-1) * f1(0), 0)  
        # assertLess(a, b) kiểm tra a có nhỏ hơn b hay không.
        # Đây là điều kiện cần để áp dụng phương pháp chia đôi và dây cung.

    def test_f2_ba_khoang_doi_dau(self): 
        # Kiểm tra cả ba khoảng phân ly nghiệm của hàm f2.

        for a, b in BRACKETS2: 
            # Duyệt lần lượt từng khoảng trong danh sách BRACKETS2. a là đầu mút trái, b là đầu mút phải.
            with self.subTest(khoang=(a, b)): 
                # subTest tạo một trường hợp kiểm thử riêng cho từng khoảng. Giúp xác định chính xác khoảng nào xảy ra lỗi.
                self.assertLess(f2(a) * f2(b), 0)
                # Kiểm tra tích giá trị hàm số tại hai đầu mút nhỏ hơn 0.Nếu đúng, khoảng đang xét thỏa mãn điều kiện đổi dấu.

# PHẦN 4. KIỂM THỬ HÀM ISOLATE_ROOTS

class TestIsolateRoots(unittest.TestCase):# Khai báo lớp kiểm thử hàm isolate_roots. Đánh giá khả năng tìm các khoảng phân ly nghiệm.

    def test_f2_tim_du_ba_khoang(self): # Kiểm tra hàm isolate_roots có tìm được đủ ba khoảng chứa nghiệm của f2 hay không.
        intervals = isolate_roots(f2, -3, 3, 0.1) 
        # Gọi hàm isolate_roots để tìm khoảng phân ly nghiệm.
        # f2: Hàm số cần tìm nghiệm.
        # -3: Điểm bắt đầu quét.
        # 3: Điểm kết thúc quét.
        # 0.1: Bước quét giữa hai điểm liên tiếp.
        # Kết quả được lưu vào biến intervals.

        self.assertEqual(len(intervals), 3)
        # len(intervals) đếm số khoảng nghiệm tìm được.
        # assertEqual kiểm tra số khoảng tìm được có bằng 3 hay không.

        for root in ROOTS2: # Duyệt lần lượt ba nghiệm tham chiếu của hàm f2.
            found = any(
                a <= root <= b for a, b in intervals
            )
            # Kiểm tra có ít nhất một khoảng [a, b] chứa nghiệm root hay không.
            # any() trả về True nếu tồn tại ít nhất một điều kiện đúng.

            self.assertTrue(
                found,
                f"Không tìm được khoảng chứa nghiệm {root}"
            )
            # Kiểm tra biến found có giá trị True hay không.
            # Nếu False, kiểm thử thất bại và in thông báo nghiệm chưa tìm được.

    def test_nghiem_tai_diem_chia(self):
        # Kiểm tra trường hợp nghiệm nằm đúng tại một điểm chia của quá trình quét. Sử dụng hàm f(x) = x có nghiệm chính xác tại x = 0.

        self.assertEqual(
            isolate_roots(lambda x: x, -1, 1, 0.5),
            [(0.0, 0.0)]
        )
        # lambda x: x tạo một hàm số đơn giản f(x) = x.
        # Quét trên đoạn [-1, 1] với bước 0.5.
        # Kết quả mong đợi là khoảng suy biến (0.0, 0.0).
        # assertEqual kiểm tra kết quả thực tế có đúng kết quả mong đợi không.

    def test_nghiem_tai_bien_phai(self):
        # Kiểm tra trường hợp nghiệm nằm đúng tại biên phải của đoạn tìm kiếm.
        # Sử dụng hàm f(x) = x - 1 có nghiệm x = 1.

        self.assertEqual(
            isolate_roots(lambda x: x - 1, 0, 1, 0.25),
            [(1.0, 1.0)]
        )
        # Quét hàm số trên đoạn [0, 1] với bước 0.25.
        # Kết quả mong đợi là khoảng suy biến (1.0, 1.0).
        # Kiểm tra thuật toán có nhận diện được nghiệm tại biên phải hay không.

    def test_tham_so_khong_hop_le(self):
        # Kiểm tra khả năng xử lý các tham số đầu vào không hợp lệ.

        with self.assertRaises(ValueError):
            # Kiểm tra chương trình có phát sinh lỗi ValueError hay không.

            isolate_roots(f2, 3, -3, 0.1)
            # Trường hợp 1: a = 3 lớn hơn b = -3.
            # Khoảng tìm kiếm không hợp lệ.

        with self.assertRaises(ValueError):
            # Tiếp tục kiểm tra trường hợp tham số không hợp lệ.

            isolate_roots(f2, -3, 3, 0)
            # Trường hợp 2: step = 0.
            # Bước quét phải là số dương nên giá trị 0 không hợp lệ.


# PHẦN 5. KIỂM THỬ PHƯƠNG PHÁP CHIA ĐÔI

class TestBisection(unittest.TestCase):
    # Khai báo lớp kiểm thử phương pháp chia đôi.
    # Mục tiêu: Kiểm tra độ chính xác, số bước lặp và khả năng xử lý lỗi.

    def test_vi_du_giao_trinh(self):
        # Kiểm tra phương pháp chia đôi với ví dụ trong giáo trình.
        # Phương trình cần giải: x^2 - 2 = 0 trên đoạn [1, 2].

        x, k, err = bisection(
            lambda t: t*t - 2,
            1, 2,
            eps=0.01
        )
        # Gọi hàm bisection.
        # lambda t: t*t - 2 là hàm số cần tìm nghiệm.
        # 1, 2 là hai đầu mút của khoảng tìm kiếm.
        # eps=0.01 là sai số cho phép.
        # x: Nghiệm gần đúng.
        # k: Số bước lặp.
        # err: Sai số ước lượng được trả về.

        self.assertEqual(k, 6)
        # Kiểm tra thuật toán thực hiện đúng 6 bước lặp.

        self.assertAlmostEqual(x, 1.4140625, places=12)
        # Kiểm tra nghiệm gần đúng có bằng 1.4140625 hay không.
        # places=12 yêu cầu so sánh chính xác đến 12 chữ số thập phân.

        self.assertAlmostEqual(err, 0.0078125, places=12)
        # Kiểm tra sai số trả về có bằng 0.0078125 hay không.

    def test_f1_chia_doi(self):
        # Kiểm tra phương pháp chia đôi khi áp dụng cho hàm f1.

        x, k, err = bisection(
            f1, -1, 0,
            eps=1e-10,
            n_max=100
        )
        # Tìm nghiệm hàm f1 trên đoạn [-1, 0].
        # eps=1e-10: Sai số cho phép là 10 mũ -10.
        # n_max=100: Số bước lặp tối đa là 100.

        self.assertAlmostEqual(x, ROOT1, delta=1e-8)
        # So sánh nghiệm tìm được với nghiệm tham chiếu ROOT1.
        # delta=1e-8: Sai lệch tuyệt đối cho phép không vượt quá 10 mũ -8.

        self.assertGreaterEqual(k, 1)
        # Kiểm tra số bước lặp lớn hơn hoặc bằng 1.

        self.assertGreaterEqual(err, 0)
        # Kiểm tra sai số trả về không được âm.

    def test_dau_mut_la_nghiem(self):
        # Kiểm tra trường hợp nghiệm nằm ngay tại đầu mút trái của khoảng.
        # Phương trình x - 1 = 0 có nghiệm x = 1.

        self.assertEqual(
            bisection(lambda t: t - 1, 1, 2),
            (1.0, 0, 0.0)
        )
        # Vì đầu mút a = 1 đã là nghiệm nên không cần lặp.
        # Kết quả mong đợi gồm:
        # x = 1.0: Nghiệm chính xác.
        # k = 0: Không cần thực hiện bước lặp.
        # err = 0.0: Sai số bằng 0.

    def test_khong_phai_khoang_phan_ly(self):
        # Kiểm tra trường hợp khoảng tìm kiếm không thỏa mãn điều kiện đổi dấu.

        with self.assertRaises(ValueError):
            bisection(lambda t: t*t + 1, -1, 1)
            # Hàm f(x) = x^2 + 1 luôn dương trên đoạn [-1, 1].
            # Vì không có sự đổi dấu nên thuật toán phải báo lỗi ValueError.

    def test_het_n_max(self):
        # Kiểm tra trường hợp thuật toán đạt giới hạn số bước lặp tối đa
        # nhưng chưa đạt được sai số yêu cầu.

        with self.assertWarns(UserWarning):
            # Kiểm tra chương trình có phát sinh cảnh báo UserWarning hay không.

            _, k, _ = bisection(
                f1, -1, 0,
                eps=1e-12,
                n_max=5
            )
            # Yêu cầu sai số rất nhỏ là 1e-12.
            # Chỉ cho phép tối đa 5 bước lặp.
            # Dấu _ dùng để bỏ qua giá trị nghiệm và sai số không cần sử dụng.

        self.assertEqual(k, 5)
        # Kiểm tra số bước lặp thực tế không vượt quá giới hạn 5 bước.

# PHẦN 6. KIỂM THỬ PHƯƠNG PHÁP DÂY CUNG

class TestFalsePosition(unittest.TestCase):
    # Khai báo lớp kiểm thử phương pháp dây cung.
    # Mục tiêu: Đánh giá độ chính xác và khả năng hoạt động của thuật toán.

    def test_can_bac_hai_cua_2(self):
        # Kiểm tra phương pháp dây cung với phương trình x^2 - 2 = 0.

        x, k, err = false_position(
            lambda t: t*t - 2,
            1, 2,
            eps=1e-8,
            n_max=200
        )
        # Tìm nghiệm trên đoạn [1, 2].
        # eps=1e-8: Sai số cho phép là 10 mũ -8.
        # n_max=200: Giới hạn tối đa 200 bước lặp.

        self.assertAlmostEqual(
            x, math.sqrt(2), delta=1e-7
        )
        # So sánh nghiệm tìm được với giá trị căn bậc hai của 2.
        # math.sqrt(2) tính giá trị căn bậc hai của 2.
        # delta=1e-7 là sai lệch tối đa được chấp nhận.

        self.assertLess(k, 200)
        # Kiểm tra thuật toán dừng trước khi đạt giới hạn 200 bước lặp.

    def test_f1_day_cung(self):
        # Kiểm tra phương pháp dây cung khi giải hàm f1.

        x, k, err = false_position(
            f1, -1, 0,
            eps=1e-10,
            n_max=1000
        )
        # Tìm nghiệm hàm f1 trên đoạn [-1, 0].
        # eps=1e-10: Sai số cho phép.
        # n_max=1000: Số bước lặp tối đa.

        self.assertAlmostEqual(x, ROOT1, delta=1e-8)
        # Kiểm tra nghiệm tìm được gần với nghiệm tham chiếu ROOT1.

        self.assertGreaterEqual(k, 1)
        # Kiểm tra thuật toán có thực hiện ít nhất một bước lặp.

        self.assertGreaterEqual(err, 0)
        # Kiểm tra sai số trả về không âm.

    def test_dau_mut_la_nghiem(self):
        # Kiểm tra trường hợp nghiệm nằm tại đầu mút trái x = 1.

        self.assertEqual(
            false_position(lambda t: t - 1, 1, 2),
            (1.0, 0, 0.0)
        )
        # Vì x = 1 đã là nghiệm nên thuật toán không cần lặp.
        # Kết quả mong đợi: (nghiệm, số bước lặp, sai số) = (1, 0, 0).

    def test_tham_so_khong_hop_le(self):
        # Kiểm tra khả năng phát hiện các tham số đầu vào không hợp lệ.

        with self.assertRaises(ValueError):
            false_position(f1, 0, -1)
            # Trường hợp 1: Đầu mút trái lớn hơn đầu mút phải.

        with self.assertRaises(ValueError):
            false_position(f1, -1, 0, n_max=0)
            # Trường hợp 2: Số bước lặp tối đa bằng 0.
            # n_max phải là số nguyên dương.

        with self.assertRaises(ValueError):
            false_position(
                lambda t: t*t + 1, -1, 1
            )
            # Trường hợp 3: Khoảng không có sự đổi dấu.
            # Hàm x^2 + 1 luôn dương nên không thỏa mãn điều kiện phân ly nghiệm.


# PHẦN 7. KIỂM THỬ SO SÁNH HAI PHƯƠNG PHÁP

class TestSoSanhHaiPhuongPhap(unittest.TestCase):
    # Khai báo lớp kiểm thử chéo giữa phương pháp chia đôi và dây cung.
    # Mục tiêu: So sánh kết quả nghiệm và số bước lặp của hai phương pháp.

    def test_f2_chia_doi_va_day_cung(self):
        # Kiểm tra cả hai phương pháp trên ba khoảng phân ly nghiệm của f2.

        for (a, b), root in zip(BRACKETS2, ROOTS2):
            # zip() ghép từng khoảng trong BRACKETS2
            # với nghiệm tham chiếu tương ứng trong ROOTS2.
            # a, b là hai đầu mút của khoảng.
            # root là nghiệm tham chiếu.

            with self.subTest(khoang=(a, b)):
                # Tạo một trường hợp kiểm thử riêng cho từng khoảng.
                # Giúp xác định khoảng xảy ra lỗi nếu có.

                self.assertLess(f2(a) * f2(b), 0)
                # Kiểm tra điều kiện đổi dấu của khoảng đang xét.
                # Nếu tích nhỏ hơn 0 thì khoảng thỏa mãn điều kiện phân ly nghiệm.

                xb, _, _ = bisection(
                    f2, a, b,
                    eps=1e-10,
                    n_max=100
                )
                # Sử dụng phương pháp chia đôi để tìm nghiệm.
                # xb lưu nghiệm gần đúng.
                # Hai dấu _ bỏ qua số bước lặp và sai số vì chưa cần sử dụng.

                xf, _, _ = false_position(
                    f2, a, b,
                    eps=1e-10,
                    n_max=1000
                )
                # Sử dụng phương pháp dây cung để tìm nghiệm.
                # xf lưu nghiệm gần đúng.
                # Sai số cho phép là 1e-10.
                # Số bước lặp tối đa là 1000.

                self.assertAlmostEqual(
                    xb, root, delta=2e-5
                )
                # So sánh nghiệm của phương pháp chia đôi với nghiệm tham chiếu.
                # Sai lệch cho phép là 2e-5.

                self.assertAlmostEqual(
                    xf, root, delta=2e-5
                )
                # So sánh nghiệm của phương pháp dây cung với nghiệm tham chiếu.
                # Sai lệch cho phép là 2e-5.

    def test_bang_so_sanh_so_buoc(self):
        # Thực nghiệm so sánh số bước lặp của hai phương pháp.
        # Áp dụng trên hàm f1 và ba khoảng của hàm f2.

        print("\n\nSO SANH CHIA DOI VA DAY CUNG")
        # In tiêu đề bảng kết quả so sánh ra màn hình.

        print(
            f"{'Ham':<5} {'Khoang':<18} "
            f"{'eps':<12} {'k chia doi':<12} "
            f"{'k day cung':<12}"
        )
        # In tên các cột của bảng.
        # <5, <18, <12 quy định độ rộng và căn lề trái cho từng cột.
        # Các cột gồm: Tên hàm, khoảng nghiệm, sai số,
        # số bước lặp chia đôi và số bước lặp dây cung.

        cases = [
            ("f1", f1, -1, 0),
            ("f2a", f2, -2.1, -2.0),
            ("f2b", f2, 0.1, 0.2),
            ("f2c", f2, 1.9, 2.0)
        ]
        # Tạo danh sách các trường hợp thực nghiệm.
        # Mỗi phần tử gồm tên hàm, hàm số, đầu mút trái và đầu mút phải.
        # f1 được kiểm tra trên [-1, 0].
        # f2 được kiểm tra trên ba khoảng phân ly nghiệm.

        for name, f, a, b in cases:
            # Duyệt qua từng trường hợp trong danh sách cases.
            # name: Tên trường hợp kiểm thử.
            # f: Hàm số cần tính.
            # a: Đầu mút trái.
            # b: Đầu mút phải.

            for eps in EPS_LIST:
                # Duyệt lần lượt năm mức sai số trong EPS_LIST.
                # Mỗi khoảng sẽ được thử với năm mức sai số khác nhau.

                with self.subTest(
                    ham=name,
                    eps=eps,
                    khoang=(a, b)
                ):
                    # Tách từng trường hợp kiểm thử.
                    # Nếu xảy ra lỗi, có thể xác định hàm, sai số và khoảng gây lỗi.

                    with warnings.catch_warnings():
                        # Tạo một vùng kiểm soát cảnh báo trong quá trình thực nghiệm.

                        warnings.simplefilter("ignore")
                        # Tạm thời ẩn các cảnh báo phát sinh.
                        # Việc này giúp bảng kết quả dễ quan sát hơn.
                        # Không làm thay đổi thuật toán tìm nghiệm.

                        _, kb, _ = bisection(
                            f, a, b,
                            eps=eps,
                            n_max=1000
                        )
                        # Chạy phương pháp chia đôi.
                        # kb lưu số bước lặp của phương pháp chia đôi.
                        # Dấu _ bỏ qua nghiệm và sai số trả về.

                        _, kf, _ = false_position(
                            f, a, b,
                            eps=eps,
                            n_max=10000
                        )
                        # Chạy phương pháp dây cung.
                        # kf lưu số bước lặp của phương pháp dây cung.
                        # n_max=10000 là số bước lặp tối đa cho phép.

                    print(
                        f"{name:<5} [{a:>5.1f},{b:>5.1f}] "
                        f"{eps:<12.0e} "
                        f"{kb:<12} {kf:<12}"
                    )
                    # In kết quả thực nghiệm ra màn hình.
                    # name: Tên hàm.
                    # a, b: Khoảng tìm nghiệm.
                    # eps: Mức sai số.
                    # kb: Số bước lặp của phương pháp chia đôi.
                    # kf: Số bước lặp của phương pháp dây cung.


# PHẦN 8. CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    # Kiểm tra file có đang được chạy trực tiếp hay không.
    # Nếu chạy trực tiếp file này thì thực hiện các bài kiểm thử bên dưới.

    unittest.main(
        # Gọi bộ kiểm thử unittest để tự động tìm và chạy các test case.

        argv=['first-arg-is-ignored'],
        # Thiết lập danh sách tham số dòng lệnh.
        # Tránh để unittest hiểu các tham số bên ngoài là tên test cần chạy.

        exit=False,
        # Không đóng phiên Python sau khi hoàn thành kiểm thử.

        verbosity=2
        # Mức hiển thị kết quả kiểm thử.
        # verbosity=2: Hiển thị tên từng test và trạng thái OK hoặc FAIL.
    )

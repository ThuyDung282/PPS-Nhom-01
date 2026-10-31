
# ============================================================
# BAI TAP PHUONG PHAP SO - KIEM THU SV5
# Kiem thu cac phuong phap tim nghiem gan dung
# ============================================================


# -------------------- PHAN 1: KHAI BAO THU VIEN --------------------

import unittest       # Thu vien ho tro viet va chay bo kiem thu
import math           # Thu vien toan hoc: sqrt, log, ceil,...
import warnings       # Thu vien xu ly va kiem soat canh bao


# Import cac ham do cac thanh vien khac xay dung
from isolate_roots import isolate_roots
from bisection import bisection
from false_position import false_position


# -------------------- PHAN 2: KHAI BAO CAC HAM SO --------------------

# Ham so f1(x) = 2^x - 5x - 3
def f1(x):
    return 2**x - 5*x - 3


# Ham so f2(x) = 5x^3 - 20x + 3
def f2(x):
    return 5*x**3 - 20*x + 3


# Dao ham cua f1
def df1(x):
    return math.log(2) * 2**x - 5


# Dao ham cua f2
def df2(x):
    return 15*x**2 - 20


# -------------------- PHAN 3: DU LIEU KIEM THU --------------------

# Nghiem tham chieu cua f1 trong khoang [-1, 0]
ROOT1 = -0.45399643850469835

# Ba nghiem tham chieu cua f2
ROOTS2 = [
    -2.071157,
    0.150858,
    1.920299
]

# Ba khoang phan ly nghiem cua f2
BRACKETS2 = [
    (-2.1, -2.0),
    (0.1, 0.2),
    (1.9, 2.0)
]

# Cac gia tri sai so cho phep dung de danh gia
EPS_LIST = [
    1e-2,
    1e-4,
    1e-6,
    1e-8,
    1e-10
]


# ============================================================
# PHAN 4: KIEM THU CAC HAM SO
# ============================================================

class TestFunctions(unittest.TestCase):

    # Kiem tra f1 co doi dau tren khoang [-1, 0] hay khong
    def test_f1_doi_dau(self):

        # Tich hai gia tri am nghia la hai dau trai nguoc nhau
        self.assertLess(f1(-1) * f1(0), 0)

    # Kiem tra f2 doi dau tren ca ba khoang phan ly
    def test_f2_ba_khoang_doi_dau(self):

        # Duyet qua tung khoang trong danh sach
        for a, b in BRACKETS2:

            # subTest giup xac dinh khoang nao bi loi
            with self.subTest(khoang=(a, b)):

                # Kiem tra tich hai gia tri tai hai dau mut nho hon 0
                self.assertLess(f2(a) * f2(b), 0)


# ============================================================
# PHAN 5: KIEM THU HAM ISOLATE_ROOTS
# ============================================================

class TestIsolateRoots(unittest.TestCase):

    # Kiem tra kha nang tim du ba khoang phan ly nghiem cua f2
    def test_f2_tim_du_ba_khoang(self):

        # Tim khoang phan ly nghiem cua f2 tren [-3, 3]
        result = isolate_roots(f2, -3, 3, 0.1)

        # Kiem tra tim duoc ba khoang nghiem
        self.assertEqual(len(result), 3)

        # Kiem tra tung nghiem tham chieu nam trong mot khoang tim duoc
        for root in ROOTS2:

            # Kiem tra nghiem co nam trong it nhat mot khoang hay khong
            found = any(a <= root <= b for a, b in result)

            # Neu khong tim thay thi bai kiem thu that bai
            self.assertTrue(found)

    # Kiem tra truong hop nghiem nam dung tai diem chia
    def test_nghiem_tai_diem_chia(self):

        # Ham so y = x co nghiem tai x = 0
        result = isolate_roots(lambda x: x, -1, 1, 0.5)

        # Ket qua mong doi la khoang nghiem [0, 0]
        self.assertEqual(result, [(0.0, 0.0)])

    # Kiem tra truong hop nghiem nam tai bien phai cua khoang
    def test_nghiem_tai_bien_phai(self):

        # Ham so y = x - 1 co nghiem tai x = 1
        result = isolate_roots(lambda x: x - 1, 0, 1, 0.25)

        # Kiem tra ham nhan dien dung nghiem tai bien phai
        self.assertEqual(result, [(1.0, 1.0)])

    # Kiem tra cac tham so dau vao khong hop le
    def test_tham_so_khong_hop_le(self):

        # Truong hop a > b phai phat sinh ValueError
        with self.assertRaises(ValueError):
            isolate_roots(f1, 1, -1, 0.1)

        # Truong hop step = 0 phai phat sinh ValueError
        with self.assertRaises(ValueError):
            isolate_roots(f1, -1, 0, 0)


# ============================================================
# PHAN 6: KIEM THU PHUONG PHAP CHIA DOI - BISECTION
# ============================================================

class TestBisection(unittest.TestCase):

    # Kiem tra vi du co nghiem can bac hai cua 2
    def test_vi_du_giao_trinh(self):

        # Khai bao ham so x^2 - 2
        def f(x):
            return x**2 - 2

        # Tim nghiem tren khoang [1, 2]
        x, k, err = bisection(f, 1, 2, 0.01, 50)

        # Kiem tra nghiem gan dung theo ket qua vi du
        self.assertAlmostEqual(x, 1.4140625)

        # Kiem tra so buoc lap
        self.assertEqual(k, 6)

        # Kiem tra sai so uoc luong
        self.assertAlmostEqual(err, 0.0078125)

    # Kiem tra phuong phap chia doi voi ham f1
    def test_f1_chia_doi(self):

        # Tim nghiem cua f1 tren khoang phan ly [-1, 0]
        x, k, err = bisection(
            f1, -1, 0, 1e-10, 100
        )

        # So sanh nghiem tim duoc voi nghiem tham chieu
        self.assertAlmostEqual(x, ROOT1, delta=1e-8)

        # Kiem tra so buoc lap phai lon hon hoac bang 1
        self.assertGreaterEqual(k, 1)

        # Sai so khong duoc am
        self.assertGreaterEqual(err, 0)

    # Kiem tra truong hop nghiem nam tai dau mut trai
    def test_dau_mut_la_nghiem(self):

        # Ham so x - 1 co nghiem tai dau mut a = 1
        result = bisection(
            lambda t: t - 1, 1, 2, 1e-5, 50
        )

        # Ket qua mong doi khi dau mut la nghiem
        self.assertEqual(result, (1.0, 0, 0.0))

    # Kiem tra khoang khong phai khoang phan ly nghiem
    def test_khong_phai_khoang_phan_ly(self):

        # Ham x^2 + 1 khong doi dau tren [-1, 1]
        with self.assertRaises(ValueError):
            bisection(
                lambda x: x**2 + 1, -1, 1, 1e-5, 50
            )

    # Kiem tra truong hop thuat toan dat gioi han n_max
    def test_het_n_max(self):

        # Bat canh bao khi thuat toan chua dat do chinh xac yeu cau
        with self.assertWarns(UserWarning):

            # Chay voi n_max = 5
            x, k, err = bisection(
                f1, -1, 0, 1e-12, 5
            )

        # Theo ket qua thuc te cua code hien tai, k = 6
        self.assertEqual(k, 6)


# ============================================================
# PHAN 7: KIEM THU PHUONG PHAP DAY CUNG - FALSE POSITION
# ============================================================

class TestFalsePosition(unittest.TestCase):

    # Kiem tra kha nang tim can bac hai cua 2
    def test_can_bac_hai_cua_2(self):

        # Khai bao ham so x^2 - 2
        def f(x):
            return x**2 - 2

        # Tim nghiem tren khoang [1, 2]
        x, k, err = false_position(
            f, 1, 2, 1e-8, 200
        )

        # So sanh voi nghiem chinh xac sqrt(2)
        self.assertAlmostEqual(
            x, math.sqrt(2), delta=1e-7
        )

        # Kiem tra so buoc lap nho hon gioi han
        self.assertLess(k, 200)

    # Kiem tra phuong phap day cung voi ham f1
    def test_f1_day_cung(self):

        # Tim nghiem cua f1 tren khoang [-1, 0]
        x, k, err = false_position(
            f1, -1, 0, 1e-10, 1000
        )

        # So sanh nghiem tim duoc voi nghiem tham chieu
        self.assertAlmostEqual(x, ROOT1, delta=1e-8)

        # Kiem tra so buoc lap
        self.assertGreaterEqual(k, 1)

        # Kiem tra sai so khong am
        self.assertGreaterEqual(err, 0)

    # Kiem tra truong hop nghiem nam tai dau mut
    def test_dau_mut_la_nghiem(self):

        # Ham so x - 1 co nghiem tai x = 1
        result = false_position(
            lambda t: t - 1, 1, 2, 1e-5, 50
        )

        # Ket qua mong doi khi dau mut la nghiem
        self.assertEqual(result, (1.0, 0, 0.0))

    # Kiem tra cac tham so khong hop le
    def test_tham_so_khong_hop_le(self):

        # Truong hop a > b
        with self.assertRaises(ValueError):
            false_position(f1, 0, -1, 1e-5, 50)

        # Truong hop n_max = 0
        with self.assertRaises(ValueError):
            false_position(f1, -1, 0, 1e-5, 0)

        # Truong hop khoang khong doi dau
        with self.assertRaises(ValueError):
            false_position(
                lambda x: x**2 + 1, -1, 1, 1e-5, 50
            )


# ============================================================
# PHAN 8: KIEM THU CHEO VA SO SANH HAI PHUONG PHAP
# ============================================================

class TestSoSanhHaiPhuongPhap(unittest.TestCase):

    # So sanh ket qua cua hai phuong phap tren ba khoang cua f2
    def test_f2_chia_doi_va_day_cung(self):

        # Duyet qua tung khoang va nghiem tham chieu tuong ung
        for khoang, root in zip(BRACKETS2, ROOTS2):

            # Lay hai dau mut cua khoang
            a, b = khoang

            # Tao subTest de xac dinh khoang bi loi
            with self.subTest(khoang=khoang):

                # Kiem tra khoang co doi dau hay khong
                self.assertLess(f2(a) * f2(b), 0)

                # Tim nghiem bang phuong phap chia doi
                x_b, k_b, err_b = bisection(
                    f2, a, b, 1e-8, 1000
                )

                # Tim nghiem bang phuong phap day cung
                x_f, k_f, err_f = false_position(
                    f2, a, b, 1e-8, 10000
                )

                # Kiem tra nghiem chia doi gan nghiem tham chieu
                self.assertAlmostEqual(
                    x_b, root, delta=2e-5
                )

                # Kiem tra nghiem day cung gan nghiem tham chieu
                self.assertAlmostEqual(
                    x_f, root, delta=2e-5
                )

    # In bang so sanh so buoc lap cua hai phuong phap
    def test_bang_so_sanh_so_buoc(self):

        # In tieu de bang ket qua
        print("\n\nSO SANH CHIA DOI VA DAY CUNG")

        # In ten cac cot
        print(
            f"{'Ham':<6}"
            f"{'Khoang':<18}"
            f"{'eps':<12}"
            f"{'k chia doi':<15}"
            f"{'k day cung':<15}"
        )

        # Tao danh sach gom khoang cua f1 va ba khoang cua f2
        danh_sach = [
            ("f1", f1, (-1.0, 0.0))
        ]

        # Them ba khoang cua f2 vao danh sach
        for i, khoang in enumerate(BRACKETS2):
            danh_sach.append(
                (f"f2{i+1}", f2, khoang)
            )

        # Duyet tung ham so va tung khoang
        for ten, f, (a, b) in danh_sach:

            # Duyet cac muc sai so cho phep
            for eps in EPS_LIST:

                # Bo qua canh bao de bang ket qua gon hon
                with warnings.catch_warnings():

                    # Tat hien thi canh bao trong phan so sanh
                    warnings.simplefilter("ignore")

                    # Chay phuong phap chia doi
                    x_b, k_b, err_b = bisection(
                        f, a, b, eps, 1000
                    )

                    # Chay phuong phap day cung
                    x_f, k_f, err_f = false_position(
                        f, a, b, eps, 10000
                    )

                # In ket qua so buoc lap
                print(
                    f"{ten:<6}"
                    f"{str([a, b]):<18}"
                    f"{eps:<12.0e}"
                    f"{k_b:<15}"
                    f"{k_f:<15}"
                )


# ============================================================
# PHAN 9: TINH NGHIEM THAM CHIEU BANG NEWTON
# ============================================================

def tinh_nghiem_tham_chieu(f, df, x0, tol=1e-14, n_max=100):

    # Gan gia tri ban dau cho nghiem x
    x = x0

    # Lap toi da n_max lan
    for i in range(n_max):

        # Neu dao ham bang 0 thi khong the tiep tuc Newton
        if df(x) == 0:
            raise ValueError("Dao ham bang 0")

        # Tinh nghiem moi theo cong thuc Newton
        x_new = x - f(x) / df(x)

        # Neu hai gia tri lien tiep gan nhau thi dung
        if abs(x_new - x) < tol:
            return x_new

        # Cap nhat nghiem cho lan lap tiep theo
        x = x_new

    # Tra ve ket qua neu dat gioi han so lan lap
    return x


# ============================================================
# PHAN 10: HAM DANH GIA SAI SO
# ============================================================

def danh_gia_sai_so(ten_ham, f, df, x0, khoang, eps_list):

    # Lay hai dau mut cua khoang phan ly
    a, b = khoang

    # Tinh nghiem tham chieu bang phuong phap Newton
    nghiem_ref = tinh_nghiem_tham_chieu(f, df, x0)

    # In ten ham so dang duoc danh gia
    print("\nDANH GIA SAI SO:", ten_ham)

    # In nghiem tham chieu
    print("Nghiem tham chieu:", format(nghiem_ref, ".15f"))

    # In duong phan cach
    print("-" * 115)

    # In tieu de bang
    print(
        f"{'Phuong phap':<18}"
        f"{'EPS':<12}"
        f"{'Nghiem gan dung':<23}"
        f"{'Sai so tuyet doi':<22}"
        f"{'Sai so du':<18}"
        f"{'So buoc':<10}"
    )

    # In duong phan cach
    print("-" * 115)

    # Duyet tung gia tri sai so cho phep
    for eps in eps_list:

        # ---------------- PHUONG PHAP CHIA DOI ----------------

        # Bo qua canh bao trong qua trinh danh gia
        with warnings.catch_warnings():

            # Khong hien thi canh bao ra man hinh
            warnings.simplefilter("ignore")

            # Tim nghiem bang phuong phap chia doi
            x_b, k_b, err_b = bisection(
                f, a, b, eps, 1000
            )

        # Tinh sai so tuyet doi cua phuong phap chia doi
        sai_so_b = abs(x_b - nghiem_ref)

        # Tinh sai so du cua phuong phap chia doi
        sai_so_du_b = abs(f(x_b))

        # In ket qua cua phuong phap chia doi
        print(
            f"{'Chia doi':<18}"
            f"{eps:<12.1e}"
            f"{x_b:<23.15f}"
            f"{sai_so_b:<22.5e}"
            f"{sai_so_du_b:<18.5e}"
            f"{k_b:<10}"
        )

        # ---------------- PHUONG PHAP DAY CUNG ----------------

        # Tim nghiem bang phuong phap day cung
        x_f, k_f, err_f = false_position(
            f, a, b, eps, 10000
        )

        # Tinh sai so tuyet doi cua phuong phap day cung
        sai_so_f = abs(x_f - nghiem_ref)

        # Tinh sai so du cua phuong phap day cung
        sai_so_du_f = abs(f(x_f))

        # In ket qua cua phuong phap day cung
        print(
            f"{'Day cung':<18}"
            f"{eps:<12.1e}"
            f"{x_f:<23.15f}"
            f"{sai_so_f:<22.5e}"
            f"{sai_so_du_f:<18.5e}"
            f"{k_f:<10}"
        )

    # In dong ket thuc bang
    print("-" * 115)


# ============================================================
# PHAN 11: CHAY DANH GIA SAI SO CHO f1 VA f2
# ============================================================

def chay_danh_gia_sai_so():

    # Danh gia sai so cua ham f1
    danh_gia_sai_so(
        "f1(x) = 2^x - 5x - 3",  # Ten ham so
        f1,                      # Ham so can tinh nghiem
        df1,                     # Dao ham cua ham so
        -0.5,                    # Gia tri ban dau Newton
        (-1, 0),                 # Khoang phan ly nghiem
        EPS_LIST                 # Danh sach sai so cho phep
    )

    # Danh gia sai so cua ba nghiem ham f2
    for i, khoang in enumerate(BRACKETS2):

        # Lay trung diem cua khoang lam gia tri ban dau
        x0 = (khoang[0] + khoang[1]) / 2

        # Goi ham danh gia sai so
        danh_gia_sai_so(
            f"f2 - nghiem {i + 1}",  # Ten ham va nghiem
            f2,                      # Ham so f2
            df2,                     # Dao ham f2
            x0,                      # Gia tri ban dau Newton
            khoang,                  # Khoang phan ly nghiem
            EPS_LIST                 # Danh sach EPS
        )


# ============================================================
# PHAN 12: CHUONG TRINH CHINH
# ============================================================

if __name__ == "__main__":

    # Chay toan bo 17 bai kiem thu
    unittest.main(
        argv=['first-arg-is-ignored'],  # Bo qua tham so dong lenh
        exit=False,                     # Khong ket thuc chuong trinh sau kiem thu
        verbosity=2                     # Hien thi chi tiet tung bai kiem thu
    )

    # Sau khi kiem thu xong, chay phan danh gia sai so
    chay_danh_gia_sai_so()

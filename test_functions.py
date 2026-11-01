
# ============================================================
# PHAN 1: KHAI BAO THU VIEN
# ============================================================

import math
import warnings


# ============================================================
# PHAN 2: CAC HAM SO CAN GIAI
# ============================================================

# Ham so f1(x) = 2^x - 5x - 3
def f1(x):
    return 2**x - 5*x - 3


# Dao ham cua f1
def df1(x):
    return math.log(2) * 2**x - 5


# Ham so f2(x) = 5x^3 - 20x + 3
def f2(x):
    return 5*x**3 - 20*x + 3


# Dao ham cua f2
def df2(x):
    return 15*x**2 - 20


# ============================================================
# PHAN 3: TIM KHOANG PHAN LY NGHIEM
# ============================================================

def isolate_roots(f, a, b, step):
    """
    Tim khoang phan ly nghiem bang phuong phap quet dau.

    Dau vao:
        f: Ham so can tim nghiem.
        a: Bien trai.
        b: Bien phai.
        step: Buoc quet.

    Dau ra:
        Danh sach cac khoang chua nghiem.
    """

    # Kiem tra dieu kien bien trai va bien phai
    if a >= b:
        raise ValueError("Can a < b.")

    # Kiem tra buoc quet phai duong
    if step <= 0:
        raise ValueError("step phai lon hon 0.")

    # Khoi tao danh sach chua cac khoang phan ly
    intervals = []

    # Gan diem quet dau tien bang a
    x_curr = float(a)

    # Quet lan luot tu a den b
    while x_curr < b:

        # Tinh diem tiep theo, khong vuot qua b
        x_next = min(x_curr + step, float(b))

        # Tinh gia tri ham tai hai diem
        y_curr = f(x_curr)
        y_next = f(x_next)

        # Neu hai gia tri trai dau thi co khoang phan ly
        if y_curr * y_next < 0:
            intervals.append((x_curr, x_next))

        # Neu diem hien tai la nghiem chinh xac
        elif y_curr == 0:
            intervals.append((x_curr, x_curr))

        # Chuyen sang diem tiep theo
        x_curr = x_next

    # Kiem tra rieng tai bien phai
    if f(b) == 0:

        # Tranh them trung nghiem tai bien phai
        if not intervals or intervals[-1] != (float(b), float(b)):
            intervals.append((float(b), float(b)))

    # Tra ve danh sach khoang phan ly
    return intervals


# ============================================================
# PHAN 4: DANH GIA SO BUOC LAP LY THUYET CUA CHIA DOI
# ============================================================

def bisection_steps(a, b, eps):
    """
    Tinh so buoc lap ly thuyet cua phuong phap chia doi.

    Cong thuc:
        n = max(0, ceil(log2((b-a)/eps)) - 1)
    """

    # Kiem tra khoang phan ly
    if a >= b:
        raise ValueError("Can b > a.")

    # Kiem tra sai so cho phep
    if eps <= 0:
        raise ValueError("eps phai lon hon 0.")

    # Tinh so buoc lap theo cong thuc
    n = math.ceil(math.log2((b - a) / eps)) - 1

    # Dam bao so buoc lap khong am
    return max(0, n)


# ============================================================
# PHAN 5: PHUONG PHAP CHIA DOI
# ============================================================

def bisection(f, a, b, eps=1e-5, n_max=50):
    """
    Giai phuong trinh f(x) = 0 bang phuong phap chia doi.

    Tra ve:
        x: Nghiem gan dung.
        k: So buoc lap.
        err: Sai so uoc luong.
    """

    # Kiem tra khoang
    if not a < b:
        raise ValueError("Can a < b.")

    # Kiem tra sai so
    if eps <= 0:
        raise ValueError("eps phai lon hon 0.")

    # Kiem tra so buoc lap toi da
    if n_max < 0:
        raise ValueError("n_max phai khong am.")

    # Tinh gia tri ham tai hai dau mut
    fa = f(a)
    fb = f(b)

    # Neu dau mut trai la nghiem
    if fa == 0:
        return float(a), 0, 0.0

    # Neu dau mut phai la nghiem
    if fb == 0:
        return float(b), 0, 0.0

    # Kiem tra dieu kien doi dau
    if fa * fb > 0:
        raise ValueError("Khoang khong phai khoang phan ly nghiem.")

    # Khoi tao bien dem so buoc lap
    i = 0

    # Tinh sai so ban dau
    err = (b - a) / 2

    # Lap khi sai so lon hon eps va chua vuot gioi han
    while err > eps and i < n_max + 1:

        # Tinh trung diem cua khoang
        x = (a + b) / 2

        # Tinh gia tri ham tai trung diem
        fx = f(x)

        # Neu trung diem la nghiem chinh xac
        if fa * fx == 0:
            return float(x), i + 1, 0.0

        # Neu nghiem nam trong nua ben phai
        elif fa * fx > 0:

            # Thay doi bien trai
            a = x

            # Cap nhat gia tri ham tai a
            fa = fx

        # Neu nghiem nam trong nua ben trai
        else:

            # Thay doi bien phai
            b = x

            # Cap nhat gia tri ham tai b
            fb = fx

        # Tinh lai sai so cua khoang moi
        err = (b - a) / 2

        # Tang bien dem
        i = i + 1

    # Neu chua dat sai so yeu cau thi phat canh bao
    if err > eps:
        warnings.warn(
            f"Het n_max={n_max} buoc nhung chua dat eps={eps}.",
            UserWarning
        )

    # Lay trung diem cua khoang cuoi cung
    x = (a + b) / 2

    # Tinh sai so cuoi cung
    err = (b - a) / 2

    # Tra ve nghiem, so buoc va sai so
    return float(x), i, float(err)


# ============================================================
# PHAN 6: DANH GIA SAI SO PHUONG PHAP DAY CUNG
# ============================================================

def false_position_error(x_k, x_prev, m1, M1):
    """
    Tinh can tren sai so cua phuong phap day cung.

    Cong thuc:
        err <= ((M1-m1)/m1) * abs(x_k-x_prev)
    """

    # Kiem tra can duoi
    if m1 <= 0:
        raise ValueError("m1 phai lon hon 0.")

    # Kiem tra can tren
    if M1 < m1:
        raise ValueError("M1 phai lon hon hoac bang m1.")

    # Tinh can tren sai so
    err = ((M1 - m1) / m1) * abs(x_k - x_prev)

    # Tra ve sai so
    return float(err)


# ============================================================
# PHAN 7: PHUONG PHAP DAY CUNG
# ============================================================

def false_position(f, a, b, eps=1e-5, n_max=50):
    """
    Giai phuong trinh f(x) = 0 bang phuong phap day cung.

    Tra ve:
        x: Nghiem gan dung.
        k: So buoc lap.
        err: Do chenh lech hai nghiem lien tiep.
    """

    # Kiem tra f co phai ham so hay khong
    if not callable(f):
        raise TypeError("f phai la mot ham so.")

    # Kiem tra khoang
    if a >= b:
        raise ValueError("Can a < b.")

    # Kiem tra sai so
    if eps <= 0:
        raise ValueError("eps phai lon hon 0.")

    # Kiem tra so buoc lap
    if not isinstance(n_max, int) or n_max < 1:
        raise ValueError("n_max phai la so nguyen duong.")

    # Tinh gia tri ham tai hai dau mut
    fa = f(a)
    fb = f(b)

    # Neu dau mut trai la nghiem
    if fa == 0:
        return float(a), 0, 0.0

    # Neu dau mut phai la nghiem
    if fb == 0:
        return float(b), 0, 0.0

    # Kiem tra dieu kien doi dau
    if fa * fb > 0:
        raise ValueError("Khoang khong thoa man dieu kien doi dau.")

    # Chon diem bat dau cho day cung
    if fa < 0:
        x_prev = float(a)
    else:
        x_prev = float(b)

    # Khoi tao nghiem va sai so
    xn = 0.0
    err = 0.0

    # Bat dau vong lap
    for i in range(1, n_max + 1):

        # Tinh gia tri ham tai nghiem truoc
        fx_prev = f(x_prev)

        # Truong hop fa am
        if fa < 0:

            # Tinh mau so
            mau_so = fx_prev - fb

            # Kiem tra mau so
            if mau_so == 0:
                raise ZeroDivisionError("Mau so bang 0.")

            # Tinh nghiem moi
            xn = x_prev - fx_prev * (x_prev - b) / mau_so

        # Truong hop fa duong
        else:

            # Tinh mau so
            mau_so = fx_prev - fa

            # Kiem tra mau so
            if mau_so == 0:
                raise ZeroDivisionError("Mau so bang 0.")

            # Tinh nghiem moi
            xn = x_prev - fx_prev * (x_prev - a) / mau_so

        # Tinh sai so giua hai lan lap
        err = abs(xn - x_prev)

        # Neu dat sai so yeu cau thi dung
        if err < eps:
            return float(xn), i, float(err)

        # Tinh gia tri ham tai nghiem moi
        fn = f(xn)

        # Cap nhat khoang phan ly
        if fa * fn < 0:

            # Thay doi bien phai
            b = xn

            # Cap nhat gia tri ham tai b
            fb = fn

        else:

            # Thay doi bien trai
            a = xn

            # Cap nhat gia tri ham tai a
            fa = fn

        # Cap nhat nghiem truoc
        x_prev = xn

    # Canh bao khi het so buoc lap
    warnings.warn(
        "Chua dat sai so yeu cau sau so buoc lap toi da.",
        RuntimeWarning
    )

    # Tra ve ket qua
    return float(xn), n_max, float(err)


# ============================================================
# PHAN 8: TINH NGHIEM THAM CHIEU BANG PHUONG PHAP NEWTON
# ============================================================

def nghiem_tham_chieu(f, df, x0):
    """
    Tinh nghiem tham chieu bang phuong phap Newton.

    Nghiem nay dung de doi chieu sai so.
    """

    # Gan gia tri ban dau
    x = x0

    # Lap toi da 100 lan
    for i in range(100):

        # Kiem tra dao ham
        if abs(df(x)) < 1e-15:
            raise ValueError("Dao ham gan bang 0.")

        # Tinh nghiem moi theo cong thuc Newton
        x_moi = x - f(x) / df(x)

        # Kiem tra dieu kien dung
        if abs(x_moi - x) < 1e-14:
            return x_moi

        # Cap nhat gia tri x
        x = x_moi

    # Tra ve ket qua sau 100 lan lap
    return x


# ============================================================
# PHAN 9: DANH GIA SAI SO CHO MOT KHOANG NGHIEM
# ============================================================

def danh_gia_khoang(ten_ham, f, df, a, b, x0, m1, M1):

    # Tinh nghiem tham chieu
    x_ref = nghiem_tham_chieu(f, df, x0)

    # In thong tin bai toan
    print("\n" + "=" * 120)
    print("HAM SO:", ten_ham)
    print("KHOANG PHAN LY:", (a, b))
    print("NGHIEM THAM CHIEU:", format(x_ref, ".14f"))
    print("=" * 120)

    # In tieu de bang
    print(
        f"{'EPS':<12}"
        f"{'Phuong phap':<15}"
        f"{'Nghiem gan dung':<22}"
        f"{'Sai so tuyet doi':<20}"
        f"{'Sai so du':<18}"
        f"{'Sai so uoc luong':<20}"
        f"{'So buoc':<10}"
    )

    print("-" * 120)

    # Duyet tung gia tri eps
    for eps in EPS_LIST:

        # ----------------------------------------------------
        # PHUONG PHAP CHIA DOI
        # ----------------------------------------------------

        # Chay phuong phap chia doi va bo qua canh bao
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            x_b, k_b, err_b = bisection(
                f, a, b, eps, 1000
            )

        # Tinh sai so tuyet doi
        sai_so_tuyet_doi_b = abs(x_b - x_ref)

        # Tinh sai so du
        sai_so_du_b = abs(f(x_b))

        # Tinh sai so uoc luong theo khoang ban dau
        sai_so_uoc_luong_b = (b - a) / (2 ** (k_b + 1))

        # In ket qua chia doi
        print(
            f"{eps:<12.1e}"
            f"{'Chia doi':<15}"
            f"{x_b:<22.12f}"
            f"{sai_so_tuyet_doi_b:<20.5e}"
            f"{sai_so_du_b:<18.5e}"
            f"{sai_so_uoc_luong_b:<20.5e}"
            f"{k_b:<10}"
        )

        # ----------------------------------------------------
        # PHUONG PHAP DAY CUNG
        # ----------------------------------------------------

        # Chay phuong phap day cung
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            x_f, k_f, err_f = false_position(
                f, a, b, eps, 10000
            )

        # Tinh sai so tuyet doi
        sai_so_tuyet_doi_f = abs(x_f - x_ref)

        # Tinh sai so du
        sai_so_du_f = abs(f(x_f))

        # Tinh can tren sai so day cung
        # err_f = abs(x_k - x_(k-1))
        # Chi can biet do chenh lech nen co the dung x_f + err_f
        sai_so_uoc_luong_f = false_position_error(
            x_f, x_f + err_f, m1, M1
        )

        # In ket qua day cung
        print(
            f"{eps:<12.1e}"
            f"{'Day cung':<15}"
            f"{x_f:<22.12f}"
            f"{sai_so_tuyet_doi_f:<20.5e}"
            f"{sai_so_du_f:<18.5e}"
            f"{sai_so_uoc_luong_f:<20.5e}"
            f"{k_f:<10}"
        )

        # Tinh so buoc lap ly thuyet cua chia doi
        n_ly_thuyet = bisection_steps(a, b, eps)

        # In so buoc lap ly thuyet
        print(
            f"{'':<12}"
            f"{'Ly thuyet':<15}"
            f"{'':<22}"
            f"{'':<20}"
            f"{'':<18}"
            f"{'':<20}"
            f"{n_ly_thuyet:<10}"
        )

    # Ket thuc bang
    print("=" * 120)


# ============================================================
# PHAN 10: AP DUNG VAO BAI TOAN f1
# ============================================================

def bai_toan_f1():

    # In tieu de bai toan
    print("\n\n********** BAI TOAN 1: f1(x) = 2^x - 5x - 3 **********")

    # Tim khoang phan ly tren [-1, 0]
    khoang_f1 = isolate_roots(f1, -1, 0, 0.1)

    # In cac khoang phan ly
    print("Cac khoang phan ly cua f1:", khoang_f1)

    # Can duoi cua |f1'(x)| tren [-1, 0]
    m1 = 5 - math.log(2)

    # Can tren cua |f1'(x)| tren [-1, 0]
    M1 = 5 - math.log(2) / 2

    # Duyet cac khoang tim duoc
    for a, b in khoang_f1:

        # Neu khoang co hai dau mut trung nhau
        if a == b:
            print("Nghiem chinh xac tai:", a)
            continue

        # Goi ham danh gia khoang
        danh_gia_khoang(
            "f1(x) = 2^x - 5x - 3",
            f1,
            df1,
            a,
            b,
            -0.5,
            m1,
            M1
        )


# ============================================================
# PHAN 11: AP DUNG VAO BAI TOAN f2
# ============================================================

def bai_toan_f2():

    # In tieu de bai toan
    print("\n\n********** BAI TOAN 2: f2(x) = 5x^3 - 20x + 3 **********")

    # Tim cac khoang phan ly tren [-3, 3]
    khoang_f2 = isolate_roots(f2, -3, 3, 0.1)

    # In danh sach khoang phan ly
    print("Cac khoang phan ly cua f2:", khoang_f2)

    # Duyet tung khoang phan ly
    for a, b in khoang_f2:

        # Tinh trung diem khoang lam gia tri ban dau Newton
        x0 = (a + b) / 2

        # Khoang nghiem am
        if b < 0:

            # Can duoi cua |f2'(x)|
            m1 = abs(df2(b))

            # Can tren cua |f2'(x)|
            M1 = abs(df2(a))

        # Khoang nghiem duong gan 0
        elif a > 0 and b < 1:

            # Can duoi cua |f2'(x)|
            m1 = abs(df2(b))

            # Can tren cua |f2'(x)|
            M1 = abs(df2(a))

        # Khoang nghiem duong lon
        else:

            # Can duoi cua |f2'(x)|
            m1 = abs(df2(a))

            # Can tren cua |f2'(x)|
            M1 = abs(df2(b))

        # Goi ham danh gia khoang
        danh_gia_khoang(
            "f2(x) = 5x^3 - 20x + 3",
            f2,
            df2,
            a,
            b,
            x0,
            m1,
            M1
        )


# ============================================================
# PHAN 12: CHUONG TRINH CHINH
# ============================================================

if __name__ == "__main__":

    # Chay bai toan thu nhat
    bai_toan_f1()

    # Chay bai toan thu hai
    bai_toan_f2()

    # Thong bao ket thuc
    print("\nDA HOAN THANH CHUONG TRINH DANH GIA SAI SO.")

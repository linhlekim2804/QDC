"""Kiểm tra dữ liệu và in bảng kết quả."""

import numpy
from textwrap import fill


def validate_points(points):
    """Kiểm tra dữ liệu có ít nhất 3 điểm, mỗi điểm gồm hai tọa độ x và y."""

    danh_sach_diem = numpy.asarray(points, dtype=float)

    if danh_sach_diem.ndim != 2:
        raise ValueError("Dữ liệu điểm phải là bảng gồm các hàng và cột.")

    so_cot = danh_sach_diem.shape[1]
    if so_cot != 2:
        raise ValueError("Mỗi điểm phải có đúng hai tọa độ x và y.")

    so_luong_diem = len(danh_sach_diem)
    if so_luong_diem < 3:
        raise ValueError("Cần ít nhất 3 điểm.")

    cac_gia_tri_hop_le = numpy.isfinite(danh_sach_diem)
    tat_ca_deu_hop_le = cac_gia_tri_hop_le.all()

    if not tat_ca_deu_hop_le:
        raise ValueError("Tọa độ không được chứa NaN hoặc Inf.")

    return danh_sach_diem


def validate_features(features):
    """Kiểm tra bảng hai đặc trưng trước khi đưa sang R."""

    bang_dac_trung = numpy.asarray(features, dtype=float)

    if bang_dac_trung.ndim != 2:
        raise ValueError("Đặc trưng phải là bảng gồm các hàng và cột.")

    so_cot = bang_dac_trung.shape[1]
    if so_cot != 2:
        raise ValueError("Cần đúng hai cột đặc trưng.")

    so_luong_diem = len(bang_dac_trung)
    if so_luong_diem < 3:
        raise ValueError("Cần đặc trưng của ít nhất 3 điểm.")

    cac_gia_tri_hop_le = numpy.isfinite(bang_dac_trung)
    tat_ca_deu_hop_le = cac_gia_tri_hop_le.all()

    if not tat_ca_deu_hop_le:
        raise ValueError("Đặc trưng không được chứa NaN hoặc Inf.")

    return bang_dac_trung


def print_step_header(step, title):
    """In tên của bước đang thực hiện."""

    so_buoc = str(step)
    dong_tieu_de = "===== QDC STEP " + so_buoc + ": " + title + " ====="
    print(dong_tieu_de)


def format_number(value):
    """Làm tròn tối đa 6 chữ số thập phân khi in."""
    chuoi = format(float(value), ".6f").rstrip("0").rstrip(".")
    if chuoi == "-0":
        chuoi = "0"
    return chuoi


def detail_indices(number_of_points, limit=None):
    """Trả các chỉ số điểm cần in chi tiết; None nghĩa là in tất cả."""

    if limit is None:
        return range(number_of_points)

    if limit < 1:
        raise ValueError("Số điểm in chi tiết phải lớn hơn 0.")

    if limit < number_of_points:
        so_diem_can_in = limit
        if limit == 1:
            print("Ta xem cách tính với A1. Các điểm còn lại được tổng hợp trong bảng cuối.")
        else:
            print("Ta xem cách tính với", limit, "điểm đầu. Bảng cuối có kết quả của tất cả các điểm.")
    else:
        so_diem_can_in = number_of_points

    return range(so_diem_can_in)


def print_section(title):
    """In tiêu đề của một phần trong báo cáo."""

    duong_ke = "=" * 80
    print()
    print(duong_ke)
    print(title)
    print(duong_ke)


def print_result_table(title, header, rows):
    """In tiêu đề cột và các hàng dữ liệu."""

    print_section(title)

    bang_chuoi = []

    for hang_du_lieu in rows:
        hang_chuoi = []

        for gia_tri in hang_du_lieu:
            if isinstance(gia_tri, (float, numpy.floating)):
                gia_tri_so = float(gia_tri)

                # Chỉ định dạng số để in, không sửa dữ liệu tính toán.
                chuoi_gia_tri = format_number(gia_tri_so)
            else:
                chuoi_gia_tri = str(gia_tri)

            hang_chuoi.append(chuoi_gia_tri)

        bang_chuoi.append(hang_chuoi)

    # Căn cột theo tiêu đề hoặc giá trị dài nhất.
    so_cot = len(header)
    do_rong_cac_cot = []

    for chi_so_cot in range(so_cot):
        ten_cot = header[chi_so_cot]
        do_rong_cot = len(ten_cot)

        for hang_chuoi in bang_chuoi:
            do_dai_gia_tri = len(hang_chuoi[chi_so_cot])

            if do_dai_gia_tri > do_rong_cot:
                do_rong_cot = do_dai_gia_tri

        do_rong_cac_cot.append(do_rong_cot)

    cac_o_tieu_de = []

    for chi_so_cot in range(so_cot):
        ten_cot = header[chi_so_cot]
        do_rong_cot = do_rong_cac_cot[chi_so_cot]

        o_tieu_de = ten_cot.ljust(do_rong_cot)
        cac_o_tieu_de.append(o_tieu_de)

    dong_tieu_de = " | ".join(cac_o_tieu_de)
    print(dong_tieu_de)

    cac_doan_duong_ke = []

    for chi_so_cot in range(so_cot):
        do_rong_cot = do_rong_cac_cot[chi_so_cot]
        doan_duong_ke = "-" * do_rong_cot
        cac_doan_duong_ke.append(doan_duong_ke)

    dong_duong_ke = "-+-".join(cac_doan_duong_ke)
    print(dong_duong_ke)

    for hang_chuoi in bang_chuoi:
        cac_o_trong_hang = []

        for chi_so_cot in range(so_cot):
            chuoi_gia_tri = hang_chuoi[chi_so_cot]
            do_rong_cot = do_rong_cac_cot[chi_so_cot]
            o_du_lieu = chuoi_gia_tri.ljust(do_rong_cot)
            cac_o_trong_hang.append(o_du_lieu)

        dong_du_lieu = " | ".join(cac_o_trong_hang)
        print(dong_du_lieu)

    print()

def point_names(indices):
    """Ghép chỉ số điểm thành A1, A2, ..."""
    ten_cac_diem = []
    for i in indices:
        ten_cac_diem.append("A" + str(i + 1))
    return ", ".join(ten_cac_diem)

def print_formula(expression):
    """Xuống dòng cho phép thế số dài."""
    print(fill(expression, width=110, initial_indent="  ", subsequent_indent="    "))

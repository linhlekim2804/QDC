"""Chạy bước 1: tính kNN và frNN."""
import sys

from step1_compute_spatial_features import step1_compute_original
from utils import BASE_DIR, read_points


if __name__ == "__main__":
    # Đọc data.txt trong thư mục QDC_goc.
    file_du_lieu = BASE_DIR / "data.txt"
    file_ket_qua = BASE_DIR / "output_step1.txt"
    danh_sach_diem = read_points(file_du_lieu)

    k = 2
    r = 1.5
    so_diem_in_chi_tiet = 1

    # Ghi phần in của bước 1 vào file output.
    man_hinh = sys.stdout
    with open(file_ket_qua, "w", encoding="utf-8") as file_output:
        try:
            sys.stdout = file_output
            ket_qua = step1_compute_original(
                danh_sach_diem,
                k=k,
                radius=r,
                show_detailed=True,
                detail_limit=so_diem_in_chi_tiet
            )
        finally:
            # Trả print về màn hình, kể cả khi có lỗi.
            sys.stdout = man_hinh

    print("Da luu ket qua vao:", file_ket_qua)

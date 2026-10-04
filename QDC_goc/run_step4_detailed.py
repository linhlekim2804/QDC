"""Chạy bước 4: phân loại mật độ."""
import sys

from step1_compute_spatial_features import compute_spatial_features
from step2_normalize import normalize_features
from step3_kmeans import cluster_features
from step4_classify import step4_classify_original
from utils import BASE_DIR, read_points


if __name__ == "__main__":
    # Đọc data.txt trong thư mục QDC_goc.
    file_du_lieu = BASE_DIR / "data.txt"
    file_ket_qua = BASE_DIR / "output_step4.txt"
    danh_sach_diem = read_points(file_du_lieu)

    k = 2
    r = 1.5
    so_diem_in_chi_tiet = 1

    # Bước 1: tính kNN và frNN.
    ket_qua = compute_spatial_features(danh_sach_diem, k, r)

    # Bước 2: tính Z-score, rồi thêm vào kết quả đã có.
    ket_qua_chuan_hoa = normalize_features(
        ket_qua["knn_sum"],
        ket_qua["frnn_count"]
    )
    ket_qua.update(ket_qua_chuan_hoa)

    # Bước 3: phân cụm trên hai giá trị đã chuẩn hóa.
    ket_qua_kmeans = cluster_features(ket_qua["features"])
    ket_qua.update(ket_qua_kmeans)

    # Ghi phần in của bước 4 vào file output.
    man_hinh = sys.stdout
    with open(file_ket_qua, "w", encoding="utf-8") as file_output:
        try:
            sys.stdout = file_output
            ket_qua = step4_classify_original(
                ket_qua,
                show_detailed=True,
                detail_limit=so_diem_in_chi_tiet
            )
        finally:
            # Trả print về màn hình, kể cả khi có lỗi.
            sys.stdout = man_hinh

    print("Da luu ket qua vao:", file_ket_qua)

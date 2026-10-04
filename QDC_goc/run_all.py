"""Chạy lần lượt 4 bước QDC và lưu mỗi bước vào một file TXT."""

from contextlib import redirect_stdout
from pathlib import Path
import sys

from utils import BASE_DIR
from utils import read_points
from step1_compute_spatial_features import step1_compute_original
from step2_normalize import step2_normalize_original
from step3_kmeans import step3_kmeans_original
from step4_classify import step4_classify_original


def run_pipeline(data_path, output_dir, k=2, radius=1.5,
                 stop_after=4, rscript=None, show_detailed=True, detail_limit=None):
    """Đọc dữ liệu rồi thực hiện các bước từ 1 đến bước cần xem.

    data_path: đường dẫn file chứa các điểm.
    output_dir: thư mục lưu các file kết quả.
    k: số điểm gần nhất cần xét.
    radius: bán kính r dùng để đếm điểm.
    stop_after: chạy đến bước nào; mặc định chạy đủ 4 bước.
    rscript: đường dẫn R, để None thì chương trình tự tìm.
    show_detailed: True để in chi tiết phép tính.
    detail_limit: số điểm cần in chi tiết, None để in tất cả.
    """
    if stop_after < 1 or stop_after > 4:
        raise ValueError("Số bước cần chạy phải từ 1 đến 4.")

    duong_dan_du_lieu = Path(data_path).resolve()
    thu_muc_ket_qua = Path(output_dir).resolve()

    # Tránh ghi kết quả đè lên file dữ liệu.
    if duong_dan_du_lieu.parent == thu_muc_ket_qua:
        if duong_dan_du_lieu.name.startswith("output_step"):
            raise ValueError("File dữ liệu không được trùng với file output.")

    danh_sach_diem = read_points(duong_dan_du_lieu)
    thu_muc_ket_qua.mkdir(parents=True, exist_ok=True)
    danh_sach_file_ket_qua = []

    # redirect_stdout ghi phần in của mỗi bước vào file output tương ứng.

    # Bước 1: Tính kNN và frNN
    file_ket_qua_buoc_1 = thu_muc_ket_qua / "output_step1.txt"

    with open(file_ket_qua_buoc_1, "w", encoding="utf-8") as file_output:
        with redirect_stdout(file_output):
            ket_qua_buoc_1 = step1_compute_original(
                danh_sach_diem,
                k=k,
                radius=radius,
                show_detailed=show_detailed,
                detail_limit=detail_limit,
                rscript=rscript
            )

    danh_sach_file_ket_qua.append("output_step1.txt")
    print("Da luu ket qua buoc 1 vao:", file_ket_qua_buoc_1)

    if stop_after == 1:
        return {
            "completed_step": 1,
            "outputs": danh_sach_file_ket_qua,
            "result": ket_qua_buoc_1
        }

    # Bước 2: Chuẩn hóa Z-score
    file_ket_qua_buoc_2 = thu_muc_ket_qua / "output_step2.txt"

    with open(file_ket_qua_buoc_2, "w", encoding="utf-8") as file_output:
        with redirect_stdout(file_output):
            ket_qua_buoc_2 = step2_normalize_original(
                ket_qua_buoc_1,
                show_detailed=show_detailed,
                detail_limit=detail_limit,
                rscript=rscript
            )

    danh_sach_file_ket_qua.append("output_step2.txt")
    print("Da luu ket qua buoc 2 vao:", file_ket_qua_buoc_2)

    if stop_after == 2:
        return {
            "completed_step": 2,
            "outputs": danh_sach_file_ket_qua,
            "result": ket_qua_buoc_2
        }

    # Bước 3: Phân cụm K-Means
    file_ket_qua_buoc_3 = thu_muc_ket_qua / "output_step3.txt"

    with open(file_ket_qua_buoc_3, "w", encoding="utf-8") as file_output:
        with redirect_stdout(file_output):
            ket_qua_buoc_3 = step3_kmeans_original(
                ket_qua_buoc_2,
                show_detailed=show_detailed,
                detail_limit=detail_limit,
                rscript=rscript
            )

    danh_sach_file_ket_qua.append("output_step3.txt")
    print("Da luu ket qua buoc 3 vao:", file_ket_qua_buoc_3)

    if stop_after == 3:
        return {
            "completed_step": 3,
            "outputs": danh_sach_file_ket_qua,
            "result": ket_qua_buoc_3
        }

    # Bước 4: Tính ngưỡng và phân loại mật độ
    file_ket_qua_buoc_4 = thu_muc_ket_qua / "output_step4.txt"

    with open(file_ket_qua_buoc_4, "w", encoding="utf-8") as file_output:
        with redirect_stdout(file_output):
            ket_qua_buoc_4 = step4_classify_original(
                ket_qua_buoc_3,
                show_detailed=show_detailed,
                detail_limit=detail_limit,
                rscript=rscript
            )

    danh_sach_file_ket_qua.append("output_step4.txt")
    print("Da luu ket qua buoc 4 vao:", file_ket_qua_buoc_4)

    return {
        "completed_step": 4,
        "outputs": danh_sach_file_ket_qua,
        "result": ket_qua_buoc_4
    }


def main(stop_after=4, show_detailed=True):
    """Đặt dữ liệu, tham số rồi gọi hàm chạy các bước."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Dữ liệu và tham số chạy thử
    duong_dan_file_du_lieu = BASE_DIR / "data.txt"
    thu_muc_luu_ket_qua = BASE_DIR
    k = 2
    r = 1.5

    # K = 3 được đặt trong hàm K-Means ở bước 3.
    ket_qua_chay = run_pipeline(
        data_path=duong_dan_file_du_lieu,
        output_dir=thu_muc_luu_ket_qua,
        k=k,
        radius=r,
        stop_after=stop_after,
        show_detailed=show_detailed,
        detail_limit=1
    )

    so_buoc_da_chay = ket_qua_chay["completed_step"]
    print("Da hoan thanh", so_buoc_da_chay, "buoc QDC.")
    return ket_qua_chay


if __name__ == "__main__":
    main()

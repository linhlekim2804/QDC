"""Gọi mã R và đọc kết quả của từng bước QDC."""

from pathlib import Path
import glob
import os
import shutil
import subprocess
import tempfile
import numpy

from utils import BASE_DIR


def find_rscript(explicit=None):
    """Tìm Rscript theo đường dẫn chỉ định, PATH hoặc thư mục cài R."""

    if explicit:
        duong_dan_chi_dinh = str(explicit)
        duong_dan_tim_duoc = shutil.which(duong_dan_chi_dinh)

        if duong_dan_tim_duoc:
            return duong_dan_tim_duoc

        file_chi_dinh = Path(duong_dan_chi_dinh)

        if file_chi_dinh.is_file():
            duong_dan_day_du = file_chi_dinh.resolve()
            return str(duong_dan_day_du)

        raise FileNotFoundError("Không tìm thấy Rscript: " + duong_dan_chi_dinh)

    duong_dan_trong_path = shutil.which("Rscript")

    if duong_dan_trong_path:
        return duong_dan_trong_path

    # Tìm trong thư mục cài R nếu PATH chưa có Rscript.
    cac_mau_duong_dan = []
    cac_mau_duong_dan.append("C:/Program Files/R/R-*/bin/Rscript.exe")
    cac_mau_duong_dan.append("C:/Program Files/R/R-*/bin/x64/Rscript.exe")

    thu_muc_nguoi_dung = Path.home()
    thu_muc_cai_r = thu_muc_nguoi_dung / "AppData" / "Local" / "Programs" / "R"
    mau_cai_r_ca_nhan = thu_muc_cai_r / "R-*" / "bin" / "Rscript.exe"
    cac_mau_duong_dan.append(str(mau_cai_r_ca_nhan))

    for mau_duong_dan in cac_mau_duong_dan:
        danh_sach_tim_duoc = glob.glob(mau_duong_dan)

        danh_sach_tim_duoc.sort(reverse=True)

        if len(danh_sach_tim_duoc) > 0:
            duong_dan_rscript = danh_sach_tim_duoc[0]
            return duong_dan_rscript

    raise FileNotFoundError(
        "Chưa tìm thấy Rscript. Cài R hoặc truyền đường dẫn vào tham số rscript của hàm."
    )


def run_r_step(step, values, k=2, radius=1.5, rscript=None):
    """Ghi dữ liệu tạm, gọi R rồi đọc kết quả.

    step: số bước cần chạy, từ 1 đến 4.
    values: bảng dữ liệu đầu vào của bước đó.
    k, radius: số điểm gần nhất và bán kính r của bước 1.
    rscript: đường dẫn Rscript; để None thì tự tìm.
    """

    duong_dan_rscript = find_rscript(rscript)
    duong_dan_ma_r = BASE_DIR / "qdc_original.R"

    # Thư mục trao đổi dữ liệu với R được xóa khi ra khỏi khối with.
    with tempfile.TemporaryDirectory(prefix="qdc_r_") as ten_thu_muc_tam:
        thu_muc_tam = Path(ten_thu_muc_tam)

        file_du_lieu_vao = thu_muc_tam / "input.csv"
        numpy.savetxt(
            file_du_lieu_vao,
            values,
            delimiter=",",
            fmt="%.17g"
        )

        # --vanilla: không nạp cấu hình hoặc workspace R đã lưu.
        lenh_chay_r = []
        lenh_chay_r.append(duong_dan_rscript)
        lenh_chay_r.append("--vanilla")
        lenh_chay_r.append(str(duong_dan_ma_r))
        lenh_chay_r.append(str(step))
        lenh_chay_r.append(str(file_du_lieu_vao))
        lenh_chay_r.append(str(thu_muc_tam))
        lenh_chay_r.append(str(k))
        lenh_chay_r.append(str(radius))

        # Không mở thêm cửa sổ R trên Windows.
        tuy_chon_cua_so = 0

        if os.name == "nt":
            tuy_chon_cua_so = subprocess.CREATE_NO_WINDOW

        ket_qua_chay_r = subprocess.run(
            lenh_chay_r,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=tuy_chon_cua_so
        )

        if ket_qua_chay_r.returncode != 0:
            thong_bao_loi = "R không thực hiện được bước QDC:\n"
            thong_bao_loi = thong_bao_loi + ket_qua_chay_r.stderr
            raise RuntimeError(thong_bao_loi)

        ket_qua = {}

        for file_csv in thu_muc_tam.glob("*.csv"):
            if file_csv.name != "input.csv":
                ten_bang = file_csv.stem

                # Giữ dạng bảng khi kết quả chỉ có một hàng hoặc một cột.
                bang_so = numpy.loadtxt(
                    file_csv,
                    delimiter=",",
                    ndmin=2
                )

                ket_qua[ten_bang] = bang_so

        file_nhan_mat_do = thu_muc_tam / "labels.txt"

        if file_nhan_mat_do.exists():
            noi_dung_nhan = file_nhan_mat_do.read_text(encoding="utf-8")
            danh_sach_nhan = noi_dung_nhan.splitlines()
            ket_qua["labels"] = numpy.array(danh_sach_nhan)

        file_thong_tin_r = thu_muc_tam / "info.txt"
        noi_dung_thong_tin = file_thong_tin_r.read_text(encoding="utf-8")
        cac_dong_thong_tin = noi_dung_thong_tin.splitlines()

        thong_tin_r = {}
        thong_tin_r["R_version"] = cac_dong_thong_tin[0]
        thong_tin_r["dbscan_version"] = cac_dong_thong_tin[1]
        thong_tin_r["seconds_r"] = float(cac_dong_thong_tin[2])

        thong_bao_cua_r = ket_qua_chay_r.stderr.strip()

        if thong_bao_cua_r != "":
            thong_tin_r["messages"] = thong_bao_cua_r

        ket_qua["engine"] = thong_tin_r

    return ket_qua

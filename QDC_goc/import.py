"""Xem các điểm trong data.txt."""

from utils import BASE_DIR

from utils import read_points


if __name__ == "__main__":

    thu_muc_chua_du_lieu = BASE_DIR
    ten_file_du_lieu = "data.txt"
    duong_dan_file_du_lieu = thu_muc_chua_du_lieu / ten_file_du_lieu

    danh_sach_diem = read_points(duong_dan_file_du_lieu)

    so_luong_diem = len(danh_sach_diem)

    print("So luong diem:", so_luong_diem)
    print("Danh sach toa do cac diem:")

    for chi_so_diem in range(so_luong_diem):

        diem_dang_xet = danh_sach_diem[chi_so_diem]

        toa_do_x = diem_dang_xet[0]
        toa_do_y = diem_dang_xet[1]

        so_thu_tu_diem = chi_so_diem + 1

        print("Diem", so_thu_tu_diem, ": x =", toa_do_x, ", y =", toa_do_y)

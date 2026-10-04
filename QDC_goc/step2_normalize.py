"""
QDC - Bước 2: Chuẩn hóa hai đặc trưng bằng Z-score.

Input:
    Dict kết quả bước 1, gồm knn_sum và frnn_count.
Output:
    Các cặp (z_knn, z_frnn), trung bình và độ lệch chuẩn từng đặc trưng.
Thực hiện:
    Gọi scale() riêng cho hai cột, như mã R ở trang 11 slide tác giả.
"""
import numpy as np
from qdc_common import validate_features, print_step_header, format_number, detail_indices, print_section, print_result_table
from qdc_common import print_formula
from r_bridge import run_r_step


def normalize_features(knn_sum, frnn_count, rscript=None):
    """Tính Z-score bằng R scale(), với độ lệch chuẩn mẫu.

    z_knn[i] = (S[i] - mean_S) / sd_S
    z_frnn[i] = (F[i] - mean_F) / sd_F

    Trả về ma trận chuẩn hóa và các tham số do chính R tính.
    """
    raw_features = validate_features(np.column_stack((knn_sum, frnn_count)))
    if np.any(np.ptp(raw_features, axis=0) == 0):
        raise ValueError("Có đặc trưng hằng; không thể chuẩn hóa Z-score.")

    calculated = run_r_step(2, raw_features, rscript=rscript)
    return {
        "features": calculated["values"],
        "means": calculated["parameters"][0],
        "stds": calculated["parameters"][1],
        "engine": calculated["engine"],
    }


def print_statistics(result):
    """Giải thích lần lượt cách tính trung bình và độ lệch chuẩn mẫu."""
    n = len(result["points"])
    print(f"Số điểm: {n} | Hai giá trị cần chuẩn hóa: kNN và frNN")
    print_section("CHUẨN BỊ CHO PHÉP CHUẨN HÓA")
    print("Ở bước trước, mỗi điểm đã có một giá trị kNN và một giá trị frNN.")
    print("Để tính Z-score, trước hết ta cần trung bình và độ lệch chuẩn riêng của từng biến.")
    print("Trung bình cho biết các giá trị tập trung quanh đâu; độ lệch chuẩn cho biết mức độ phân tán.")
    print("\nCác công thức sử dụng:")
    print("  Trung bình = tổng các giá trị / n")
    print("  s = sqrt(tổng (giá trị - trung bình)^2 / (n - 1))")
    print("  z = (giá trị - trung bình) / s")
    print("Ở đây, n là số điểm và s là độ lệch chuẩn mẫu, theo cách tính của scale() trong R.")
    print("Ta tính riêng cho kNN và frNN như sau.")

    for column, (key, name) in enumerate([("knn_sum", "kNN"), ("frnn_count", "frNN")]):
        values = result[key]
        mean = result["means"][column]
        std = result["stds"][column]
        deviations = values - mean
        squared_deviations = deviations ** 2
        total = float(np.sum(values))
        squared_sum = float(np.sum(squared_deviations))
        variance = squared_sum / (n - 1)

        print_section(f"TÍNH TRUNG BÌNH VÀ ĐỘ LỆCH CHUẨN CỦA {name}")
        print(f"Với {n} điểm, ta có {n} giá trị {name}.")
        print("Đầu tiên, cộng các giá trị này rồi chia cho số điểm để tìm trung bình:")
        if n <= 30:
            addition = " + ".join(format_number(v) for v in values)
            print_formula(f"Trung bình {name} = ({addition}) / {n}")
        print(f"  Trung bình {name} = {format_number(total)} / {n} ≈ {format_number(mean)}")

        print("\nTiếp theo, lấy từng giá trị trừ trung bình rồi bình phương độ lệch đó.")
        print(f"Chẳng hạn, với giá trị {name} của A1:")
        print(f"  Độ lệch = {format_number(values[0])} - ({format_number(mean)})"
              f" ≈ {format_number(deviations[0])}")
        print(f"  Bình phương độ lệch ≈ ({format_number(deviations[0])})^2"
              f" ≈ {format_number(squared_deviations[0])}")
        print("Làm tương tự với các điểm còn lại rồi cộng các bình phương độ lệch:")
        if n <= 30:
            addition = " + ".join(format_number(v) for v in squared_deviations)
            print_formula(f"Tổng bình phương độ lệch ≈ {addition}")
        print(f"  Tổng bình phương độ lệch ≈ {format_number(squared_sum)}")

        print(f"\nVì dùng độ lệch chuẩn mẫu, ta chia tổng trên cho n - 1 = {n} - 1 = {n - 1}.")
        print("Sau đó lấy căn bậc hai để được độ lệch chuẩn:")
        print(f"  s^2 = {format_number(squared_sum)} / {n - 1} ≈ {format_number(variance)}")
        print(f"  s = sqrt({format_number(squared_sum)} / {n - 1}) ≈ {format_number(std)}")
        print(f"Vậy {name} có trung bình khoảng {format_number(mean)}"
              f" và độ lệch chuẩn khoảng {format_number(std)}.")

    print("\nĐã có trung bình và độ lệch chuẩn của cả hai biến, ta có thể tính Z-score cho từng điểm.")
    print("Các số trong lời giải được làm tròn cho dễ đọc; phép tính vẫn dùng giá trị đầy đủ.")


def print_detailed_calculation(result, limit=None):
    """Thay số và giải thích ý nghĩa của Z-score."""
    print_section("TÍNH Z-SCORE CHO ĐIỂM ĐANG XÉT")
    raw = np.column_stack((result["knn_sum"], result["frnn_count"]))
    for i in detail_indices(len(raw), limit):
        print(f"\nA{i + 1} có kNN = {format_number(raw[i, 0])} và frNN = {int(raw[i, 1])}.")
        print("Lấy mỗi giá trị trừ trung bình của biến đó, rồi chia cho độ lệch chuẩn tương ứng:")
        for column, (name, feature) in enumerate([("z_kNN", "kNN"), ("z_frNN", "frNN")]):
            value = raw[i, column]
            mean = result["means"][column]
            std = result["stds"][column]
            z = result["features"][i, column]
            print(f"  {name}(A{i + 1}) = ({format_number(value)} - ({format_number(mean)}))"
                  f" / {format_number(std)} ≈ {format_number(z)}")
            if z < 0:
                print(f"  Nghĩa là {feature} của A{i + 1} thấp hơn trung bình"
                      f" khoảng {format_number(abs(z))} độ lệch chuẩn.")
            elif z > 0:
                print(f"  Nghĩa là {feature} của A{i + 1} cao hơn trung bình"
                      f" khoảng {format_number(z)} độ lệch chuẩn.")
            else:
                print(f"  Nghĩa là {feature} của A{i + 1} bằng trung bình của biến này.")
        print(f"\nSau chuẩn hóa, A{i + 1} được biểu diễn bởi cặp"
              f" ({format_number(result['features'][i, 0])}, {format_number(result['features'][i, 1])}).")


def print_summary(result):
    """Bảng tham số và các cặp giá trị sau chuẩn hóa."""
    print_result_table("THAM SỐ CHUẨN HÓA", ["Đặc trưng", "Trung bình", "Độ lệch chuẩn mẫu"],
                       [[name, result["means"][i], result["stds"][i]]
                        for i, name in enumerate(["kNN", "frNN"])])
    print_result_table("BẢNG KẾT QUẢ STEP 2",
                       ["Điểm", "kNN", "frNN", "z_kNN", "z_frNN"],
                       [["A" + str(i + 1), result["knn_sum"][i], result["frnn_count"][i], *values]
                        for i, values in enumerate(result["features"])])
    print("Từ đây, K-Means sẽ phân cụm trên các cặp (z_kNN, z_frNN), thay cho tọa độ ban đầu.")


def step2_normalize_original(step1_result, show_detailed=False,
                             detail_limit=None, rscript=None):
    """Nhận kết quả bước 1, bổ sung đặc trưng chuẩn hóa rồi chuyển cho bước 3."""
    print_step_header(2, "CHUẨN HÓA Z-SCORE")
    normalized = normalize_features(step1_result["knn_sum"], step1_result["frnn_count"], rscript)

    result = dict(step1_result)
    result.update(normalized)
    print_statistics(result)
    if show_detailed:
        print_detailed_calculation(result, detail_limit)
    print_summary(result)
    return result


if __name__ == "__main__":
    from run_all import main
    main(stop_after=2, show_detailed=True)

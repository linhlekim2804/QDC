"""
QDC - Bước 4: Tính ngưỡng và gán nhãn mật độ.

Input:
    Dict kết quả bước 3, gồm đặc trưng chuẩn hóa và mã cụm K-Means.
Output:
    Hai ngưỡng t1, t2 và nhãn low-density / mid-density / high-density.
Thực hiện:
    max(min(...)) và ifelse() trong R như mã ở trang 11 slide tác giả.
"""
import numpy as np
from qdc_common import validate_features, print_step_header, format_number, detail_indices, print_section, print_result_table, point_names
from qdc_common import print_formula
from r_bridge import run_r_step


def classify_density(features, clusters, rscript=None):
    """Phân loại bằng hai ngưỡng được rút ra từ ba cụm K-Means.

    t1 = max_c min(z_knn trong cụm c)
    t2 = max_c min(z_frnn trong cụm c)

    Nếu z_knn > t1: Low.
    Nếu không thuộc Low và z_frnn > t2: High.
    Còn lại: Medium.
    """
    features = validate_features(features)
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or len(clusters) != len(features):
        raise ValueError("Cần một mã cụm cho mỗi điểm.")
    if set(clusters.tolist()) != {1, 2, 3}:
        raise ValueError("Cần đủ ba cụm có mã 1, 2, 3.")

    calculated = run_r_step(4, np.column_stack((features, clusters)), rscript=rscript)
    return {
        "thresholds": calculated["thresholds"][0],
        "cluster_ids": calculated["minima"][:, 0].astype(int),
        "minima": calculated["minima"][:, 1:],
        "labels": calculated["labels"],
        "engine": calculated["engine"],
    }


def density_name(label):
    """Tên mật độ để in trong báo cáo."""
    if label == "low-density":
        return "Thấp"
    if label == "high-density":
        return "Cao"
    return "Trung bình"


def print_statistics(result):
    """Lấy min trong từng cụm, rồi lấy max để xác định ngưỡng."""
    print(f"Số điểm: {len(result['features'])} | Số cụm đầu vào: 3")
    print_section("CÔNG THỨC TÍNH NGƯỠNG")
    print("Từ ba cụm vừa tìm được, ta cần xác định hai ngưỡng để xét mật độ của từng điểm.")
    print("Trước hết, tìm giá trị nhỏ nhất của mỗi đặc trưng trong từng cụm.")
    print("Sau đó, lấy giá trị lớn nhất trong ba giá trị nhỏ nhất để được ngưỡng tương ứng:")
    print("m_kNN[c] = min(z_kNN của các điểm trong cụm c)")
    print("m_frNN[c] = min(z_frNN của các điểm trong cụm c)")
    print("t1 = max(m_kNN[1], m_kNN[2], m_kNN[3])")
    print("t2 = max(m_frNN[1], m_frNN[2], m_frNN[3])")
    print("\nTa thay các giá trị của từng cụm vào công thức:")
    for row, cluster_id in enumerate(result["cluster_ids"]):
        indices = np.flatnonzero(result["clusters"] == cluster_id)
        print(f"\nCụm {cluster_id}: {point_names(indices)}.")
        for column, name in enumerate(["z_kNN", "z_frNN"]):
            values = result["features"][indices, column]
            if len(values) <= 30:
                members = ", ".join(format_number(v) for v in values)
                print_formula(f"min({name}) = min({members}) ≈ {format_number(result['minima'][row, column])}")
            else:
                print(f"  min({name}) = {format_number(result['minima'][row, column])}")
    print_result_table("GIÁ TRỊ NHỎ NHẤT TRONG TỪNG CỤM",
                       ["Cụm", "min z_kNN", "min z_frNN"],
                       [[c, *values] for c, values in zip(result["cluster_ids"], result["minima"])])
    for column, name in enumerate(["t1", "t2"]):
        values = ", ".join(format_number(v) for v in result["minima"][:, column])
        print(f"{name} = max({values}) ≈ {format_number(result['thresholds'][column])}")
    print_result_table("NGƯỠNG PHÂN LOẠI", ["t1", "t2"], [result["thresholds"]])
    print("Sau khi có t1 và t2, ta xét mỗi điểm theo thứ tự sau:")
    print("  Nếu z_kNN lớn hơn t1, điểm được xếp vào mức mật độ thấp.")
    print("  Nếu chưa thuộc mức thấp, xét tiếp z_frNN: lớn hơn t2 thì mật độ cao.")
    print("  Các điểm còn lại có mật độ trung bình.")
    print("Chú ý dấu >: giá trị bằng ngưỡng chưa thỏa điều kiện đang xét.")
    print("Các phép so sánh dùng số đầy đủ, không dùng số đã làm tròn trên báo cáo.")


def print_detailed_calculation(result, limit=None):
    """So sánh từng đặc trưng với ngưỡng theo đúng thứ tự."""
    print_section("THAY SỐ VÀ KẾT LUẬN MẬT ĐỘ")
    t1, t2 = result["thresholds"]
    for i in detail_indices(len(result["features"]), limit):
        z_knn, z_frnn = result["features"][i]
        print(f"\nXét A{i + 1}: Z = ({format_number(z_knn)}, {format_number(z_frnn)}).")
        print("Trước hết, so sánh z_kNN với t1 để xét mức mật độ thấp.")
        low = "Đúng" if z_knn > t1 else "Sai"
        print(f"  z_kNN > t1: {format_number(z_knn)} > {format_number(t1)} -> {low}.")
        if z_knn <= t1:
            print("  Điều kiện trên chưa thỏa, nên xét tiếp z_frNN với t2.")
            high = "Đúng" if z_frnn > t2 else "Sai"
            print(f"  z_frNN > t2: {format_number(z_frnn)} > {format_number(t2)} -> {high}.")
        print(f"  Vậy A{i + 1} có mật độ {density_name(result['labels'][i]).lower()}.")


def print_summary(result):
    """Bảng kết quả và danh sách điểm theo từng mức mật độ."""
    rows = []
    for i, point in enumerate(result["points"]):
        rows.append(["A" + str(i + 1), *point, result["knn_sum"][i], result["frnn_count"][i],
                     *result["features"][i], result["clusters"][i], density_name(result["labels"][i])])
    print("\nÁp dụng cùng quy tắc cho các điểm, ta có kết quả phân loại dưới đây.")
    print_result_table("BẢNG KẾT QUẢ STEP 4",
                       ["Điểm", "x", "y", "kNN", "frNN", "z_kNN", "z_frNN", "Cụm", "Mật độ"], rows)
    groups = []
    for label in ["high-density", "mid-density", "low-density"]:
        indices = np.flatnonzero(result["labels"] == label)
        names = point_names(indices) if len(indices) else "Không có"
        groups.append([density_name(label), len(indices), names])
    print_result_table("TỔNG KẾT PHÂN LOẠI MẬT ĐỘ", ["Mật độ", "Số điểm", "Các điểm"], groups)
    print(f"Tổng cộng {len(result['labels'])} điểm. Hoàn thành 4 bước QDC.")


def step4_classify_original(step3_result, show_detailed=False,
                            detail_limit=None, rscript=None):
    """Hoàn thành QDC bằng cách bổ sung ngưỡng và nhãn mật độ cuối."""
    print_step_header(4, "PHÂN LOẠI MẬT ĐỘ")
    classified = classify_density(step3_result["features"], step3_result["clusters"], rscript)
    result = dict(step3_result)
    result.update(classified)
    print_statistics(result)
    if show_detailed:
        print_detailed_calculation(result, detail_limit)
    print_summary(result)
    return result


if __name__ == "__main__":
    from run_all import main
    main(stop_after=4, show_detailed=True)

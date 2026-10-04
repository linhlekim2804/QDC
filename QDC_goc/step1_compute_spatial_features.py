"""
QDC - Bước 1: Tính hai đặc trưng không gian.

Input:
    Tập điểm (x, y), số điểm gần nhất k và bán kính cố định r.
Output:
    Tổng khoảng cách kNN và số điểm trong bán kính r của mỗi điểm.
Thực hiện:
    Gọi dbscan::kNNdist() và dbscan::frNN() của R như mã tác giả.
"""
import numpy as np
from qdc_common import validate_points, print_step_header, format_number, detail_indices, print_section, print_result_table, point_names
from r_bridge import run_r_step


def compute_spatial_features(points, k, radius, rscript=None):
    """Tính hai đặc trưng bằng kNNdist() và frNN() của R.

    S[i] = tổng khoảng cách từ Pi đến k điểm gần nhất khác Pi.
    F[i] = số điểm Pj khác Pi có d(Pi, Pj) <= r.
    """
    points = validate_points(points)
    if not isinstance(k, (int, np.integer)) or not 1 <= k < len(points):
        raise ValueError("k phải là số nguyên thỏa mãn 1 <= k < n.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Bán kính r phải dương và hữu hạn.")

    calculated = run_r_step(1, points, k=k, radius=radius, rscript=rscript)
    raw_features = calculated["values"]
    return {
        "points": points,
        "k": int(k),
        "r": float(radius),
        "K": 3,
        "knn_sum": raw_features[:, 0],
        "frnn_count": raw_features[:, 1].astype(int),
        "knn_distances": calculated["distances"],
        "engine": calculated["engine"],
    }


def print_statistics(result):
    """In dữ liệu, tham số và công thức."""
    print(f"Số điểm: {len(result['points'])} | k = {result['k']} | r = {result['r']}")
    print_result_table("CÁC ĐIỂM ĐẦU VÀO", ["Điểm", "x", "y"],
                       [["A" + str(i + 1), *point] for i, point in enumerate(result["points"])])
    print_section("CÔNG THỨC")
    print("Trước hết, ta dùng khoảng cách Euclid để biết các điểm nằm cách nhau bao xa.")
    print("d(Ai, Aj) = sqrt((xi - xj)^2 + (yi - yj)^2)")
    print("kNN(Ai) = tổng khoảng cách đến k điểm gần nhất khác Ai.")
    print("frNN(Ai) = số điểm khác Ai có khoảng cách <= r.")
    print("Không đếm chính Ai; điểm khác trùng tọa độ vẫn được tính.")
    print("Các số được làm tròn tối đa 6 chữ số thập phân cho dễ đọc; phép tính vẫn dùng số đầy đủ.")


def distances_for_display(points, i):
    """Tính khoảng cách để trình bày lời giải, không thay kết quả kNN/frNN của R."""
    distances = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
    distances[i] = np.inf
    return distances


def print_detailed_calculation(result, limit=None):
    """Thay tọa độ vào công thức rồi chọn k điểm và đếm điểm trong bán kính."""
    print_section("CÁCH TÍNH CHO MỘT ĐIỂM")
    points = result["points"]
    k = result["k"]
    r = result["r"]
    for i in detail_indices(len(points), limit):
        x, y = points[i]
        distances = distances_for_display(points, i)
        print(f"\nXét A{i + 1} = ({format_number(x)}, {format_number(y)}).")
        print("\nĐầu tiên, tính khoảng cách từ điểm này đến từng điểm còn lại:")
        for j, point in enumerate(points):
            if j == i:
                continue
            xj, yj = point
            print(f"  d(A{i + 1}, A{j + 1})"
                  f" = sqrt(({format_number(x)} - ({format_number(xj)}))^2"
                  f" + ({format_number(y)} - ({format_number(yj)}))^2)"
                  f" ≈ {format_number(distances[j])}")

        nearest = np.argsort(distances, kind="stable")[:k]
        print(f"\nVì k = {k}, ta chọn {k} điểm gần nhất: {point_names(nearest)}.")
        print("Cộng các khoảng cách tương ứng lại, ta được giá trị kNN:")
        addition = " + ".join(format_number(d) for d in result["knn_distances"][i])
        print(f"  kNN(A{i + 1}) = {addition} ≈ {format_number(result['knn_sum'][i])}")

        print(f"\nTiếp theo là frNN. Với r = {format_number(r)}, ta xét các khoảng cách vừa tính.")
        print("Điểm nào có khoảng cách nhỏ hơn hoặc bằng r thì được đếm:")
        inside = []
        for j in range(len(points)):
            if j == i:
                continue
            if distances[j] <= r:
                inside.append(j)
                print(f"  A{j + 1}: {format_number(distances[j])} <= {format_number(r)} -> đếm.")
            else:
                print(f"  A{j + 1}: {format_number(distances[j])} > {format_number(r)} -> không đếm.")
        print("  Các điểm trong bán kính:", (point_names(inside) if inside else "không có") + ".")
        print(f"  frNN(A{i + 1}) = {result['frnn_count'][i]}.")
        print(f"\nVậy A{i + 1} có (kNN, frNN)"
              f" = ({format_number(result['knn_sum'][i])}, {result['frnn_count'][i]}).")


def print_summary(result):
    """Liệt kê các điểm gần nhất và hai đặc trưng."""
    rows = []
    for i in range(len(result["points"])):
        distances = distances_for_display(result["points"], i)
        nearest = np.argsort(distances, kind="stable")[:result["k"]]
        values = ", ".join(format_number(d) for d in result["knn_distances"][i])
        rows.append(["A" + str(i + 1), point_names(nearest), values,
                     result["knn_sum"][i], result["frnn_count"][i]])
    print("\nTính tương tự cho các điểm, ta thu được bảng kNN và frNN dưới đây.")
    print_result_table("BẢNG KẾT QUẢ STEP 1",
                       ["Điểm", "k điểm gần nhất", "Khoảng cách", "kNN", "frNN"], rows)
    print("Nếu nhiều điểm có cùng khoảng cách, bảng chọn theo số thứ tự để minh họa.")
    print("Đổi điểm tại vị trí bằng nhau không đổi tổng kNN.")
    print("Đến đây, mỗi điểm đã có hai giá trị kNN và frNN. Bước tiếp theo là chuẩn hóa chúng bằng Z-score.")


def step1_compute_original(points, k=2, radius=1.5, show_detailed=False,
                           detail_limit=None, rscript=None):
    """Thực hiện bước 1, in kết quả và trả dict cho bước chuẩn hóa.

    Các giá trị mặc định k=2, r=1.5 phục vụ bộ điểm minh họa trong data.txt.
    Đây không phải tham số bắt buộc của QDC.
    """
    print_step_header(1, "TÍNH kNN VÀ frNN")
    result = compute_spatial_features(points, k, radius, rscript)
    print_statistics(result)
    if show_detailed:
        print_detailed_calculation(result, detail_limit)
    print_summary(result)
    return result


if __name__ == "__main__":
    from run_all import main
    main(stop_after=1, show_detailed=True)

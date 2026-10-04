"""
QDC - Bước 3: Phân cụm K-Means trên hai đặc trưng đã chuẩn hóa.

Input:
    Dict kết quả bước 2, chứa các vector (z_knn, z_frnn).
Output:
    Mã cụm, tâm cụm, số vòng lặp và tổng bình phương khoảng cách trong cụm.
Thực hiện:
    Gọi stats::kmeans(features, centers=3) với mặc định của R.
"""
import numpy as np
from qdc_common import validate_features, print_step_header, format_number, detail_indices, print_section, print_result_table, point_names
from qdc_common import print_formula
from r_bridge import run_r_step, find_rscript


def cluster_features(features, rscript=None):
    """Chạy K-Means như lời gọi trong mã tác giả.

    Mục tiêu: giảm tổng ||Zi - tâm cụm của Zi||².
    R mặc định dùng Hartigan-Wong, nstart=1, iter.max=10.
    Tâm ban đầu được R chọn ngẫu nhiên theo mặc định của kmeans().
    """
    features = validate_features(features)
    if len(np.unique(features, axis=0)) < 3:
        raise ValueError("Cần ít nhất 3 vector đặc trưng khác nhau.")
    calculated = run_r_step(3, features, rscript=rscript)
    return {
        "clusters": calculated["clusters"][:, 0].astype(int),
        "centers": calculated["centers"],
        "iterations": int(calculated["diagnostics"][0, 0]),
        "tot_withinss": float(calculated["diagnostics"][0, 1]),
        "engine": calculated["engine"],
    }


def print_statistics(result):
    """Giới thiệu đầu vào và các tâm cuối cùng."""
    print(f"Số điểm: {len(result['features'])} | K = 3")
    print_section("CÔNG THỨC VÀ KẾT QUẢ K-MEANS")
    print("Sau chuẩn hóa, mỗi điểm được biểu diễn bởi Z(Ai) = (z_kNN(Ai), z_frNN(Ai)).")
    print("Ta đưa các cặp giá trị này vào K-Means để chia thành 3 cụm.")
    print("Tâm của mỗi cụm được tính bằng trung bình của các điểm thuộc cụm đó:")
    print("Cj = (trung bình z_kNN trong cụm j, trung bình z_frNN trong cụm j).")
    print("d(Ai, Cj) = sqrt((z_kNN(Ai) - Cj[1])^2 + (z_frNN(Ai) - Cj[2])^2).")
    print("R chạy kmeans(..., centers=3), dùng Hartigan-Wong và chọn tâm đầu ngẫu nhiên.")
    print(f"Số vòng lặp R trả về: {result['iterations']}.")
    print("Dưới đây, ta dùng các tâm cuối cùng để xem lại khoảng cách và cách tính trung bình trong từng cụm.")
    print("Lời gọi R hiện tại chỉ trả kết quả cuối, nên phần này không thể hiện từng lần cập nhật tâm.")
    print("\nCác tâm cuối cùng:")
    for j, center in enumerate(result["centers"], 1):
        print(f"  C{j} = ({format_number(center[0])}, {format_number(center[1])})")


def print_detailed_calculation(result, limit=None):
    """Tính khoảng cách đến các tâm và diễn giải cách tính tâm cụm."""
    print_section("KHOẢNG CÁCH TỪ ĐIỂM ĐẾN CÁC TÂM")
    for i in detail_indices(len(result["features"]), limit):
        z_knn, z_frnn = result["features"][i]
        print(f"\nXét A{i + 1}: Z = ({format_number(z_knn)}, {format_number(z_frnn)}).")
        distances = []
        for cluster_id, center in enumerate(result["centers"], 1):
            distance = float(np.linalg.norm(result["features"][i] - center))
            distances.append(distance)
            print(f"  d(A{i + 1}, C{cluster_id})"
                  f" = sqrt(({format_number(z_knn)} - ({format_number(center[0])}))^2"
                  f" + ({format_number(z_frnn)} - ({format_number(center[1])}))^2)"
                  f" ≈ {format_number(distance)}")
        assigned = int(result["clusters"][i])
        if np.isclose(distances[assigned - 1], min(distances), rtol=1e-12, atol=1e-12):
            print(f"  C{assigned} có khoảng cách nhỏ nhất đến A{i + 1};"
                  f" kết quả K-Means xếp điểm này vào cụm {assigned}.")
        else:
            print(f"  Theo kết quả R, A{i + 1} thuộc cụm {assigned}.")

    print_section("CÁC ĐIỂM TRONG CỤM VÀ CÁCH TÍNH TÂM")
    print("Để thấy các tâm trên được tính như thế nào, ta xét các điểm trong từng cụm.")
    print("Với mỗi đặc trưng, cộng các giá trị trong cụm rồi chia cho số điểm của cụm đó.")
    for cluster_id, center in enumerate(result["centers"], 1):
        indices = np.flatnonzero(result["clusters"] == cluster_id)
        count = len(indices)
        print(f"\nCụm {cluster_id}: {point_names(indices)}.")
        print(f"Số điểm trong cụm {cluster_id}: {count}.")
        for column, name in enumerate(["z_kNN", "z_frNN"]):
            values = result["features"][indices, column]
            total = float(np.sum(values))
            if count <= 30:
                addition = " + ".join("(" + format_number(v) + ")" for v in values)
                print_formula(f"Tâm {name} = ({addition}) / {count}")
            print(f"  Tâm {name} = {format_number(total)} / {count}"
                  f" ≈ {format_number(center[column])}")
        print(f"  Vậy C{cluster_id} = ({format_number(center[0])}, {format_number(center[1])}).")


def print_summary(result):
    """Bảng tâm cụm và khoảng cách của tất cả các điểm."""
    centers = []
    for cluster_id, center in enumerate(result["centers"], 1):
        count = int(np.count_nonzero(result["clusters"] == cluster_id))
        centers.append([cluster_id, count, *center])
    print_result_table("BẢNG TỔNG KẾT TÂM CỤM",
                       ["Cụm", "Số điểm", "Tâm z_kNN", "Tâm z_frNN"], centers)

    rows = []
    for i, values in enumerate(result["features"]):
        distances = []
        for center in result["centers"]:
            distances.append(float(np.linalg.norm(values - center)))
        rows.append(["A" + str(i + 1), *values, *distances, result["clusters"][i]])
    print_result_table("BẢNG KẾT QUẢ STEP 3",
                       ["Điểm", "z_kNN", "z_frNN", "d(C1)", "d(C2)", "d(C3)", "Cụm"], rows)
    print(f"Tổng bình phương khoảng cách trong cụm: {format_number(result['tot_withinss'])}.")
    print("Cột Số điểm cho biết mỗi cụm có bao nhiêu điểm.")
    print("Sau K-Means, ta đã biết mỗi điểm thuộc cụm nào. Để xác định mật độ, bước 4 sẽ tính hai ngưỡng.")


def step3_kmeans_original(step2_result, show_detailed=False,
                          detail_limit=None, rscript=None):
    """Nhận kết quả bước 2 và bổ sung kết quả K-Means để bước 4 phân loại."""
    print_step_header(3, "PHÂN CỤM K-MEANS")
    clustered = cluster_features(step2_result["features"], rscript=rscript)
    result = dict(step2_result)
    result.update(clustered)
    print_statistics(result)
    if show_detailed:
        print_detailed_calculation(result, detail_limit)
    print_summary(result)
    return result


if __name__ == "__main__":
    from run_all import main
    main(stop_after=3, show_detailed=True)

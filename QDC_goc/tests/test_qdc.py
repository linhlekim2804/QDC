"""Kiểm tra quy trình R, dữ liệu biên và cách tổ chức đúng 4 bước."""
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import BASE_DIR, read_points
from r_bridge import find_rscript
from step1_compute_spatial_features import compute_spatial_features, step1_compute_original
from step2_normalize import normalize_features
from step4_classify import classify_density
from run_all import run_pipeline
from qdc_common import format_number
from step1_compute_spatial_features import print_detailed_calculation as print_step1_detail


def read_report_table(path, title):
    """Đọc bảng trong báo cáo để tiếp tục đối chiếu số liệu với R độc lập."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = lines.index(title)
    header = [name.strip() for name in lines[start + 2].split("|")]
    rows = []
    for line in lines[start + 4:]:
        if not line.strip():
            break
        rows.append(dict(zip(header, [value.strip() for value in line.split("|")])))
    return rows


class QDCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Kiểm tra này cần R thực sự; không thay bằng một backend khác.
        try:
            cls.rscript = find_rscript()
        except FileNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_spatial_features_match_geometry_at_duplicates_and_boundary(self):
        points = np.array([[0, 0], [0, 0], [1, 0], [2, 0], [0, 3]], dtype=float)
        distance_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distance_matrix, np.inf)
        expected_sums = np.sort(distance_matrix, axis=1)[:, :2].sum(axis=1)
        result = compute_spatial_features(points, 2, 1, self.rscript)
        np.testing.assert_allclose(result["knn_sum"], expected_sums)
        np.testing.assert_array_equal(result["frnn_count"], (distance_matrix <= 1).sum(axis=1))
        # k = 1 phải vẫn trả về một cột khoảng cách cho mỗi điểm.
        single = compute_spatial_features(points, 1, 1, self.rscript)
        self.assertEqual(single["knn_distances"].shape, (5, 1))
        report = io.StringIO()
        original_sums = result["knn_sum"].copy()
        with redirect_stdout(report):
            print_step1_detail(result, limit=1)
        self.assertIn("d(A1, A2)", report.getvalue())
        self.assertIn("Các điểm trong bán kính: A2, A3.", report.getvalue())
        self.assertIn("frNN(A1) = 2.", report.getvalue())
        np.testing.assert_array_equal(result["knn_sum"], original_sums)

    def test_slide_values_and_strict_thresholds(self):
        # Mẫu cố định để kiểm tra dấu >, độc lập với data.txt dùng minh họa.
        points = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [4, 1],
                           [4, 2], [5, 1], [7, 4], [9, 5], [9, 1]], dtype=float)
        raw = compute_spatial_features(points, 2, 1.5, self.rscript)
        np.testing.assert_allclose(raw["knn_sum"], [
            2, 2, 2, 2, 2, 1+np.sqrt(2), 1+np.sqrt(2),
            np.sqrt(5)+np.sqrt(13), np.sqrt(5)+4, 4+np.sqrt(13)])
        np.testing.assert_array_equal(raw["frnn_count"], [3, 3, 3, 3, 2, 2, 2, 0, 0, 0])
        normalized = normalize_features(raw["knn_sum"], raw["frnn_count"], self.rscript)
        np.testing.assert_allclose(normalized["means"], [3.4511665630673747, 1.8])
        np.testing.assert_allclose(normalized["stds"], [2.1961789369887406, 1.3165611772087666])
        groups = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        classified = classify_density(normalized["features"], groups, self.rscript)
        self.assertEqual(classified["labels"].tolist(), ["mid-density"]*8 + ["low-density"]*2)

    def test_low_priority_and_equality(self):
        features = np.array([[0, 0], [1, 2], [0, 3], [2, 4]], dtype=float)
        # minima = (0,0), (1,2), (0,3), nên t1=1, t2=3.
        result = classify_density(features, [1, 2, 3, 3], self.rscript)
        self.assertEqual(result["labels"].tolist(),
                         ["mid-density", "mid-density", "mid-density", "low-density"])

    def test_constant_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_features([1, 2, 3], [0, 0, 0], self.rscript)

    def test_detailed_output_and_partial_steps(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            run_result = run_pipeline(BASE_DIR / "data.txt", folder, stop_after=2,
                                rscript=self.rscript, show_detailed=True)
            self.assertEqual(run_result["completed_step"], 2)
            first = read_report_table(Path(folder) / "output_step1.txt", "BẢNG KẾT QUẢ STEP 1")
            self.assertIn("frNN", first[0])
            report = (Path(folder) / "output_step2.txt").read_text(encoding="utf-8")
            self.assertIn("z_kNN(A1) = (", report)
            self.assertFalse((Path(folder) / "output_step3.txt").exists())
            self.assertEqual(run_result["outputs"], ["output_step1.txt", "output_step2.txt"])
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()),
                             ["output_step1.txt", "output_step2.txt"])

    def test_features_and_classification_match_R_for_actual_partition(self):
        # Tọa độ kiểm tra cố định bằng công thức, không điều khiển khởi tạo K-Means.
        dense = np.arange(35)
        medium = np.arange(20)
        sparse = np.arange(15)
        generated = np.vstack((
            np.column_stack((0.12 * np.cos(dense), 0.12 * np.sin(dense))),
            np.column_stack((2 + 0.7 * np.cos(medium), 2 + 0.7 * np.sin(medium))),
            np.column_stack((-4 + sparse * 0.7, -3 + (sparse * 3 % 11) * 0.8)),
        ))
        datasets = [(read_points(BASE_DIR / "data.txt"), 2, 1.5), (generated, 5, 0.5)]
        for points, k, radius in datasets:
            with self.subTest(n=len(points)), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                input_path = folder / "points.csv"
                np.savetxt(input_path, points, delimiter=",", header="x,y", comments="", fmt="%.17g")
                result_dir = folder / "result"
                result_dir.mkdir()
                with redirect_stdout(io.StringIO()):
                    run_result = run_pipeline(input_path, result_dir, k=k, radius=radius,
                                            rscript=self.rscript)
                # Giữ phân hoạch thực tế để đối chiếu đúng bước phân loại.
                actual = read_report_table(result_dir / "output_step4.txt", "BẢNG KẾT QUẢ STEP 4")
                assigned = np.array([int(row["Cụm"]) for row in actual])
                cluster_path = folder / "clusters.txt"
                np.savetxt(cluster_path, assigned, fmt="%d")
                reference_path = folder / "reference.csv"
                process = subprocess.run(
                    [self.rscript, "--vanilla", str(BASE_DIR / "tests/reference_author.R"),
                     str(input_path), str(reference_path), str(k), str(radius), str(cluster_path)],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                with reference_path.open(encoding="utf-8") as stream:
                    reference = list(csv.DictReader(stream))
                self.assertEqual(len(reference), len(actual))
                calculated = run_result["result"]
                raw_values = {
                    "knn_sum": calculated["knn_sum"],
                    "frnn_count": calculated["frnn_count"],
                    "z_knn": calculated["features"][:, 0],
                    "z_frnn": calculated["features"][:, 1],
                }
                mapping = [
                    ("knndist1", "knn_sum", "kNN"),
                    ("frnn1", "frnn_count", "frNN"),
                    ("knndist1.scaled", "z_knn", "z_kNN"),
                    ("frnn1.scaled", "z_frnn", "z_frNN"),
                ]
                density_labels = {"Thấp": "low-density", "Trung bình": "mid-density",
                                  "Cao": "high-density"}
                for i, (left, right) in enumerate(zip(reference, actual)):
                    for old_name, result_name, column_name in mapping:
                        np.testing.assert_allclose(float(left[old_name]), raw_values[result_name][i],
                                                   rtol=1e-12, atol=1e-12)
                        self.assertEqual(right[column_name], format_number(raw_values[result_name][i]))
                    self.assertEqual(right["Điểm"], "A" + str(i + 1))
                    self.assertEqual(int(left["km.set1"]), int(right["Cụm"]))
                    self.assertEqual(left["outcome.set1"], density_labels[right["Mật độ"]])
                    self.assertEqual(left["outcome.set1"], calculated["labels"][i])
                self.assertEqual(run_result["completed_step"], 4)
                expected = [f"output_step{step}.txt" for step in range(1, 5)]
                self.assertEqual(run_result["outputs"], expected)
                self.assertEqual(sorted(p.name for p in result_dir.iterdir()), expected)
                centers = read_report_table(result_dir / "output_step3.txt", "BẢNG TỔNG KẾT TÂM CỤM")
                self.assertEqual(sum(int(row["Số điểm"]) for row in centers), len(points))
                features = calculated["features"]
                step3_rows = read_report_table(result_dir / "output_step3.txt", "BẢNG KẾT QUẢ STEP 3")
                np.testing.assert_array_equal([int(row["Cụm"]) for row in step3_rows], assigned)
                np.testing.assert_array_equal(calculated["clusters"], assigned)
                for row in centers:
                    cluster_id = int(row["Cụm"])
                    members = features[assigned == cluster_id]
                    center = calculated["centers"][cluster_id - 1]
                    self.assertEqual(len(members), int(row["Số điểm"]))
                    np.testing.assert_allclose(center, members.mean(axis=0), rtol=1e-12, atol=1e-12)
                    self.assertEqual(row["Tâm z_kNN"], format_number(center[0]))
                    self.assertEqual(row["Tâm z_frNN"], format_number(center[1]))
                for i, row in enumerate(step3_rows):
                    for cluster_id, center in enumerate(calculated["centers"], 1):
                        distance = np.linalg.norm(features[i] - center)
                        self.assertEqual(row[f"d(C{cluster_id})"], format_number(distance))
                thresholds = read_report_table(result_dir / "output_step4.txt", "NGƯỠNG PHÂN LOẠI")
                self.assertEqual(thresholds[0]["t1"], format_number(calculated["thresholds"][0]))
                self.assertEqual(thresholds[0]["t2"], format_number(calculated["thresholds"][1]))
                grouped = read_report_table(result_dir / "output_step4.txt", "TỔNG KẾT PHÂN LOẠI MẬT ĐỘ")
                for row in grouped:
                    indices = np.flatnonzero(calculated["labels"] == density_labels[row["Mật độ"]])
                    self.assertEqual(int(row["Số điểm"]), len(indices))
                    names = ", ".join("A" + str(i + 1) for i in indices) if len(indices) else "Không có"
                    self.assertEqual(row["Các điểm"], names)



if __name__ == "__main__":
    unittest.main()

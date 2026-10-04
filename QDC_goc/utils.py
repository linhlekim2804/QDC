"""Đọc tọa độ và ghi bảng TXT."""
from pathlib import Path
import csv
import re
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

def read_points(path):
    """Đọc hai cột x, y; cho phép tiêu đề và các dòng chú thích bắt đầu bằng #."""
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[,;\s]+", line)
        if not rows and [s.lower() for s in parts] == ["x", "y"]:
            continue
        if len(parts) != 2:
            raise ValueError(f"Dong {line_number}: can dung hai cot x, y.")
        try:
            rows.append([float(value) for value in parts])
        except ValueError as exc:
            raise ValueError(f"Dong {line_number}: toa do khong hop le.") from exc
    points = np.asarray(rows, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Can it nhat 3 diem voi hai cot x, y.")
    if not np.isfinite(points).all():
        raise ValueError("Toa do phai huu han, khong chua NaN/Inf.")
    return points

def write_table(path, header, rows):
    """Ghi bảng TXT phân cách bằng tab, giữ độ chính xác của số thực."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

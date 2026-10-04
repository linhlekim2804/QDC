# QDC gốc — cấu trúc 4 bước, dùng các hàm R của tác giả

Các file bước được viết theo cách trình bày trong hai file mẫu bạn gửi:
mô tả Input/Output, các hàm tính toán, phần in thống kê/chi tiết,
hàm thực hiện trọn bước trả về dict và phần chạy riêng.

## Bốn bước

| Bước | File Python | Phép tính thực hiện trong R |
|---|---|---|
| 1 | step1_compute_spatial_features.py | dbscan::kNNdist(all=TRUE), apply(..., sum), dbscan::frNN(eps=r), đếm số điểm |
| 2 | step2_normalize.py | scale() riêng cho hai đặc trưng |
| 3 | step3_kmeans.py | stats::kmeans(features, centers=3) |
| 4 | step4_classify.py | max của các min trong cụm; ifelse() để gán nhãn |

Mỗi bước có hàm chính `stepN_..._original()`, `print_statistics()`
và `print_detailed_calculation()`. Hàm chính nhận kết quả bước trước,
in nội dung và trả dict cho bước tiếp theo.

`qdc_original.R` chứa phần tính toán R, chia thành đúng bốn hàm bước tương ứng.
`r_bridge.py` chỉ truyền dữ liệu giữa Python và R. Chúng là file hỗ trợ,
không phải bước thứ năm hoặc một thuật toán cải tiến.
`qdc_common.py` chứa các hàm kiểm tra và trình bày dùng chung.

## Chạy ví dụ

```powershell
cd D:\DATN\QDC\QDC_goc
python run_all.py
```

Hoặc chạy đến từng bước:

```powershell
python run_step1_detailed.py
python run_step2_detailed.py
python run_step3_detailed.py
python run_step4_detailed.py
```

Các file `run_stepN_detailed.py` viết tuần tự trong khối `if __name__ == "__main__":`.
Bạn sửa `file_du_lieu`, `k`, `r` và `so_diem_in_chi_tiet` ngay trong file chạy.
Các lệnh đọc dữ liệu, tính bước trước và ghi kết quả được đặt theo thứ tự thực hiện.

Các hàm `compute_spatial_features()`, `normalize_features()`, `cluster_features()`
tính dữ liệu cho các bước chuẩn bị. `ket_qua.update(...)` thêm kết quả của bước mới
vào bảng kết quả đang có. Hàm `stepN_..._original()` tính và in báo cáo cho bước cần xem.

`sys.stdout = file_output` chuyển các lệnh print vào file.
`finally` khôi phục `sys.stdout` về màn hình kể cả khi bước tính gặp lỗi.
Mỗi script chỉ ghi `output_stepN.txt`; import script không tự chạy.
Đường dẫn tính từ thư mục mã nguồn, nên chạy từ thư mục khác vẫn tìm đúng data.txt.

Trong `run_all.py`, bốn đoạn gọi bước được viết tách riêng, theo thứ tự 1 đến 4.
Kết quả của bước trước được truyền trực tiếp cho bước sau. Chỉ chạy K-Means một lần.

Muốn đổi dữ liệu hoặc tham số, sửa các biến trong hàm `main()`:

```python
duong_dan_file_du_lieu = BASE_DIR / "data.txt"
thu_muc_luu_ket_qua = BASE_DIR
k = 2
r = 1.5
```

Các file bước cũng chạy trực tiếp được, ví dụ `python step2_normalize.py`.
Cách này chạy từ bước 1 đến bước cần xem và lưu các báo cáo tương ứng.
Các script dùng tham số khai báo trong mã, không nhận tham số dòng lệnh.

Input có hai cột x y, tiêu đề tùy chọn, phân cách bằng khoảng trắng/tab/phẩy/chấm phẩy.
Dùng dấu chấm cho phần thập phân. r cùng đơn vị với tọa độ Euclid.
Không tự đổi hệ tọa độ hoặc lấy mẫu.
Ví dụ mặc định là 20 điểm minh họa tự tạo trong `data.txt`, không lấy từ bài báo hoặc slide.
Các điểm có khoảng cách không đều, gồm vùng tập trung, vùng vừa và các điểm cách xa nhau.
Tham số chạy: k=2, r=1.5, K=3.
Với bộ này, 20 tổng khoảng cách kNN khác nhau và frNN có các giá trị từ 0 đến 9.
frNN là số đếm nguyên nên một số điểm vẫn có thể nhận cùng giá trị.

## Kết quả: mỗi bước một file

Bốn mẫu output HCA-BGP chỉ được tham khảo về cách trình bày.
Các công thức và lời gọi R vẫn theo QDC của tác giả.

Sau khi chạy đủ bốn bước, chỉ có bốn file `output_step*.txt`:

- `output_step1.txt`: tọa độ, thay số Euclid cho A1, chọn k điểm gần nhất, xét bán kính và bảng kNN/frNN.
- `output_step2.txt`: công thức Z-score, trung bình/độ lệch chuẩn, thay số và bảng chuẩn hóa.
- `output_step3.txt`: thay số khoảng cách tới tâm cuối cùng, liệt kê điểm trong từng cụm, tính tâm và bảng d(C1), d(C2), d(C3).
- `output_step4.txt`: thay số min/max để tính t1, t2, xét A1 và bảng kết quả Cao/Trung bình/Thấp.

Mỗi báo cáo chỉ trình bày bước tương ứng. Các trình chạy mặc định thay số chi tiết cho A1, rồi in bảng kết quả của tất cả các điểm.
Hàm run_pipeline() có tham số show_detailed và detail_limit nếu cần điều chỉnh phần giải thích;
bảng cuối vẫn chứa toàn bộ dữ liệu.
Bảng dùng tên điểm A1, A2, ..., tiêu đề tiếng Việt và số làm tròn tối đa 6 chữ số thập phân.
Việc làm tròn chỉ áp dụng khi in; các bước tiếp theo nhận số đầy đủ từ R.
Output hiện là báo cáo đọc trực tiếp, không còn là file TSV độc lập.

Ví dụ `python step2_normalize.py` cập nhật báo cáo bước 1 và 2;
`python run_step2_detailed.py` chỉ cập nhật `output_step2.txt`.
Các trình chạy chỉ xuất báo cáo TXT tương ứng với các bước.

Phần K-Means trình bày tâm cụm cuối cùng do R trả về cùng phép tính khoảng cách
để đối chiếu. Lời gọi R hiện tại không trả lịch sử từng vòng lặp.
Các phép tính thêm để giải thích không thay đổi kết quả chuyển giữa bốn bước.
Bước 1 tính lại các khoảng cách Euclid chỉ để trình bày phép thế tọa độ và tên điểm.
Khi khoảng cách bằng nhau, bảng minh họa chọn theo số thứ tự; tổng kNN không đổi.
Thời gian in lời giải chi tiết có thêm chi phí này, nên không dùng thời gian in báo cáo để đánh giá riêng thuật toán.
Trong các file `run_stepN_detailed.py`, đặt `so_diem_in_chi_tiet = None` để giải chi tiết mọi điểm.
Trong `run_all.py`, đổi `detail_limit=1` thành `detail_limit=None` nếu cần in chi tiết mọi điểm.

Cách trình bày tham khảo ảnh viết tay trong QDC.pptx (slide 8–15).
Cần phân biệt kết quả K-Means với nhãn mật độ cuối: bước 4 phải tính ngưỡng và dùng dấu `>` như mã gốc.
Số cụm luôn là K, nên số cụm không đổi không phải điều kiện hội tụ.
Trong minh họa gán điểm/cập nhật tâm, cần kiểm tra phân hoạch và tâm đã ổn định.
Output không dựng lại lịch sử lặp hoặc tâm khởi tạo mà lời gọi R không cung cấp.

Dữ liệu trao đổi số dùng 17 chữ số có nghĩa, không lấy các số đã làm tròn khi in
để tính bước tiếp theo.

## Bám theo mã công bố

Nguồn triển khai trực tiếp là **trang 11 của 01 - QDC.pdf** và mục 3.1 bài báo.
Trang mã tái lập cũng có ví dụ dùng hàm QDC() và phần kiểm tra độ nhạy
dùng data.Normalization(type="n1"); bộ mã này chọn lời gọi scale() trong slide.

- Dùng chính R dbscan để tính đặc trưng; giữ mặc định tìm kiếm của thư viện.
  KD-tree có thể hoạt động bên trong thư viện theo mặc định của nó.
- Gọi kNNdist và frNN riêng, không thêm phần dùng chung cây hoặc tái sử dụng kết quả.
- K-Means giữ Hartigan-Wong, nstart=1, iter.max=10 theo mặc định R.
- K-Means chọn tâm ban đầu ngẫu nhiên theo mặc định của R, như lời gọi trong slide.
- Giữ dấu >, kiểm tra Low trước High, còn lại Medium.
- Số cụm K cố định bằng 3 trong phiên bản ba mức mật độ này.
- Không tự xử lý đặc trưng hằng bằng cách thay đổi Z-score; báo lỗi rõ ràng.
- Không thêm M, R của ô lưới, clamp, phân chia đệ quy hoặc các phép tính HCA-BGP.
- Chưa thêm đồ thị; phần tham khảo mẫu tập trung vào cấu trúc hàm, comment và in chi tiết.

Đây là bản tổ chức lại quy trình công bố, không phải sao chép toàn bộ thư viện spatialWarsaw.
Phiên bản dbscan/R có thể khác môi trường tác giả năm 2025.
Mã số cụm không có thứ tự cao/trung bình/thấp cố định.
Các điểm trong data.txt là dữ liệu minh họa tự tạo, không phải số liệu thực nghiệm của tác giả.
Kết quả K-Means, mã cụm và nhãn mật độ có thể khác giữa các lần chạy.
Đọc output_step4.txt để xem kết quả thực tế của lần chạy đó.
Dùng run_all.py để tạo bốn báo cáo từ cùng một lần phân cụm.
Mỗi script run_step chạy độc lập; bước 4 sẽ tự tính lại các bước trước,
nên không được ghép kết quả K-Means của hai lần chạy riêng thành một ví dụ.
Mẫu 10 điểm cũ được giữ cố định trong phép kiểm tra ngưỡng ở `tests/test_qdc.py`.

## Cài đặt và kiểm tra

Python cần NumPy; R cần dbscan (và các phụ thuộc được trình cài R tự cài).

Cài thư viện Python:

```powershell
python -m pip install -r requirements.txt
```

Nếu máy chưa có thư viện dbscan, mở R hoặc RStudio và chạy một lần:

```r
install.packages("dbscan", repos = "https://cloud.r-project.org")
```

Thư viện đã cài được R sử dụng trực tiếp khi chạy QDC.
Không cần giữ một script cài đặt riêng trong thư mục thuật toán.

Kiểm tra chương trình:

```powershell
python -m unittest discover -s tests -v
```

Chương trình tự tìm Rscript trong PATH và các thư mục cài R thông thường.
Nếu cần chỉ rõ đường dẫn, truyền tham số rscript khi gọi hàm run_pipeline().
Cả bốn bước cần R, riêng bước 1 cần thêm dbscan.
Bộ kiểm tra đối chiếu kNN, frNN và Z-score với script R độc lập theo các phép tính công bố.
Tâm cụm được kiểm tra với trung bình của các điểm R đã gán.
Ngưỡng và nhãn được đối chiếu trên chính phân hoạch của lần chạy đang xét;
không buộc hai lần khởi tạo ngẫu nhiên độc lập phải cho cùng cách chia cụm.

## Nguồn

- Bài báo: https://doi.org/10.1016/j.compenvurbsys.2025.102289
- Slide tác giả, trang 11:
  https://kongres2025.stat.gov.pl/Content/Docs/Prezentacje/Sess28/01%20-%20QDC.pdf
- Mã tái lập:
  https://rstudio-pubs-static.s3.amazonaws.com/1296408_e84d8260d93247768073487b8c873b02.html

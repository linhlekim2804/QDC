# QDC theo mã R ở trang 11 của slide tác giả.
# Tên các trường kết quả dùng chung với r_bridge.py.


# Ghi bảng số
write_numeric <- function(values, filename, output_dir) {
  bang_so <- as.matrix(values)
  so_hang <- nrow(bang_so)
  cac_dong_can_ghi <- character(so_hang)

  for (chi_so_hang in seq_len(so_hang)) {
    hang_so <- bang_so[chi_so_hang, ]

    # Ghi 17 chữ số có nghĩa khi chuyển dữ liệu sang Python.
    cac_so_dang_chuoi <- sprintf("%.17g", hang_so)
    dong_csv <- paste(cac_so_dang_chuoi, collapse=",")
    cac_dong_can_ghi[chi_so_hang] <- dong_csv
  }

  duong_dan_file <- file.path(output_dir, filename)
  writeLines(cac_dong_can_ghi, duong_dan_file)
}


# Bước 1: Tính kNN và frNN
step1_spatial_features <- function(points, k, radius) {
  # all=TRUE trả đủ k khoảng cách cho mỗi điểm.
  khoang_cach_knn <- dbscan::kNNdist(points, k=k, all=TRUE)

  # S[i]: tổng khoảng cách đến k điểm gần nhất, không tính chính điểm i.
  tong_khoang_cach_knn <- apply(khoang_cach_knn, 1, sum)

  toa_do_cac_diem <- as.matrix(points)
  ket_qua_frnn <- dbscan::frNN(toa_do_cac_diem, eps=radius)

  # F[i]: số điểm khác trong bán kính r, đếm từ danh sách id.
  danh_sach_ma_diem <- ket_qua_frnn$id
  so_diem_trong_ban_kinh <- unlist(lapply(danh_sach_ma_diem, length))

  # Hai cột của bảng đặc trưng: S và F.
  bang_dac_trung <- cbind(tong_khoang_cach_knn, so_diem_trong_ban_kinh)

  ket_qua <- list()
  ket_qua$values <- bang_dac_trung
  ket_qua$distances <- khoang_cach_knn

  return(ket_qua)
}


# Bước 2: Chuẩn hóa Z-score
step2_normalization <- function(raw_features) {
  tong_khoang_cach_knn <- raw_features[, 1]
  so_diem_trong_ban_kinh <- raw_features[, 2]

  # z[i] = (giá trị - trung bình) / độ lệch chuẩn mẫu (mẫu số n - 1).
  knn_chuan_hoa <- scale(tong_khoang_cach_knn)
  frnn_chuan_hoa <- scale(so_diem_trong_ban_kinh)

  if (any(!is.finite(knn_chuan_hoa))) {
    stop("Cannot normalize a constant or non-finite kNN feature.")
  }

  if (any(!is.finite(frnn_chuan_hoa))) {
    stop("Cannot normalize a constant or non-finite frNN feature.")
  }

  trung_binh_knn <- attr(knn_chuan_hoa, "scaled:center")
  trung_binh_frnn <- attr(frnn_chuan_hoa, "scaled:center")
  do_lech_chuan_knn <- attr(knn_chuan_hoa, "scaled:scale")
  do_lech_chuan_frnn <- attr(frnn_chuan_hoa, "scaled:scale")

  cot_z_knn <- as.numeric(knn_chuan_hoa)
  cot_z_frnn <- as.numeric(frnn_chuan_hoa)
  bang_chuan_hoa <- cbind(cot_z_knn, cot_z_frnn)

  # Bảng tham số: hàng 1 là trung bình, hàng 2 là độ lệch chuẩn mẫu.
  hang_trung_binh <- c(trung_binh_knn, trung_binh_frnn)
  hang_do_lech_chuan <- c(do_lech_chuan_knn, do_lech_chuan_frnn)
  bang_tham_so <- rbind(hang_trung_binh, hang_do_lech_chuan)

  ket_qua <- list()
  ket_qua$values <- bang_chuan_hoa
  ket_qua$parameters <- bang_tham_so

  return(ket_qua)
}


# Bước 3: K-Means trên hai đặc trưng
step3_kmeans <- function(features) {
  # Chọn 3 cụm như ví dụ của tác giả.
  # Mặc định của R: Hartigan-Wong, nstart=1, iter.max=10.
  mo_hinh_kmeans <- stats::kmeans(features, centers=3)

  ma_loi <- mo_hinh_kmeans$ifault

  if (!is.null(ma_loi)) {
    if (ma_loi != 0) {
      stop("K-Means did not converge.")
    }
  }

  so_vong_lap <- mo_hinh_kmeans$iter

  if (so_vong_lap > 10) {
    stop("K-Means did not converge.")
  }

  ma_cum_cua_cac_diem <- mo_hinh_kmeans$cluster
  tam_cua_cac_cum <- mo_hinh_kmeans$centers
  tong_binh_phuong_khoang_cach <- mo_hinh_kmeans$tot.withinss

  thong_tin_kmeans <- c(so_vong_lap, tong_binh_phuong_khoang_cach)

  ket_qua <- list()
  ket_qua$clusters <- ma_cum_cua_cac_diem
  ket_qua$centers <- tam_cua_cac_cum
  ket_qua$diagnostics <- thong_tin_kmeans

  return(ket_qua)
}


# Bước 4: Tính ngưỡng và phân loại
step4_classification <- function(features, clusters) {
  cac_ma_cum <- unique(clusters)
  cac_ma_cum_can_co <- c(1, 2, 3)

  if (!setequal(cac_ma_cum, cac_ma_cum_can_co)) {
    stop("Expected three nonempty clusters.")
  }

  z_knn <- features[, 1]
  z_frnn <- features[, 2]

  min_knn_cum_1 <- min(z_knn[clusters == 1])
  min_knn_cum_2 <- min(z_knn[clusters == 2])
  min_knn_cum_3 <- min(z_knn[clusters == 3])

  min_frnn_cum_1 <- min(z_frnn[clusters == 1])
  min_frnn_cum_2 <- min(z_frnn[clusters == 2])
  min_frnn_cum_3 <- min(z_frnn[clusters == 3])

  # Mỗi ngưỡng là giá trị lớn nhất trong ba giá trị nhỏ nhất.
  t1 <- max(min_knn_cum_1, min_knn_cum_2, min_knn_cum_3)
  t2 <- max(min_frnn_cum_1, min_frnn_cum_2, min_frnn_cum_3)

  # Dùng dấu >, xét mật độ thấp trước, rồi đến mật độ cao.
  nhan_mat_do <- ifelse(
    z_knn > t1,
    "low-density",
    ifelse(
      z_frnn > t2,
      "high-density",
      "mid-density"
    )
  )

  cac_min_knn <- c(min_knn_cum_1, min_knn_cum_2, min_knn_cum_3)
  cac_min_frnn <- c(min_frnn_cum_1, min_frnn_cum_2, min_frnn_cum_3)
  bang_gia_tri_nho_nhat <- cbind(cac_ma_cum_can_co, cac_min_knn, cac_min_frnn)
  hai_nguong <- c(t1, t2)

  ket_qua <- list()
  ket_qua$thresholds <- hai_nguong
  ket_qua$minima <- bang_gia_tri_nho_nhat
  ket_qua$labels <- nhan_mat_do

  return(ket_qua)
}


# Nhận dữ liệu từ Python
tham_so_nhan_duoc <- commandArgs(trailingOnly=TRUE)

# Thứ tự tham số: bước, file đầu vào, thư mục tạm, k, r.
if (length(tham_so_nhan_duoc) != 5) {
  stop("Expected: step input.csv output_dir k radius")
}

so_buoc <- as.integer(tham_so_nhan_duoc[1])
file_du_lieu_vao <- tham_so_nhan_duoc[2]
thu_muc_ket_qua <- tham_so_nhan_duoc[3]
k <- as.integer(tham_so_nhan_duoc[4])
ban_kinh_r <- as.numeric(tham_so_nhan_duoc[5])

bang_du_lieu <- read.csv(file_du_lieu_vao, header=FALSE)
du_lieu <- as.matrix(bang_du_lieu)

if (any(!is.finite(du_lieu))) {
  stop("Input contains non-finite values.")
}


thoi_diem_bat_dau <- proc.time()[["elapsed"]]

if (so_buoc == 1) {
  da_co_dbscan <- requireNamespace("dbscan", quietly=TRUE)

  if (!da_co_dbscan) {
    stop("Install the R package dbscan before running step 1.")
  }

  ket_qua <- step1_spatial_features(du_lieu, k, ban_kinh_r)

} else if (so_buoc == 2) {
  ket_qua <- step2_normalization(du_lieu)

} else if (so_buoc == 3) {
  ket_qua <- step3_kmeans(du_lieu)

} else if (so_buoc == 4) {
  # Hai cột đầu là đặc trưng, cột thứ ba là mã cụm.
  hai_cot_dac_trung <- du_lieu[, 1:2, drop=FALSE]
  cot_ma_cum <- du_lieu[, 3]

  ket_qua <- step4_classification(hai_cot_dac_trung, cot_ma_cum)

} else {
  stop("Step must be between 1 and 4.")
}

thoi_diem_ket_thuc <- proc.time()[["elapsed"]]
thoi_gian_tinh <- thoi_diem_ket_thuc - thoi_diem_bat_dau


# Ghi kết quả để Python đọc lại
if (so_buoc == 1) {
  write_numeric(ket_qua$values, "values.csv", thu_muc_ket_qua)
  write_numeric(ket_qua$distances, "distances.csv", thu_muc_ket_qua)

} else if (so_buoc == 2) {
  write_numeric(ket_qua$values, "values.csv", thu_muc_ket_qua)
  write_numeric(ket_qua$parameters, "parameters.csv", thu_muc_ket_qua)

} else if (so_buoc == 3) {
  write_numeric(ket_qua$clusters, "clusters.csv", thu_muc_ket_qua)
  write_numeric(ket_qua$centers, "centers.csv", thu_muc_ket_qua)

  bang_thong_tin_kmeans <- matrix(ket_qua$diagnostics, nrow=1)
  write_numeric(bang_thong_tin_kmeans, "diagnostics.csv", thu_muc_ket_qua)

} else if (so_buoc == 4) {
  bang_nguong <- matrix(ket_qua$thresholds, nrow=1)
  write_numeric(bang_nguong, "thresholds.csv", thu_muc_ket_qua)
  write_numeric(ket_qua$minima, "minima.csv", thu_muc_ket_qua)

  file_nhan_mat_do <- file.path(thu_muc_ket_qua, "labels.txt")
  writeLines(ket_qua$labels, file_nhan_mat_do)
}


# Thông tin phiên R
phien_ban_r <- R.version.string
da_co_dbscan <- requireNamespace("dbscan", quietly=TRUE)

if (da_co_dbscan) {
  phien_ban_dbscan <- as.character(packageVersion("dbscan"))
} else {
  phien_ban_dbscan <- "not installed"
}

thoi_gian_dang_chuoi <- sprintf("%.17g", thoi_gian_tinh)
cac_dong_thong_tin <- c(phien_ban_r, phien_ban_dbscan, thoi_gian_dang_chuoi)
file_thong_tin <- file.path(thu_muc_ket_qua, "info.txt")

# info.txt nằm trong thư mục tạm và được xóa sau khi Python đọc xong.
writeLines(cac_dong_thong_tin, file_thong_tin)

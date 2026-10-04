# Đối chiếu đặc trưng và phân loại theo đoạn R của tác giả, trang 11 slide.
# Mã cụm là đầu vào từ lần chạy đang xét; không chạy lại K-Means để ép so sánh
# hai lần khởi tạo ngẫu nhiên độc lập.
args <- commandArgs(trailingOnly=TRUE)
popul <- read.csv(args[1])
k <- as.integer(args[3])
r <- as.numeric(args[4])
clusters <- scan(args[5], what=integer(), quiet=TRUE)
stopifnot(length(clusters) == nrow(popul), setequal(unique(clusters), 1:3))

knn.dist <- dbscan::kNNdist(popul[, 1:2], k, all=TRUE)
popul$knndist1 <- apply(knn.dist, 1, sum)
agg.radius <- dbscan::frNN(as.matrix(popul[, 1:2]), eps=r)
popul$frnn1 <- unlist(lapply(agg.radius$id, length))
popul$knndist1.scaled <- as.numeric(scale(popul$knndist1))
popul$frnn1.scaled <- as.numeric(scale(popul$frnn1))
popul$km.set1 <- clusters
t1 <- max(min(popul$knndist1.scaled[popul$km.set1 == 1]),
          min(popul$knndist1.scaled[popul$km.set1 == 2]),
          min(popul$knndist1.scaled[popul$km.set1 == 3]))
t2 <- max(min(popul$frnn1.scaled[popul$km.set1 == 1]),
          min(popul$frnn1.scaled[popul$km.set1 == 2]),
          min(popul$frnn1.scaled[popul$km.set1 == 3]))
popul$outcome.set1 <- ifelse(popul$knndist1.scaled > t1, "low-density",
                           ifelse(popul$frnn1.scaled > t2, "high-density", "mid-density"))
# write.table đủ để đối chiếu sai số số thực ở mức 1e-12.
write.csv(popul, args[2], row.names=FALSE)

# README – Nhận xét của thầy và kế hoạch sửa bài

**Bài báo:** Cải thiện phân đoạn cây ngô và cỏ dại trên ảnh UAV đa phổ 5 kênh
**Nguồn đối chiếu:** WeedsGalore (WACV 2025), PhenoBench (TPAMI 2024)

---

## 1. Kết luận chung của thầy

Bài có kết quả thực nghiệm đáng chú ý và có tiềm năng như một nghiên cứu cải tiến ứng dụng, **nhưng chưa nên nộp ở phiên bản hiện tại** vì 3 lý do:

1. Số liệu chưa thống nhất.
2. Thí nghiệm PhenoBench mô tả mâu thuẫn.
3. Tính mới chưa được đặt trong quan hệ với các nghiên cứu gần nhất.

> Lưu ý: thầy chỉ kiểm tra tính nhất quán và nguồn công bố. Thầy **chưa thể xác nhận kết quả chạy mô hình** vì chưa có mã nguồn, checkpoint và log.

---

## 2. Những phần ĐÃ ĐƯỢC

**Phép tính đúng:**
- Bảng 6: trung bình và độ lệch chuẩn của 5 seed khớp; mIoU từng seed khớp với trung bình 3 IoU.
- Bảng 3: các mức giảm 1.80 và 5.34 đúng (mốc 84.45).
- Bảng 2: tăng 1.73 điểm mIoU, giảm ~90.8% tham số, ~90.2% FLOPs, tốc độ tăng ~7.05 lần – đúng theo số đã ghi.
- Bảng 8: mIoU và các chênh lệch theo độ phân giải đúng về số học.
- ECE giảm từ 0.0056 xuống 0.0025 ≈ giảm 55.4%.

**Thông tin khớp với nguồn:**
- 82.90% mIoU của DeepLabV3+ trên WeedsGalore (3 lớp, đầu vào MSI, tập test).
- Số ảnh 104/26/26 và số lượng cây/cỏ từng split.
- Thống kê 8.55, 5.70 và 1 lớp Weed của PhenoBench trong Bảng 1.

**Điểm mạnh về thực nghiệm:** chạy 5 seed, có calibration (ECE), có thêm bộ dữ liệu bổ sung.

**Phần đóng góp có thể bảo vệ:** một decoder bổ sung chi tiết ở độ phân giải cao cho SegFormer-B0 đa phổ, đạt cân bằng độ chính xác – chi phí tốt trong thiết lập đang xét.

---

## 3. Những phần CHƯA ĐƯỢC

### 3.1. Lỗi số liệu (cần sửa chắc chắn)

| Vị trí | Vấn đề | Cần sửa |
|---|---|---|
| Bảng 2, 3, 4–6 | Mô hình đầy đủ có 3 kết quả: 84.63, 84.45 ± 0.16, 85.58 ± 0.27; Bảng 2 và 5 không ghi tập đánh giá | Ghi rõ là một lần chạy / trung bình validation / trung bình test. Không dùng thay thế nhau |
| Bảng 4, cấu hình RGB | (98.18 + 71.30 + 73.85)/3 = **81.11**, không phải 81.32 | Kiểm tra lại mIoU hoặc IoU thành phần; nếu dùng cách tổng hợp khác phải định nghĩa rõ |
| Đoạn sau Bảng 4 | "Test mIoU tăng 4.26 điểm" dựa trên 81.32 | Nếu RGB = 81.11 thì mức tăng ≈ **4.47** điểm |
| Đoạn sau Bảng 5 | Viết chi phí tăng "3.6 GFLOPs" | Theo bảng: 17.09 − 3.49 = **13.60 GFLOPs** (tăng ~390%, ~4.90 lần) |
| Mô tả nhánh refinement | Nói "không làm gia tăng đáng kể chi phí tính toán" | Sửa thành: tăng ít tham số nhưng tăng đáng kể tính toán so với SegFormer cơ bản |
| Bảng 1 / phần PhenoBench | "5,7 weed classes", "8,55 instances/image" | 5.70 là số weed instances/ảnh; 8.55 là số crop instances/ảnh; số lớp Weed = 1 |
| Bảng 1 – GSD WeedsGalore | Ghi 2.5 cm/pixel | Đúng là 2.5 mm/pixel = **0.25 cm/pixel** |
| Bảng 1 – năm WeedsGalore | Ghi 2024 | Bài chính thức ở WACV 2025; nếu giữ 2024 phải giải thích là mốc nào |

### 3.2. Thí nghiệm PhenoBench mâu thuẫn (ảnh hưởng trực tiếp đến kết luận)

Bản thảo đang viết cả hai cách:
- Phần dữ liệu: "huấn luyện và đánh giá theo pipeline riêng của PhenoBench".
- Phần kết quả: dùng PhenoBench để đánh giá mô hình "được training trên WeedsGalore", rồi kết luận mô hình duy trì hiệu năng khi chuyển dữ liệu.

Phải xác định đúng trường hợp thực tế:

| Quy trình thực tế | Kết luận được phép đưa ra |
|---|---|
| Huấn luyện riêng trên PhenoBench | Kiến trúc hoạt động trên một bộ dữ liệu bổ sung |
| Khởi tạo từ WeedsGalore rồi fine-tune trên PhenoBench | Khả năng chuyển giao sau fine-tuning |
| Giữ nguyên checkpoint WeedsGalore, không học từ PhenoBench | Khả năng tổng quát hóa trực tiếp qua bộ dữ liệu |

Bảng 9: baseline trên official test, Ours trên validation → chỉ mang tính tham khảo, **chưa đủ để kết luận vượt baseline**.

### 3.3. Trích dẫn

- Mục "Tài liệu tham khảo" đang **trống**.
- Nhiều chỗ nhắc WeedsGalore, SegFormer, MiT-B0, PhenoBench, Temperature Scaling nhưng không có trích dẫn hiển thị.
- Bảng 5 còn một "[3]" không có mục tương ứng.
- Cần ghi riêng nguồn của mIoU công bố và nguồn của Params/FLOPs/latency do mình đo lại.
- Cần phân biệt năm dữ liệu và năm tài liệu tham khảo (PhenoBench: dữ liệu 2023, bài TPAMI 2024, tập 46, số 12, trang 9583–9594).
- Latency 116.57 ms và 16.53 ms phải được đo lại trên **cùng T4, cùng kích thước ảnh, cùng quy trình** (WeedsGalore gốc huấn luyện trên A100, không ghép số đo từ môi trường khác nhau).

### 3.4. Tính mới còn yếu

| Thành phần | Đánh giá của thầy |
|---|---|
| MiT-B0 pretrained, multi-scale fusion | Kế thừa SegFormer, không phải đóng góp mới |
| Transformer nhẹ cho phân đoạn cỏ dại đa phổ, chuyển trọng số từ RGB | Đã có: Castellano, De Marinis, Vessio – Neurocomputing 2023 |
| Nhánh chi tiết không gian + nhánh ngữ nghĩa | Ý tưởng đã có trong BiSeNet / BiSeNet V2; phải giải thích điểm khác biệt |
| Mô hình nhẹ RGB–NIR–RE trên WeedsGalore | Đã có: *Lightweight Multispectral Crop-Weed Segmentation for Precision Agriculture* (2025) |
| 5 seed, calibration, bộ dữ liệu bổ sung | Điểm mạnh thực nghiệm, **không tự tạo tính mới phương pháp** |

Cần chứng minh mức cải thiện đến từ **thiết kế nhánh refinement**, chứ không chỉ từ pretrained backbone, loss, tăng độ phân giải decoder hay tăng tính toán.

### 3.5. Các điểm khác

- Thiếu thông tin tái lập: optimizer, weight decay, scheduler, augmentation, normalization từng band, seed thực tế, decoder dimensions, ignore labels, cách tính mIoU.
- Ablation chưa thống nhất thống kê: mô hình đầy đủ có Mean ± Std, các cấu hình loại bỏ chỉ có một số; RGB cũng cần cùng số seed với 5-band.
- ECE thiếu quy trình: số bin, cách chia bin, tính theo toàn bộ pixel hay theo lớp; nên bổ sung ECE riêng Crop/Weed (background chiếm ưu thế làm ECE tổng thấp). Bảng 7 cần chỉ rõ seed/checkpoint được calibration.
- Hình 4.1 sai chú thích (hình là so sánh phân đoạn, chú thích ghi so sánh đặc điểm hai dataset); chỉ có một ví dụ nên chưa đủ hỗ trợ nhận định về weed nhỏ.
- Hình kiến trúc: nhánh refinement có đường đi dư tới concatenation; các stage MiT chưa thể hiện rõ luồng tuần tự.
- Còn câu mang tính ghi chú biên tập: "Đây là một điều chỉnh của chúng tôi… cần được mô tả rõ trong phương pháp" → xóa.

---

## 4. Việc cần làm tiếp theo (checklist)

### Ưu tiên 1 – Sửa ngay (thầy nêu rõ là ưu tiên)

- [ ] Kiểm tra lại mIoU RGB ở Bảng 4 (81.32 hay 81.11) và sửa mức tăng 4.26 → 4.47 nếu cần.
- [ ] Sửa chênh lệch FLOPs: 3.6 → 13.60 GFLOPs (~4.90 lần); sửa câu "không tăng đáng kể chi phí tính toán".
- [ ] Làm rõ quy trình PhenoBench (thuộc trường hợp nào trong 3 trường hợp ở mục 3.2) và viết lại phần dữ liệu + kết quả cho thống nhất.
- [ ] Ghi rõ nguồn của từng baseline (số công bố hay tự chạy lại).
- [ ] Hoàn thiện danh mục tài liệu tham khảo và gắn trích dẫn trong toàn bài; xóa/sửa "[3]" ở Bảng 5.

### Ưu tiên 2 – Củng cố đối chứng cho nhánh refinement

- [ ] So sánh SegFormer-B0 chuẩn với mô hình đề xuất: cùng input, pretrained, loss, training, **5 seed**.
- [ ] Thêm đối chứng decoder độ phân giải 1/2 **không có** nhánh raw-input (tách lợi ích của nhánh khỏi lợi ích tăng độ phân giải).
- [ ] So sánh nhánh raw-input với việc dùng trực tiếp feature F1 của MiT.
- [ ] Báo cáo Boundary IoU/F1 hoặc hiệu năng trên weed nhỏ để hỗ trợ tuyên bố bảo toàn biên và đối tượng nhỏ.
- [ ] Chạy RGB và các cấu hình ablation cùng số seed với 5-band.

### Ưu tiên 3 – PhenoBench

- [ ] Chạy ERFNet, DeepLabV3+, SegFormer trên **cùng validation**, hoặc đánh giá mô hình đề xuất trên **official test**.
- [ ] Bổ sung: cách chuyển đầu vào 5-band sang RGB, xử lý trọng số hai kênh NIR–RE và nhánh refinement, dữ liệu dùng để huấn luyện/chọn checkpoint, normalization.

### Ưu tiên 4 – Related work và tính mới

- [ ] Thêm vào related work: Castellano et al. (Neurocomputing 2023), BiSeNet/BiSeNet V2, bài đa phổ nhẹ 2025.
- [ ] Viết rõ điểm khác biệt của thiết kế nhánh refinement so với các công trình trên.
- [ ] Không tuyên bố vượt số liệu trên split khác (bài 2025 dùng split 109/23/24, bản thảo dùng 104/26/26).
- [ ] Định vị đóng góp: "decoder bổ sung chi tiết ở độ phân giải cao cho SegFormer-B0 đa phổ, cân bằng độ chính xác – chi phí", không tuyên bố tính mới mạnh.

### Ưu tiên 5 – Tái lập, ECE, hình vẽ

- [ ] Bổ sung hyperparameter: optimizer, weight decay, scheduler, augmentation, normalization từng band, seed, decoder dimensions, ignore labels, cách tính mIoU.
- [ ] Đo lại latency trên cùng T4, cùng kích thước ảnh, cùng quy trình.
- [ ] ECE: nêu số bin, cách chia bin, theo pixel hay theo lớp; thêm ECE Crop/Weed; chỉ rõ seed/checkpoint ở Bảng 7.
- [ ] Sửa chú thích Hình 4.1; thêm nhiều ví dụ định tính (đặc biệt weed nhỏ).
- [ ] Sửa hình kiến trúc: bỏ đường dư tới concatenation, thể hiện rõ luồng tuần tự các stage MiT.
- [ ] Sửa Bảng 1: GSD = 0.25 cm/pixel (2.5 mm), năm WeedsGalore, "5.70 weed instances / 8.55 crop instances / 1 lớp Weed".
- [ ] Xóa câu ghi chú biên tập còn sót.
- [ ] Chuẩn bị mã nguồn, checkpoint và log để có thể xác nhận kết quả.

---

## 5. Thứ tự gợi ý

1. Sửa các lỗi số liệu và chú thích (nhanh, không cần chạy thêm).
2. Hoàn thiện tài liệu tham khảo và related work.
3. Làm rõ quy trình PhenoBench.
4. Chạy thêm thí nghiệm đối chứng (ablation, 5 seed, Boundary IoU, PhenoBench cùng tập đánh giá).
5. Cập nhật bảng/hình, bổ sung phần tái lập, rồi gửi lại thầy kiểm tra trước khi nộp.

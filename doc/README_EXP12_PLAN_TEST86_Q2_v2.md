# Exp12 v2 — Kế hoạch hướng tới test mIoU > 86 % và bài Scopus Q2 (bản nháp để duyệt)

> Phạm vi: một mô hình SegFormer (MiT) duy nhất cho mỗi cấu hình, giữ nguyên mọi thành phần hiện có (không bỏ, không ghép mô hình). Tài nguyên: Kaggle T4, khoảng 60 giờ GPU, backbone lớn hơn B0 được phép, deviation test đã được duyệt (bằng chứng duyệt phải đính kèm manifest). Mọi con số GPU-giờ là **ước lượng**, đo lại ở E12.0.
>
> **Thay đổi so với v1:** (A) tách nhánh Performance và nhánh Scientific; (B) độ phân giải ×1,5 chỉ là nội suy, không thêm thông tin; (C) đổi tên giao thức thành *Fixed-Schedule Training Protocol* (FSP); (D) khóa công thức loss và cờ huấn luyện từ source; (E) bổ sung ba điều kiện và các chỉ số cho claim cỏ nhỏ.

## 1. Hai nhánh tách rời

| Nhánh | Câu hỏi | Mô hình | Chỉ số chính | Tiêu chí |
|---|---|---|---|---|
| **PERF** (performance) | Một SegFormer đạt test mIoU bao nhiêu? | Backbone (B0/B1/B2) × FSP × tỉ lệ ảnh × TTA, head = LLR gốc; chỉ nhận thêm head mới nếu head qua cổng mIoU | mIoU, IoU từng lớp | Trung bình 5 seed test mIoU **> 86,0 %** (đăng ký trước) |
| **SCI** (đóng góp khoa học) | Sub-cell occupancy decoding có cải thiện cỏ nhỏ thật không? | LLR gốc, OccCond-g, MCSD-g, trên cùng backbone và cùng FSP, ở B0 **và** ở backbone đã chọn | Small-Weed Recall, kèm cổng bảo vệ ở mục 4 | Ba điều kiện ở mục 4 |

Các mô hình là đối chứng độc lập, không phải ensemble. Bài báo phải phân biệt rõ **mô hình có mIoU cao nhất** (PERF) và **mô hình chứa đóng góp mới** (SCI). Nếu PERF-final là LLR gốc trên B2 + FSP + TTA, bài **không** được viết rằng sub-cell decoding gây ra mức mIoU đó.

## 2. Khóa công thức loss và giao thức (kiểm tra từ source)

Trong `train.py`: `loss = (1 − dice_weight) × CE + dice_weight × SoftDice`. Với `--dice_weight=0.3` công thức thực chạy là **0,7·CE + 0,3·Dice**, CE có trọng số lớp (0,9 / 1,0 / 1,3). Đó chính là công thức Exp10, nên README không cần đổi công thức. Nhưng:
- **Mặc định trong `train.py` khác giao thức đã dùng:** `weight_bg=0.35`, `weight_weed=1.5`, `dice_weight=0.5`. Mọi lệnh Exp12 phải truyền tường minh bộ cờ COMMON của Exp10 (0,9 / 1,0 / 1,3, dice 0,3, lr 3e-4, batch 8, 100 epoch, augmentation cũ). Nếu bỏ sót cờ, loss đổi mà không ai để ý.
- Thêm kiểm tra tự động: `run_config.json` của mọi run Exp12 phải khớp một file `exp12_frozen_config.json` (băm SHA-256 ghi vào manifest); khác là dừng.
- Mọi thay đổi loss là **một yếu tố riêng** và không được trộn vào FSP.

**Fixed-Schedule Training Protocol (FSP):** chỉ khác Exp10 ở ba điểm: warmup + lr giảm cosine, EMA trọng số, báo checkpoint EMA cuối (không chọn epoch theo val). FSP bỏ độ lệch chọn epoch tốt nhất, nhưng **không** bảo đảm tốt hơn lịch cũ và không làm validation độc lập hơn. Kết quả có thể thấp hơn con số best-val cũ.

## 3. Các giai đoạn

Nền chung: MiT 5 băng + multi-scale fusion + LLR raw, loss ở mục 2.

| Giai đoạn | Nội dung | Seed |
|---|---|---|
| E12.0 | Profile tốc độ, VRAM, kiểm tra nạp B1/B2, kiểm tra nội suy nhãn, test đơn vị | — |
| E12.1a | Đánh giá `last.pth` (epoch cuối) của E11-B0 hiện có (không train): đo trực tiếp độ lệch chọn best-val | 0–4 |
| E12.1 | B0-LLR dưới FSP so với lịch cũ (cặp, 5 seed) | 0–4 |
| E12.2 | Backbone B0 / B1 / B2 (LLR gốc, FSP) | 0–4 |
| E12.3 | Nhánh SCI tại **B0**: LLR gốc, OccCond-g, MCSD-g (FSP) | 0–4 |
| E12.3b | Nhánh SCI tại **backbone thắng E12.2**: cùng ba nhánh | 0–4 |
| E12.4 | Tỉ lệ ảnh ×1,5 (có điều kiện, xem mục 3.1) | 0–4 |
| E12.5 | TTA lật/xoay (cùng một mạng, chỉ suy luận) | — |
| E12.6 | **Xác nhận**: PERF-final, SCI-final, REF (LLR gốc cùng backbone) | 5–9 |
| E12.7 | **Test, một lần**: cả ba cấu hình đã đăng ký, checkpoint EMA cuối | 5–9 |

Quy tắc chọn (đăng ký trước): thêm thành phần vào PERF chỉ khi chênh lệch mIoU trung bình 5 seed so với giai đoạn trước có CI95 (bootstrap theo ảnh và seed, `compare_runs.py`) dưới > 0; nếu không giữ cấu hình đơn giản hơn. Tiêu chí > 86,0 % chỉ gắn với PERF-final. SCI-final và REF được báo với chỉ số riêng và không có ngưỡng mIoU. Nếu PERF-final trùng REF thì dùng chung một bộ run.

### 3.1 Về độ phân giải ×1,5

Ảnh gốc là 600×600. Nội suy lên 900×900 **không tạo thông tin không gian mới** và không được diễn giải là khôi phục chi tiết. Nó chỉ đổi cách backbone và decoder xử lý (mật độ token, trường tiếp nhận, nhánh LLR ở H/2 sẽ là 450). Chỉ chạy E12.4 nếu E12.0 cho thấy đủ VRAM và thời gian ở B2, hoặc B1, và nhãn nội suy bằng nearest không làm hỏng biên. Báo cáo kết quả như một yếu tố thực nghiệm, không như một cải thiện đặc trưng phương pháp.

## 4. Claim cỏ nhỏ phải đạt đồng thời ba điều kiện (nhánh SCI)

1. **Không đánh đổi:** Small-Weed Recall tăng với CI95 dưới > 0, trong khi Crop IoU không giảm quá 1,0 pp và mIoU trung bình không giảm quá 0,3 pp so với LLR gốc cùng backbone.
2. **Hiệu quả còn khi đổi backbone:** hiệu ứng cùng dấu và CI95 dưới > 0 ở ít nhất hai backbone (B0 và backbone chọn ở E12.2); báo hiệu ứng riêng từng backbone và kiểm tra tương tác.
3. **Không chỉ do dự đoán Weed rộng hơn:** PrecisionWeed không kém hơn quá 1,0 pp; báo tỉ lệ khối lượng Weed dự đoán trên GT; và so với **đường recall–precision của LLR gốc khi dịch bias logit Weed** (lưới đăng ký trước, chỉ dùng như đối chứng chẩn đoán trên validation): đầu mới phải nằm phía trên đường này (recall cao hơn ở precision tương đương).

**Bảng đánh giá bắt buộc** cho mọi nhánh: mIoU, IoU từng lớp (bg, Crop, Weed), BIoU Crop/Weed, Small/Medium/Large recall@0,5 và coverage, PrecisionWeed, số tham số, FLOPs, độ trễ (T4, 600×600, cùng quy ước profile).

## 5. Quy tắc bảo vệ tính hợp lệ

- Validation không còn độc lập: mọi lựa chọn Exp01–Exp11 và các cổng của Exp12 dùng cùng 26 ảnh. Seed 5–9 chỉ kiểm tra độ ổn định qua khởi tạo và huấn luyện, **không** tạo tập dữ liệu xác nhận độc lập. Phải nêu rõ trong bài.
- Cấm: chọn checkpoint/seed theo val hay test, chỉnh siêu tham số sau khi thấy kết quả, chạy test lần hai, dùng ảnh test (kể cả không nhãn) để thiết kế hay huấn luyện.
- Nếu PERF-final dưới 86 % thì báo nguyên con số kèm CI95 theo ảnh; không chạy lại.
- Số bài cũ (85,49 ± 0,30) chỉ gắn cho mô hình Word cũ và giao thức lịch sử.

## 6. Ngân sách T4 (ước lượng)

Giả định: B0 ≈ 20 phút/run (đã đo), B1 ≈ 35, B2 ≈ 55 phút (suy đoán), ×2,25 ở ×1,5.

| Giai đoạn | Giờ |
|---|---|
| E12.0 | 1,0 |
| E12.1a + E12.1 | 2,0 |
| E12.2 (B1, B2) | 7,5 |
| E12.3 (B0: OccCond-g, MCSD-g) | 3,3 |
| E12.3b (backbone B2: OccCond-g, MCSD-g) | 9,2 |
| E12.4 (có điều kiện) | 10,3 |
| E12.5 TTA + đánh giá | 2,0 |
| E12.6 (PERF-final ≤ 10,3; SCI-final 4,6; REF 4,6) | ≤ 19,5 |
| E12.7 | 0,5 |
| **Tổng tối đa** | **≈ 55** (dự trữ ≈ 5) |
| Nếu bỏ E12.4 (và PERF-final = REF) | ≈ 34–39 |

Biện pháp cắt khi vượt ngân sách: bỏ E12.4 trước, rồi giảm E12.3b xuống 3 seed (ghi rõ), không bỏ E12.6. Rủi ro: B2 ở ×1,5 có thể hết VRAM (nếu dùng batch 4 + tích lũy gradient thì đó là thay đổi giao thức, phải ghi lại); quota Kaggle theo tuần và giới hạn 12 giờ/phiên.

## 7. Cần sửa mã

1. `segformer.py`: lấy số kênh decoder từ `config.hidden_sizes` (hiện cứng 32/64/160/256), thêm `--backbone={b0,b1,b2}`; checkpoint B0 cũ phải load được.
2. `train.py`: warmup + cosine, EMA, lưu/đánh giá checkpoint EMA cuối, kiểm tra khớp `exp12_frozen_config.json`.
3. `evaluate_ccmc.py`: TTA lật/xoay, cờ `--backbone`, cờ dịch bias logit Weed (chỉ cho đối chứng ở mục 4), xuất PrecisionWeed và tỉ lệ khối lượng Weed.
4. Dataset: tăng tỉ lệ ảnh cho ảnh và nhãn (nhãn nearest).
5. Test đơn vị: nạp B1/B2, tương thích checkpoint B0, EMA, TTA, đường dẫn bias logit, kiểm tra cờ đóng băng.

## 8. Hồ sơ trước khi mở test (E12.7)

Bản đề nghị deviation kèm bằng chứng duyệt; manifest bất biến mới (commit, hash split, hash checkpoint và `run_config.json` của seed 5–9 của ba cấu hình, hash mã đánh giá, hash `exp12_frozen_config.json`); giữ nguyên marker và hồ sơ abort của K8.3; xác nhận chưa có mô hình Exp12 nào từng được đánh giá trên test; cell test chỉ chạy một lần và tạo marker trước khi đánh giá.

## 9. Tính mới (làm song song, không tốn GPU)

Mỗi thành phần của đầu sub-cell có tiền lệ: dự đoán pixel-shuffle gần như là Dense Upsampling Convolution, và ràng buộc tỉ lệ lớp bằng Sinkhorn đã có trong bài toán sinh ảnh theo tỉ lệ lớp (Semantic Palette, CVPR 2021). Chưa kết luận được về tổ hợp. Cần tìm kiếm có hệ thống, ghi truy vấn và ngày: sub-pixel/sub-cell prediction, DUC, Sinkhorn/optimal transport trong phân đoạn có giám sát, giám sát theo tỉ lệ lớp theo ô, phân đoạn cỏ dại UAV đa phổ với SegFormer/MiT, và các bài trích dẫn WeedsGalore. Chỉ viết "theo hiểu biết của chúng tôi, chưa có công bố…" sau bước này; không viết "novel operator" hay "first".

## 10. Quy tắc dừng

- Nếu E12.2 và E12.4 đều không qua cổng, hoặc sau E12.6 trung bình validation (seed 5–9) thấp hơn trung bình E12.2/E12.3b (seed 0–4) quá 0,3 pp, dừng và báo cáo; không mở test để "kiểm tra".
- Nếu dùng quá 50 giờ trước E12.6, bỏ E12.4 và các nhánh phụ.

# REVIEW_REPORT — CCMC-SR (cập nhật sau mỗi thí nghiệm)

> File này ghi lại các lần review kết quả Kaggle so với kế hoạch [README_CCMC_SR_v2_3](../paper/README_CCMC_SR_v2_3_oracle_leakage_reviewed.md). Mỗi lần có kết quả mới sẽ thêm một mục và cập nhật bảng trạng thái.
> Quy ước: `DONE` = đã chạy và có artifact; `PARTIAL` = chưa đủ seed/CI; `NOT RUN` = chưa chạy (kèm lý do). Gate: `PASS / FAIL / INCONCLUSIVE / SCREEN-PASS / NO-GO / PENDING`.

## 1. Bảng trạng thái tổng

| Gate | Câu hỏi | Bằng chứng cần có | Trạng thái | Cập nhật |
|---|---|---|---|---|
| G1 | SR có phải nút thắt của cỏ nhỏ? | Diag-D0 + Diag-D2 (Exp01) | PENDING | — |
| G2 | Oracle có bác bỏ động lực Weed-specific? (sàng lọc một chiều) | C3 so với C2, C3 so với C3-K5 (Exp02) | PENDING | — |
| G3 | H2: learned Weed prior hơn learned Veg prior? | L3 − L2 ≥ +3 pp Small-Weed Recall, CI > 0, Crop không giảm quá 1 pp (Exp03) | PENDING | — |
| G4 | Lợi ích có đến từ việc nối prior vào SR? | L3 > K1, K2, K3, K5, K6 (Exp04–05) | PENDING | — |

| Exp | Trạng thái | Ghi chú |
|---|---|---|
| 00 Code + setup | DONE (review tĩnh) | Chưa chạy test; S6 trên Kaggle là bước kiểm tra bắt buộc |
| 01 Baseline + chẩn đoán | Sẵn sàng chạy | [01_baseline_diagnostics.md](01_baseline_diagnostics.md) |

---

## 2. Exp00 — Review triển khai code (02/11/2026)

### 2.1 Những gì đã cài đặt, đối chiếu với kế hoạch

| Yêu cầu trong kế hoạch | Cài đặt | Vị trí |
|---|---|---|
| CCMC-SR: `w = eps+(1-eps)p`, `w_norm = w/mean_window(w)`, `X_mod = X(1+λ(w_norm−1))`, rìa ngoài cửa sổ SR có hệ số 1, tính FP32 | `window_normalized_factor` | [src/nets/ccmc_sr.py](../src/nets/ccmc_sr.py) |
| Chỉ sửa đầu vào SR conv dùng cho K/V; Q và residual giữ nguyên; stage 2–3; mọi block | Forward pre-hook trên `attention.self.sr` (đã xác minh source HF 4.45.2: Q được tính trước SR) | `CCMCMiTEncoder` |
| Prior từ **output stage 1** (sau norm, dạng 4D), cấm điều biến stage 1 | Vòng lặp encoder viết tường minh, giống hệt HF; dùng stage 1 thì báo lỗi | `CCMCMiTEncoder.forward` |
| Prior head khởi tạo đồng đều, `λ = sigmoid(a)`, init 0,05 | Conv cuối khởi tạo bằng 0 nên `p = 0,5` và `w_norm = 1`; mỗi stage có một `λ` | `PriorHead`, `lam_logit` |
| Aux loss soft occupancy 150×150, BCE, 0,3·L_aux, không detach F1 | `aux_target` + `train.py` | [src/train.py](../src/train.py) |
| L0, L1, L2, L3, K1, K2, K2b, K3, K5, K6, C2, C3, C3-K5, C3-K6 | `--prior_mode`, `--injection`, `--prior_perturb` | xem [doc/README.md](README.md) |
| Oracle intervention trên **cùng** checkpoint (2×2), GT-Weed + nhiễu | `--prior_override`, `--prior_noise` (dilate, erode, shift, drop) | [src/evaluate_ccmc.py](../src/evaluate_ccmc.py) |
| Small-Weed Recall theo bin tam phân vị train, ghép one-to-one, IoU 0,5 (và 0,3) | Dùng instance chính thức nếu có, nếu không thì connected components (ghi rõ `source`) | [src/ccmc_metrics.py](../src/ccmc_metrics.py), [make_instance_bins.py](../src/scripts/make_instance_bins.py) |
| Boundary IoU, theo ngày bay, ma trận nhầm lẫn, chất lượng prior | Có trong `*_summary.json` | `evaluate_ccmc.py` |
| Paired bootstrap theo cluster (ảnh hoặc tile), paired seed, gate tự động | `compare_runs.py --gate=h2/g4/screen/abs` | [compare_runs.py](../src/scripts/compare_runs.py) |
| Diag-D2 linear probe (avg trước SR, sau SR, max trước SR), AUROC/AP/recall@precision, contrast small − large có CI | `probe_sr.py` | [probe_sr.py](../src/scripts/probe_sr.py) |
| FLOPs có tính matmul attention, latency đồng bộ CUDA, trung vị/p10/p90, peak VRAM | `profile_model.py` (FlopCounterMode) | [profile_model.py](../src/scripts/profile_model.py) |
| Test chỉ chạy một lần | `--split=test` bị từ chối nếu thiếu `--confirm_test_locked=True` | `evaluate_ccmc.py` |
| Tái lập/resume phiên Kaggle | `run_config.json` (git commit, split hash, versions), `history.jsonl`, `last.pth` + `--resume` | `train.py` |

### 2.2 Quyết định thiết kế đăng ký trước (cần bạn xác nhận; nếu đổi thì phải đổi **trước** khi xem kết quả validation)

1. **K5 (gate trước SR, không chuẩn hóa cửa sổ):** `factor = 1 + λ(w/mean_ảnh(w) − 1)`. Chuẩn hóa theo toàn ảnh giúp K5 vẫn trùng baseline khi prior đồng đều và có cùng biên độ với CCMC-SR. Như vậy khác biệt duy nhất so với L3 là chuẩn hóa **cục bộ theo cửa sổ SR**. Gate thô `X·w` (không chuẩn hóa) không trùng baseline lúc khởi tạo và chỉ có thể làm yếu feature, nên không dùng.
2. **K6:** prior được avg-pool về lưới K/V và chuẩn hóa theo toàn lưới, rồi gate nhân vào token K/V **sau** LayerNorm của SR, trước phép chiếu `key`/`value`. *Sửa so với kế hoạch ("trước LayerNorm"):* HF áp `layer_norm` theo từng token ngay sau `sr`, mà LayerNorm bất biến với phép nhân vô hướng trên token. Nếu đặt gate trước LayerNorm thì gate bị triệt tiêu, K6 gần như trùng L0 và trở thành đối chứng giả (L3 sẽ "thắng" K6 một cách vô nghĩa).
3. **Một `λ` cho mỗi stage**, dùng chung cho 2 block của stage đó. Nhật ký train ghi `λ` theo từng epoch.
4. **Resize prior:** 150→75 (stage 2) khớp chính xác 2×2; 150→38 (stage 3) dùng `adaptive_avg_pool2d` nên chỉ căn chỉnh xấp xỉ. Chưa đo sensitivity khi lệch 1 pixel (để ở ablation).
5. **Prior head:** Conv3×3(32→32)–BN–ReLU–Conv1×1(32→1), khoảng 9,3 nghìn tham số; tạo **sau** decoder để decoder của L3 được khởi tạo giống L0 với cùng seed. Thứ tự batch và augmentation dùng generator riêng, nên L0/L2/L3/K* có cùng seed sẽ thấy cùng chuỗi batch (paired thật).
6. **Small-Weed Recall chính** = recall@IoU 0,5 của bin nhỏ. Recall@0,3 và coverage (≥ 50 % pixel của instance được dự đoán là Weed) là thước đo phụ. Coverage giúp phân biệt "bỏ sót" với "dính vào cỏ bên cạnh".
7. **Cluster bootstrap:** mặc định theo ảnh; báo cáo thêm theo tile (`2023-06-06_0956` và `2023-06-15_0956` cùng tile) vì cách này thận trọng hơn. Gate dùng kết quả **tile** nếu số tile ≥ 10.
8. **K2 khi đánh giá:** phép hoán vị dùng seed cố định 0 để kết quả tái lập được.
9. **C2/C3:** checkpoint được chọn theo val mIoU trong khi vẫn dùng prior GT của val, nên có hai lớp leakage. Đây chỉ là sàng lọc, mọi output đều mang cờ `GT_LEAKAGE`.

**Rủi ro toán học cần biết trước (ảnh hưởng claim novelty):** vì LayerNorm sau SR bất biến với hệ số vô hướng trên từng token K/V, thành phần "tổng khối lượng của cửa sổ" gần như bị LayerNorm loại bỏ (chỉ còn khác biệt nhỏ qua bias của SR conv và qua phép trộn `1 + λ(·)`). Hệ quả: **K5 (gate trước SR, chuẩn hóa toàn ảnh) có thể cho kết quả rất gần L3**, vì trong mỗi cửa sổ hai toán tử có tỉ lệ trọng số tương đối gần như nhau. Dự đoán đăng ký trước: nếu L3 ≈ K5 thì đây là hệ quả toán học, **không** được claim đóng góp riêng của "window normalization". Khi đó novelty phải dựa vào việc *tái trọng số theo prior bên trong receptive field của SR*, chứng minh bằng L3 so với K6 (gate sau giảm độ phân giải), K1, K2, K3 và L3 so với L2.

### 2.3 Kiểm toán sơ bộ từ dữ liệu có sẵn trong repo (chưa cần chạy)

| Vấn đề | Phát hiện | Hệ quả |
|---|---|---|
| 82,16 so với 83,53 (không LLR) | [paper/ablation_results.csv](../paper/ablation_results.csv): `no_refine` gồm 5 seed **trên test** = 82,94 / 83,98 / 83,42 / 83,50 / 83,83, trung bình **83,53**, khớp con số trong kế hoạch. README gốc ghi "SegFormer + pretrained MiT, **CE**" đạt **82,07 % val**; còn CE + Dice (λ = 0,5, trọng số 0,8/1,0/1,5) đạt 82,62 % val. Vì vậy 82,16 nhiều khả năng đến từ cấu hình **chỉ dùng CE** và/hoặc khác split | Không gộp hai số. Exp01 chạy lại L0 (CE + Dice 0,3, trọng số 0,9/1,0/1,3, có fusion, không LLR) trong cùng pipeline, chỉ báo cáo **val**. Cần đối chiếu log gốc của Bảng 5 để xác nhận |
| FLOPs 3,49 G và 17,09 G | `measure_model_stats` trong `train.py` đo bằng **THOP tại 512×512** (mặc định `input_size=(512,512)`), không phải 600×600, và THOP bỏ sót matmul của attention. Latency cũ đo bằng `time.time()` **không** có `cuda.synchronize` | Hai số này không dùng được làm bằng chứng hiệu quả. Exp01 đo lại ở 600×600 bằng FlopCounterMode, có đồng bộ CUDA |
| `set_detect_anomaly(True)` luôn bật trong `train.py` cũ | Đã chuyển thành cờ `--detect_anomaly` (mặc định tắt). Cờ này chỉ thêm kiểm tra, không đổi giá trị số, nhưng làm train chậm đáng kể | Các lần chạy cũ và mới có thể so sánh về số liệu; thời gian train thì không |

### 2.4 Kiểm thử

Theo yêu cầu, **không chạy local**. Toàn bộ unit test (toán tử + tích hợp HF: encoder loop trùng HF, trùng baseline lúc khởi tạo, Q/stage 1 bất biến, K/V stage 2 thay đổi cho cả `ccmc_sr`/`pre_sr_gate`/`post_sr_gate`, backward không NaN, 512/600, batch 2) chạy ở Cell S6 trên Kaggle với `transformers==4.45.2`. **Không chạy Exp01 nếu S6 có test fail.**

Review tĩnh trước khi chạy (02/11/2026): phát hiện và sửa lỗi K6 (gate trước LayerNorm bị triệt tiêu, xem 2.2 mục 2). Các script khác đã đối chiếu cờ dòng lệnh trong cell Kaggle với cờ được định nghĩa trong code.

### 2.5 Giới hạn và rủi ro còn mở

- Định dạng file `instances/*.png` của WeedsGalore chưa được kiểm tra trực tiếp. Cell E1.1 in ra `mode/dtype`; nếu `source = cc` thì phải ghi "không tương đương instance GT" trong bài.
- Diag-D1 (quét `R`) và proxy "tiếp giáp tán cây" chưa cài đặt (kế hoạch cho phép NOT RUN).
- Dataset thứ hai (N7) chưa có pipeline.
- Unit test chưa đo FLOPs của prior head/modulation trên Kaggle; Cell E1.8 sẽ đo.

---

## 3. Exp01 — Baseline L0 + Diag-D0 + Diag-D2 (chờ kết quả)

Checklist sẽ dùng khi review:

- [ ] Unit test trên Kaggle: tất cả pass, không skip.
- [ ] `instance_bins.json`: nguồn instance, ngưỡng bin, `n_small/n_medium/n_large` trên val (≥ 20?).
- [ ] L0 s0/s1: val mIoU, Crop, Weed, chênh lệch giữa seed, `best_epoch`, thời gian/epoch (để ước lượng `t`).
- [ ] Kiểm toán: L0 val trong pipeline mới so với các dòng 82,07 / 82,62 (val) của README cũ.
- [ ] Diag-D0: recall@0,5 / @0,3 / coverage theo bin; `large − small` kèm CI; theo ngày; ma trận nhầm lẫn (Weed→Crop hay Weed→Background).
- [ ] LiteSent so với L0: LLR tăng chủ yếu ở bin nào.
- [ ] Diag-D2: AUROC pre_avg / post_sr / pre_max theo stage; contrast small − large kèm CI; kết quả có nhất quán giữa 2 seed không.
- [ ] Profile: params/GFLOPs/GMACs/latency của L0, LiteSent, L3, K5, K6 ở 600×600.
- [ ] Quyết định G1 và đề xuất nội dung Exp02.

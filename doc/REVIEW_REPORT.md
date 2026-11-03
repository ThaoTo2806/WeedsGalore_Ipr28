# REVIEW_REPORT — CCMC-SR (cập nhật sau mỗi thí nghiệm)

> File này ghi lại các lần review kết quả Kaggle so với kế hoạch [README_CCMC_SR_v2_3](../paper/README_CCMC_SR_v2_3_oracle_leakage_reviewed.md). Mỗi lần có kết quả mới sẽ thêm một mục và cập nhật bảng trạng thái.
> Quy ước: `DONE` = đã chạy và có artifact; `PARTIAL` = chưa đủ seed/CI; `NOT RUN` = chưa chạy (kèm lý do). Gate: `PASS / FAIL / INCONCLUSIVE / SCREEN-PASS / NO-GO / PENDING`.

## 1. Bảng trạng thái tổng

| Gate | Câu hỏi | Bằng chứng cần có | Trạng thái | Cập nhật |
|---|---|---|---|---|
| G1 | SR có phải nút thắt của cỏ nhỏ? | Diag-D0 + Diag-D2 (Exp01) | **PASS có điều kiện** (stage 2; hiệu ứng nhỏ, tương quan, không phải nhân quả) | 02/11/2026, Exp01 |
| G2 | Oracle có bác bỏ động lực Weed-specific? (sàng lọc một chiều) | C3 so với C2, C3 so với C3-K5 | **INCONCLUSIVE** (C3a − C2a = +2,96 [−1,26; +8,31]). **Quy tắc dừng CCMC-SR đã kích hoạt:** cơ chế hoạt động nhưng C3a/C3f không vượt L0 dù có rò rỉ nhãn | 03/11/2026, Exp03 |
| G3, G4 | — | — | **NOT RUN** (CCMC-SR dừng theo quy tắc 4.7) | 03/11/2026 |
| G3 | H2: learned Weed prior hơn learned Veg prior? | L3 − L2 ≥ +3 pp Small-Weed Recall, CI > 0, Crop không giảm quá 1 pp (Exp03) | PENDING | — |
| G4 | Lợi ích có đến từ việc nối prior vào SR? | L3 > K1, K2, K3, K5, K6 (Exp04–05) | PENDING | — |

| Exp | Trạng thái | Ghi chú |
|---|---|---|
| 00 Code + setup | DONE | 44/44 unit test pass trên Kaggle (`transformers 4.45.2`, torch 2.11, T4, commit `ea354cd`) |
| 01 Baseline + chẩn đoán | DONE | L0 s0/s1, LiteSent s0, D0, D2 (2 seed), profile. Diag-D1 NOT RUN (đúng kế hoạch). Xem mục 3 |
| 02 Oracle C2/C3 | DONE nhưng **vô hiệu về khoa học** | λ không học được; xem mục 4 |
| 03 Oracle với λ hoạt động, 3 seed | DONE | 13/13 run; xem mục 5 |
| 04 Định vị nút thắt (decoder 1/2, nguồn chi tiết) | Sẵn sàng chạy | [04_bottleneck_localization.md](04_bottleneck_localization.md) |

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
7. **Cluster bootstrap:** ~~theo tile~~ → **theo ảnh** (sửa sau Exp01, xem 3.6 mục 4: mã tile không nhất quán giữa các ngày, 25 tile cho 26 ảnh).
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

## 3. Exp01 — Baseline L0 + Diag-D0 + Diag-D2 (review 02/11/2026)

Artifact: [doc/exp01_results/results/exp01](exp01_results/results/exp01). Môi trường: T4, torch 2.11.0+cu128, transformers 4.45.2, commit `ea354cd`, split hash `675667edf3f35f16`, 44/44 unit test pass.

### 3.1 Trạng thái từng mục

| Cell | Nội dung | Trạng thái |
|---|---|---|
| E1.1 | Bin kích thước | DONE: instance **chính thức** (PNG mode `L`, uint8), 156/156 ảnh; edges = **[45, 147] px**; val có 620/567/621 instance nhỏ/vừa/lớn, nên mọi bin đủ n ≥ 20 |
| E1.2 | L0 seed 0, 1 | DONE |
| E1.3 | LiteSent seed 0 | DONE |
| E1.4–E1.6 | Diag-D0, LiteSent so với L0 | DONE |
| E1.7 | Diag-D2 probe (2 seed) | DONE |
| E1.8 | Profile | DONE |
| Diag-D1 | Quét `R` | NOT RUN (thăm dò, kế hoạch cho phép) |

### 3.2 Baseline và kiểm toán

| Mô hình (val) | mIoU | Crop | Weed | best epoch | s/epoch |
|---|---|---|---|---|---|
| L0 s0 | 82,94 | 75,11 | 75,54 | 93 | 9,9 |
| L0 s1 | 82,79 | 74,74 | 75,46 | 94 | 10,0 |
| LiteSent s0 | 84,56 | 77,44 | 77,81 | 92 | 14,3 |

- **Tái lập được:** chênh lệch giữa 2 seed của L0 là 0,15 mIoU. LiteSent đạt 84,56, khớp con số 84,63 val trong README cũ.
- **Kiểm toán 82,16 so với 83,53:** L0 với CE + Dice đạt khoảng 82,9 val. Các dòng cũ là 82,07 (CE) và 82,62 (CE + Dice với λ = 0,5). Vậy 82,16 thuộc cấu hình khác loss (nhiều khả năng chỉ dùng CE). Tham chiếu chuẩn từ nay là **L0 trong pipeline này**. Số 83,53 là test của cấu hình tương tự L0, không dùng để so với val.
- **Lưu ý:** best epoch rơi vào 92–94/100, tức mô hình có thể chưa bão hòa. Protocol 100 epoch đã khóa và áp dụng như nhau cho mọi mô hình nên vẫn công bằng. Đây là giới hạn cần ghi trong bài, không đổi giữa chừng.
- Train mất khoảng **17 phút/run** (`t ≈ 0,3 GPU-giờ`). Năm seed của 3 mô hình chỉ tốn khoảng 5 GPU-giờ, nên quota không phải nút thắt.

### 3.3 Diag-D0: cỏ nhỏ có bị bỏ sót nhiều hơn không? → **CÓ, rõ ràng**

| L0 s0 (val) | small (≤ 45 px) | medium | large (> 147 px) |
|---|---|---|---|
| Coverage (≥ 50 % pixel dự đoán là Weed) | **63,5** | 92,4 | 98,6 |
| Recall@IoU 0,5 | 49,7 | 69,8 | 47,8 |
| Recall@IoU 0,3 | 66,9 | 77,6 | 57,8 |

- `cov_large − cov_small` (bootstrap theo ảnh): L0 s0 **+35,0 [23,6; 44,2]**; L0 s1 **+40,2 [28,5; 49,3]**.
- **Hiệu ứng vẫn còn khi tách theo ngày**, tức không chỉ do ngày đầu có nhiều cỏ nhỏ: 2023-05-30 +17,2 [5,3; 30,0]; 2023-06-06 +28,9 [9,1; 39,2] (s0). s1 cho kết quả tương tự.
- **Phát hiện về metric:** recall@IoU 0,5 của bin *large* thấp (khoảng 48 %) dù coverage đạt 98,6 %. Lý do là cỏ lớn chạm nhau bị dự đoán thành một khối dính liền, nên ghép one-to-one theo IoU thất bại; ngày 2023-06-15 chỉ có recall50_large = 10 % trong khi coverage là 97 %. Vì vậy phép so sánh `recall50 large − small` ở Cell E1.5 (≈ 0, CI rất rộng) **không đo được việc bỏ sót** và không dùng cho G1. Với bin small, recall50 vẫn hợp lệ: cỏ nhỏ ít dính nhau, và recall50_small (49,7) cùng chiều với cov_small (63,5).
- Ma trận nhầm lẫn: lỗi của Weed chủ yếu là **Weed → Background (6,8 %)**, lớn hơn Weed → Crop (2,4 %). Đây là kiểu lỗi "mất cỏ" mà CCMC-SR nhắm tới, không phải lỗi phân biệt Crop/Weed.
- Theo ngày: 2023-05-25 có Weed IoU chỉ 37,5 % và chứa 283/620 cỏ nhỏ của val. Đây là nhóm có nhiều dư địa cải thiện nhất (dự đoán T3 trong kế hoạch).

**LiteSent so với L0 (seed 0, paired bootstrap):** LLR tăng Small-Weed Recall **+14,5 [+11,4; +17,7]**, cov_small +8,4 [+4,6; +12,4], nhưng recall50_large chỉ +2,1 và mIoU +1,6. Lợi ích của chi tiết độ phân giải cao **tập trung vào cỏ nhỏ**. Điều này ủng hộ giả thuyết "mất chi tiết không gian", nhưng LLR tác động ở decoder nên **không** chứng minh riêng SR là nguyên nhân.

### 3.4 Diag-D2: linear decodability trước và sau SR

| Stage (R) | AUROC small: pre_avg → post_sr | pre_max | Contrast (small − large), s0 | s1 |
|---|---|---|---|---|
| 1 (8) | 0,605 → 0,566 | 0,694 | −0,019 [−0,049; +0,008] | −0,051 [−0,094; −0,018] |
| **2 (4)** | **0,845 → 0,816** | **0,904** | **+0,028 [+0,014; +0,042]** | **+0,025 [+0,007; +0,040]** |
| 3 (2) | 0,941 → 0,944 | 0,945 | −0,001 [−0,006; +0,003] | −0,004 [−0,007; −0,001] |

Diễn giải:
- **Stage 2:** cỏ nhỏ mất khả năng giải mã qua SR (AUROC giảm khoảng 0,028) trong khi cỏ lớn gần như không mất (0,001). Contrast dương với CI > 0 **ở cả 2 seed**. Đây là bằng chứng chính cho H1.
- **Stage 1:** cỏ nhỏ gần mức ngẫu nhiên ngay cả *trước* SR (AUROC 0,61), tức hiệu ứng sàn. Contrast âm vì cỏ lớn có nhiều thứ để mất hơn, nên stage 1 không diễn giải được cho H1. Dù sao, stage 1 cũng không nằm trong pilot vì prior lấy từ output stage 1.
- **Stage 3:** SR 2×2 không làm mất thông tin, nên **điều biến stage 3 có thể không cần**.
- **Phát hiện quan trọng cho thiết kế:** `pre_max` luôn cao hơn `pre_avg` và `post_sr` đối với cỏ nhỏ ở stage 1–2 (stage 2: 0,904 so với 0,816). Tín hiệu cỏ nhỏ **có mặt** trong cửa sổ trước SR, nhưng phép gộp tuyến tính gần với trung bình đã làm loãng nó. Đây đúng là cơ chế CCMC-SR nhắm tới: tái trọng số bên trong cửa sổ để tín hiệu cỏ không bị pha loãng.
- Độ lớn hiệu ứng nhỏ (ΔAUROC ≈ 0,03) và đây là bằng chứng tương quan, không phải nhân quả.

### 3.5 Chi phí tính toán (600×600, fp32, batch 1, T4, FlopCounter có tính matmul attention)

| Mô hình | Params | GFLOPs (GMACs) | Latency trung vị (p10–p90) | Peak VRAM |
|---|---|---|---|---|
| L0 | 3,456 M | 13,27 (6,63) | 14,85 ms (14,59–14,89) | 145 MB |
| L3 / K5 / K6 | 3,465 M (+9,3 k) | 13,68 (6,84), **+3,1 %** | 15,35 / 15,50 / 15,53 ms, **+3,4 %** | 151 MB |
| LiteSent | 3,662 M | 50,41 (25,21), **+280 %** | 24,68 ms, +66 % | 299 MB |

- Các số cũ 3,49 G và 17,09 G thực chất là **MACs của THOP ở 512×512**, chỉ tính conv. Không dùng lại hai số này trong bài.
- Ý nghĩa cho paper: LLR mua +14,5 pp Small-Weed Recall với giá +280 % FLOPs. L3 chỉ tốn +3 %. Nếu L3 lấy lại được một phần đáng kể khoảng cách này với CI rõ thì đó là kết quả quality/compute có giá trị.
- Số `modulation_elementwise_Mops` của K6 đang ước lượng theo kích thước đầu vào SR nên chưa chính xác; con số nhỏ nên không ảnh hưởng kết luận.

### 3.6 Quyết định G1: **PASS có điều kiện**

Theo quy tắc đăng ký trước, D0 cho thấy nhóm nhỏ suy giảm rõ (coverage, CI > 0, kể cả trong từng ngày) **và** D2 cho contrast dương với CI > 0 ở ít nhất một stage (stage 2, cả 2 seed). Các điều kiện kèm theo:
1. Bằng chứng mang tính tương quan; hiệu ứng ở stage 2 nhỏ. Trong bài chỉ được viết "consistent with", không viết "SR causes".
2. Stage 3 không có dấu hiệu mất thông tin. Cấu hình pilot vẫn giữ stage 2–3 như đã khóa, nhưng **ablation "chỉ stage 2" được nâng lên ưu tiên cao** (Exp04/05).
3. **Sửa đổi metric trước khi có bất kỳ kết quả nào của phương pháp:**
   - **(a)** Gate vẫn dùng `SmallWeedRecall` (recall@0,5, bin nhỏ), giữ nguyên như đã đăng ký. `cov_small` được nâng thành *đồng chính*: chỉ kết luận PASS khi hai chỉ số **cùng dấu**.
   - **(b)** Không dùng recall@IoU của bin large để đo bỏ sót.
   - **(c)** Boundary IoU đổi sang độ rộng cố định **3 px** (`--boundary_px=3`). Tỉ lệ 0,02 cho khoảng 17 px, rộng hơn hầu hết cỏ, nên BIoU ≈ IoU như thấy ở Exp01.

   Lý do sửa đổi là tính hợp lệ của metric trên L0, không dựa trên kết quả của L3.
4. Mục 2.2.7 (cluster theo tile) được thay thế: val có 25 tile cho 26 ảnh, và mã tile không nhất quán giữa các ngày bay. Vì vậy cluster theo tile ≈ cluster theo ảnh. Gate dùng **cluster theo ảnh**. Trong bài phải ghi giới hạn: không có metadata vị trí để bootstrap theo ô ruộng.

### 3.7 Code đã sửa sau review Exp01

- `evaluate_ccmc.py`: thêm `--boundary_px` (mặc định 3). Summary ghi lại `boundary_px`.
- `make_instance_bins.py`: sửa cách đếm `n_unique` (chỉ ảnh hưởng thông tin in ra).
- Từ Exp02, L0/LiteSent được **đánh giá lại** với BIoU 3 px, để mọi so sánh dùng cùng định nghĩa.

### 3.8 Việc tiếp theo

**Exp02 = C2, C3, C3-K5, C3-K6 (2 seed)** cùng gate G2-screen: [02_oracle_c2_c3.md](02_oracle_c2_c3.md). Theo kế hoạch, chưa train L3 khi chưa có G2.

Việc bạn cần làm trước Exp02:
1. Commit + push code đã sửa.
2. Upload `exp01_ckpts.zip` thành Kaggle Dataset.
3. Commit `results/instance_bins.json` (bin đã khóa: [45, 147]).

---

## 4. Exp02 — Oracle C2/C3, C3-K5, C3-K6 (review 02/11/2026)

Artifact: [doc/exp02_results/results/exp02](exp02_results/results/exp02).

### 4.1 Trạng thái

| Mục | Trạng thái |
|---|---|
| 8 run train (C2, C3, C3K5, C3K6 × seed 0, 1) | DONE |
| Đánh giá lại L0 s0/s1, LiteSent (BIoU 3 px) | DONE |
| E2.4 nhiễu prior trên C3 s0 | DONE |
| C3 so với C2, C3K5, C3K6 | DONE |
| C3 so với L0 và C2 so với L0 | **Lỗi** `Missing columns pq_*` (L0 không có cột chất lượng prior), đã sửa trong `compare_runs.py` |

### 4.2 Kết quả (val, trung bình 2 seed)

| Mô hình | SmallWeedRecall | cov_small | mIoU | Weed IoU | λ cuối (stage 2; 3) |
|---|---|---|---|---|---|
| L0 | 48,23 (49,68 / 46,77) | 60,89 | 82,87 | 75,50 | — |
| C3 (GT-Weed) | 48,15 | 62,66 | 82,82 | 75,44 | 0,051; 0,053 |
| C2 (GT-Veg) | 49,84 | 64,68 | 82,83 | 75,49 | 0,050; 0,052 |
| C3-K5 | 44,68 | 57,02 | 82,66 | 75,32 | 0,052; 0,054 |
| C3-K6 | 47,74 | 62,26 | 82,56 | 75,08 | 0,052; 0,054 |
| LiteSent (tham chiếu, 1 seed) | 64,19 | 71,94 | 84,56 | 77,81 | — |

### 4.3 Chẩn đoán: **cơ chế chưa từng hoạt động**, Exp02 không kiểm định được giả thuyết

1. **λ bị kẹt về mặt toán học.** Adam dịch mỗi tham số khoảng ≤ lr mỗi bước. Với lr 3e-4 và 100 epoch × 13 iter = 1 300 bước, logit `a` đổi được nhiều nhất khoảng 0,39, nên λ ≤ sigmoid(−2,94 + 0,39) ≈ 0,07. Quỹ đạo thực tế là 0,050 → 0,054. Khởi tạo λ = 0,05 trong kế hoạch là để đảm bảo trùng baseline, nhưng phần đó đã được prior head đồng đều đảm bảo, nên λ nhỏ là không cần thiết và làm tắt cơ chế. *Đây là lỗi thiết kế tối ưu hóa, không phải bằng chứng về giả thuyết.*
2. **Bằng chứng trực tiếp:** C3 s0 với prior GT bị dilate, erode, drop 30 % hay shift 4 px cho kết quả **trùng** prior sạch (mIoU 82,791–82,795; Small-Weed Recall 47,742 ở cả 5 trường hợp). Mô hình không nhạy với prior, kể cả prior là nhãn thật.
3. Vì vậy mọi khác biệt giữa C2, C3, C3-K5, C3-K6 và L0 là **nhiễu huấn luyện**. Prior ở mức λ ≈ 0,05 chỉ làm quỹ đạo huấn luyện lệch đi như một seed khác.

### 4.4 Bài học thống kê quan trọng (đưa vào phần phương pháp của paper)

- Cùng mô hình L0, hai seed chênh **2,9 pp** Small-Weed Recall (49,68 so với 46,77) và 5,3 pp cov_small. Mức này ngang ngưỡng +3 pp của H2.
- Bootstrap chỉ theo ảnh (2 seed, cố định) cho CI **quá hẹp**. C3 − C2 = −1,69 [−2,79; −0,36] trông "có ý nghĩa". C3 − C3-K5 = +3,47 [+1,90; +5,30] được gate ghi "PASS", nhưng phần lớn do một seed lệch (C3K5_s1 = 42,58). Trong khi đó các mô hình này về thực chất là cùng một mô hình.
- **Đã sửa trong `compare_runs.py`:**
  - **(a)** Bootstrap phân tầng: resample seed paired, rồi resample ảnh (`--seed_bootstrap`, mặc định bật).
  - **(b)** PASS/SCREEN-PASS cần ít nhất 3 seed paired (`--min_seeds_pass=3`).
  - **(c)** Chỉ dùng các cột chung giữa hai mô hình.
- Exp03 có **A/A test** (L0_s0 so với L0_s1, chỉ bootstrap theo ảnh) để định lượng việc bootstrap theo ảnh đánh giá thấp phương sai. Kết quả này đưa vào phụ lục thống kê.

### 4.5 Quyết định

- **G2 = INCONCLUSIVE (cơ chế chưa kích hoạt).** Không ghi NO-GO, và không dùng bất kỳ so sánh nào của Exp02 làm bằng chứng về loại prior hay vị trí can thiệp.
- Kết quả "PASS" của C3 so với C3-K5 bị **hủy**, vì sinh ra từ nhiễu seed với cơ chế không hoạt động.
- Chưa sang L2/L3 cho tới khi Exp03 trả lời được câu hỏi: *khi λ thực sự học được, prior GT có tác động lên đầu ra không?*

### 4.6 Code đã sửa sau Exp02

- `train.py`: thêm `--lam_lr` (param group riêng cho logit λ). Exp03 dùng `--lam_init=0.5 --lam_lr=0.01`; khởi tạo vẫn trùng baseline với L2/L3 vì prior head đồng đều. Cấu hình λ cố định dùng `--learn_lambda=False --lam_init=0.999`.
- `compare_runs.py`: bootstrap phân tầng theo seed, yêu cầu ≥ 3 seed, chỉ dùng cột chung.

### 4.7 Quy tắc dừng đăng ký trước cho Exp03 (chống "đào dữ liệu")

| Kết quả Exp03 | Quyết định |
|---|---|
| Activity: \|C3a(GT) − C3a(uniform)\| < 1 pp Small-Weed Recall ở cả 3 seed | Toán tử **không có đòn bẩy**, kể cả với nhãn thật. Dừng CCMC-SR dạng hiện tại |
| Activity rõ, nhưng C3a **và** C3f đều không vượt L0 (CI chứa 0 hoặc âm) | Toán tử thay đổi đầu ra nhưng không có lợi. Dừng CCMC-SR; không chuyển sang L3 |
| C3a hoặc C3f > L0, nhưng C3a ≤ C2a | G2 NO-GO cho tính *Weed-specific*. Có thể chuyển hướng sang prior thực vật (L2 làm phương pháp chính) và phải ghi rõ trong bài |
| C3a > L0 và C3a > C2a (SCREEN-PASS, 3 seed) | Sang Exp04: L2, L3 với cùng cấu hình λ (init 0,5, lam_lr 0,01) |

### 4.8 Về yêu cầu "cải tiến thật, chưa có paper công bố"

- **CCMC-SR không bỏ bớt gì của mô hình pretrained.** Toàn bộ trọng số MiT-B0 được giữ nguyên; chỉ **đầu vào của phép giảm K/V** được tái trọng số theo prior lớp. Đây là sửa đổi toán tử bên trong backbone, không phải cắt bớt hay ghép mô hình.
- **Không thể bảo đảm tuyệt đối "chưa có paper nào công bố".** Điều làm được là rà soát có hệ thống: tìm kiếm có ghi lại từ khóa, cơ sở dữ liệu, ngày, và đối chiếu công thức với các công trình gần nhất ở mục 4 của kế hoạch (MLSRS-SegFormer APW, Dynamic Spatial Sparsification, ALGM, VLTP, SegMAN). Việc này nằm ở P7, nhưng **chỉ có ý nghĩa nếu Exp03 cho thấy toán tử có tác dụng**. Một phương pháp mới mà không có hiệu quả thì cũng không đăng Q1 được.
- Nếu Exp03 rơi vào hai dòng đầu của bảng 4.7, phương án dự phòng **vẫn nằm trong kế hoạch** (mục 4.3): "dual-statistic SR". Phương án này dùng một nhánh gộp có trọng số theo prior và một nhánh SR chuẩn, chung conv và pha trộn bằng hệ số học được. Lý do được dẫn từ Diag-D2: `pre_max` giữ tín hiệu cỏ nhỏ tốt hơn hẳn `post_sr`. Phương án này chỉ được mở khi có lý do ghi rõ trong báo cáo, không mở tự động.

---

## 5. Exp03 — Oracle với λ hoạt động, 3 seed (review 03/11/2026)

Artifact: [doc/exp03_results/results/exp03](exp03_results/results/exp03). 13/13 run DONE. So sánh dùng bootstrap phân tầng (seed paired + ảnh).

### 5.1 λ đã học được (sửa lỗi Exp02 thành công)

| Mô hình | λ cuối (stage 2; 3), theo seed |
|---|---|
| C3a (GT-Weed, CCMC-SR) | 0,85; 0,82 · 0,80; 0,86 · 0,87; 0,80 |
| C2a (GT-Veg, CCMC-SR) | 0,50; 0,62 · 0,52; 0,59 · 0,61; 0,54 |
| C3K5a (GT-Weed, gate trước SR) | 0,83; 0,89 · 0,78; 0,93 · 0,86; 0,91 |
| C3f | 0,999 (cố định) |

Khi prior là GT-Weed, λ tăng mạnh hơn GT-Veg. Mô hình **muốn** dùng prior Weed nhiều hơn prior thực vật, đây là một quan sát mô tả.

### 5.2 Kết quả chính (Small-Weed Recall, val)

| Mô hình | s0 / s1 / s2 | Trung bình | So với L0 [CI 95 %] |
|---|---|---|---|
| **L0** | 49,68 / 46,77 / 52,90 | **49,78** (SD 3,1) | — |
| C3a | 45,97 / 48,87 / 50,48 | 48,44 | −1,34 [−4,74; +2,30] |
| C3f (λ = 1) | 46,94 / 46,29 / 49,52 | 47,58 | −2,20 [−5,14; +0,35] |
| C2a | 46,61 / 47,58 / 42,26 | 45,48 | −4,30 [−10,32; +0,92] |
| C3K5a | 49,52 / 40,97 / 50,81 | 47,10 | (C3a − C3K5a = +1,34 [−3,78; +7,54]) |
| LiteSent (1 seed, tham chiếu) | 64,19 | — | +14,5 (Exp01) |

mIoU của mọi mô hình nằm trong khoảng 82,4–82,9, không khác L0.

### 5.3 Cơ chế **có** hoạt động: đổi prior trên cùng checkpoint

| Checkpoint | GT-Weed so với prior đồng đều (Small-Weed Recall) | cov_small |
|---|---|---|
| C3a | **+4,14 [+1,84; +6,69]** | +5,48 [+2,92; +8,27] |
| C3f | **+9,78 [+5,27; +15,67]** | +14,09 [+9,21; +19,53] |

Diễn giải: mô hình **dựa vào** prior. Bỏ prior đi thì tụt, ví dụ C3f với prior đồng đều chỉ còn 37,8 so với L0 49,8. Tuy vậy, dùng prior cũng **không vượt** L0. Toán tử chỉ khiến mạng *chuyển phụ thuộc* sang prior, không bổ sung năng lực. Ngay cả nhãn thật đưa qua K/V SR của stage 2–3 cũng không tăng recall cỏ nhỏ.

### 5.4 A/A test: bằng chứng phương pháp thống kê

L0_s0 so với L0_s1 (cùng mô hình, chỉ khác seed), chỉ bootstrap theo ảnh: Small-Weed Recall **+2,90 [+0,21; +5,41]**, cov_small **+5,32 [+2,98; +7,32]**, đều "có ý nghĩa". Hai mô hình giống hệt nhau nhưng bootstrap theo ảnh vẫn báo khác biệt, tức đây là **dương tính giả**. Kết hợp với độ lệch chuẩn giữa seed của L0 (3,1 pp), kết quả này chứng minh:
- **(a)** Các so sánh 1 seed trong nhiều bài trên WeedsGalore có thể không đáng tin.
- **(b)** Bootstrap phân tầng theo seed + ảnh với ≥ 3 seed là bắt buộc.

Đây là đóng góp phương pháp đánh giá **dùng được cho paper**.

### 5.5 Quyết định theo quy tắc đăng ký trước (mục 4.7, dòng 2)

> "Activity rõ, nhưng C3a **và** C3f đều không vượt L0 → toán tử thay đổi đầu ra nhưng không có lợi. **Dừng CCMC-SR; không chuyển sang L3.**"

- **G2:** C3a − C2a = +2,96 [−1,26; +8,31], INCONCLUSIVE. Nhưng câu hỏi này không còn quyết định, vì ngay cả trần oracle cũng không vượt L0.
- **G3/G4: NOT RUN.** Không chạy L2/L3/K-controls; không claim CCMC-SR.
- **Giải thích cơ chế (giả thuyết làm việc):** Diag-D2 cho thấy SR ở stage 2 làm mất một phần khả năng giải mã cỏ nhỏ (ΔAUROC ≈ 0,03). Hiệu ứng có thật nhưng **nhỏ và không phải nút thắt chính**. Thông tin dùng để phân đoạn cỏ nhỏ chủ yếu đi qua F1 (stride 4) và decoder, chứ không qua tóm tắt K/V của stage 2–3. LLR giúp +14,5 pp chính vì nó tác động vào độ phân giải đầu ra và chi tiết mức thấp.

### 5.6 Tiến độ: đã được nửa đường chưa?

Trả lời thẳng: **với gói bằng chứng chẩn đoán và đánh giá thì khoảng 40 %; với phương pháp mới cho Q1 thì chưa bắt đầu lại.** Tức là chưa tới nửa đường.

| Phần của paper | Trạng thái | Dùng được không |
|---|---|---|
| Baseline tái lập, kiểm toán số liệu cũ, FLOPs/latency đo đúng | DONE | Có |
| Chẩn đoán cỏ nhỏ (D0: coverage theo bin và theo ngày; lỗi Weed → Background) | DONE | Có |
| Probe SR (D2): mất thông tin ở stage 2, max-pool giữ tốt hơn | DONE | Có (phân tích) |
| Oracle CCMC-SR: âm tính có kiểm soát (cơ chế hoạt động nhưng không lợi) | DONE | Có (kết quả âm tính/ablation, **không** phải đóng góp chính) |
| Phương pháp thống kê: A/A test, bootstrap phân tầng | DONE | Có |
| **Phương pháp mới vượt baseline có ý nghĩa** | **Chưa có** | Đây là phần quyết định Q1 |
| Rà soát novelty, 5 seed, test một lần, dataset thứ hai | Chưa | — |

### 5.7 Việc tiếp theo: Exp04, định vị nút thắt trước khi thiết kế phương pháp mới

[04_bottleneck_localization.md](04_bottleneck_localization.md): so LiteSent (decoder 1/2 + băng thô) với **D½-none** (decoder 1/2, không có chi tiết) và **D½-F1** (decoder 1/2, chi tiết lấy từ F1), mỗi cấu hình 3 seed. Thí nghiệm này cũng trả lời trực tiếp hai đối chứng thầy yêu cầu. Bảng diễn giải đã khóa trước trong file Exp04. Phương pháp mới chỉ được thiết kế *sau* khi biết nút thắt nằm ở:
- (i) độ phân giải decoder,
- (ii) patch embedding stage 1 của MiT, hay
- (iii) cách decoder dùng F1.

Nếu là (ii), hướng ưu tiên là **sửa toán tử patch embedding stage 1 bên trong MiT**: giữ trọng số pretrained, không ghép mô hình. Việc này phải rà soát novelty có ghi lại trước khi code.

Code đã thêm: `--llr_source {raw, none, f1}` cho `SegFormer5Band`. Mặc định là `raw` nên checkpoint LiteSent cũ vẫn tải được.

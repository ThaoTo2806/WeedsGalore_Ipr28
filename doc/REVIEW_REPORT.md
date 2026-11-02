# REVIEW_REPORT — CCMC-SR (cập nhật sau mỗi thí nghiệm)

> File này ghi lại các lần review kết quả Kaggle so với kế hoạch [README_CCMC_SR_v2_3](../paper/README_CCMC_SR_v2_3_oracle_leakage_reviewed.md). Mỗi lần có kết quả mới sẽ thêm một mục và cập nhật bảng trạng thái.
> Quy ước: `DONE` = đã chạy và có artifact; `PARTIAL` = chưa đủ seed/CI; `NOT RUN` = chưa chạy (kèm lý do). Gate: `PASS / FAIL / INCONCLUSIVE / SCREEN-PASS / NO-GO / PENDING`.

## 1. Bảng trạng thái tổng

| Gate | Câu hỏi | Bằng chứng cần có | Trạng thái | Cập nhật |
|---|---|---|---|---|
| G1 | SR có phải nút thắt của cỏ nhỏ? | Diag-D0 + Diag-D2 (Exp01) | **PASS có điều kiện** (stage 2; hiệu ứng nhỏ, tương quan, không phải nhân quả) | 02/11/2026, Exp01 |
| G2 | Oracle có bác bỏ động lực Weed-specific? (sàng lọc một chiều) | C3 so với C2, C3 so với C3-K5 (Exp02) | PENDING | — |
| G3 | H2: learned Weed prior hơn learned Veg prior? | L3 − L2 ≥ +3 pp Small-Weed Recall, CI > 0, Crop không giảm quá 1 pp (Exp03) | PENDING | — |
| G4 | Lợi ích có đến từ việc nối prior vào SR? | L3 > K1, K2, K3, K5, K6 (Exp04–05) | PENDING | — |

| Exp | Trạng thái | Ghi chú |
|---|---|---|
| 00 Code + setup | DONE | 44/44 unit test pass trên Kaggle (`transformers 4.45.2`, torch 2.11, T4, commit `ea354cd`) |
| 01 Baseline + chẩn đoán | DONE | L0 s0/s1, LiteSent s0, D0, D2 (2 seed), profile. Diag-D1 NOT RUN (đúng kế hoạch). Xem mục 3 |
| 02 Oracle C2/C3 | Sẵn sàng chạy | [02_oracle_c2_c3.md](02_oracle_c2_c3.md) |

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

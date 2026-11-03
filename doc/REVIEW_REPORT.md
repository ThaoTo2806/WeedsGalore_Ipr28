# REVIEW_REPORT — CCMC-SR (cập nhật sau mỗi thí nghiệm)

> File này ghi lại các lần review kết quả Kaggle so với kế hoạch [README_CCMC_SR_v2_3](../paper/README_CCMC_SR_v2_3_oracle_leakage_reviewed.md). Mỗi lần có kết quả mới sẽ thêm một mục và cập nhật bảng trạng thái.
> Quy ước: `DONE` = đã chạy và có artifact; `PARTIAL` = chưa đủ seed/CI; `NOT RUN` = chưa chạy (kèm lý do). Gate: `PASS / FAIL / INCONCLUSIVE / SCREEN-PASS / NO-GO / PENDING`.

## 1. Bảng trạng thái tổng

| Gate   | Câu hỏi                                                       | Bằng chứng cần có                                                           | Trạng thái                                                                                                                                                | Cập nhật          |
| ------ | ------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| G1     | SR có phải nút thắt của cỏ nhỏ?                               | Diag-D0 + Diag-D2 (Exp01)                                                   | **PASS có điều kiện** (stage 2; hiệu ứng nhỏ, tương quan, không phải nhân quả)                                                                            | 02/11/2026, Exp01 |
| G2     | Oracle có bác bỏ động lực Weed-specific? (sàng lọc một chiều) | C3 so với C2, C3 so với C3-K5                                               | **INCONCLUSIVE** (C3a − C2a = +2,96 [−1,26; +8,31]). **Quy tắc dừng CCMC-SR đã kích hoạt:** cơ chế hoạt động nhưng C3a/C3f không vượt L0 dù có rò rỉ nhãn | 03/11/2026, Exp03 |
| G3, G4 | —                                                             | —                                                                           | **NOT RUN** (CCMC-SR dừng theo quy tắc 4.7)                                                                                                               | 03/11/2026        |
| G3     | H2: learned Weed prior hơn learned Veg prior?                 | L3 − L2 ≥ +3 pp Small-Weed Recall, CI > 0, Crop không giảm quá 1 pp (Exp03) | PENDING                                                                                                                                                   | —                 |
| G4     | Lợi ích có đến từ việc nối prior vào SR?                      | L3 > K1, K2, K3, K5, K6 (Exp04–05)                                          | PENDING                                                                                                                                                   | —                 |

| Exp                                               | Trạng thái                                             | Ghi chú                                                                                     |
| ------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| 00 Code + setup                                   | DONE                                                   | 44/44 unit test pass trên Kaggle (`transformers 4.45.2`, torch 2.11, T4, commit `ea354cd`)  |
| 01 Baseline + chẩn đoán                           | DONE                                                   | L0 s0/s1, LiteSent s0, D0, D2 (2 seed), profile. Diag-D1 NOT RUN (đúng kế hoạch). Xem mục 3 |
| 02 Oracle C2/C3                                   | DONE nhưng **vô hiệu về khoa học**                     | λ không học được; xem mục 4                                                                 |
| 03 Oracle với λ hoạt động, 3 seed                 | DONE                                                   | 13/13 run; xem mục 5                                                                        |
| 04 Định vị nút thắt (decoder 1/2, nguồn chi tiết) | DONE                                                   | 8/8 run; nút thắt = độ phân giải dự đoán; xem mục 6                                         |
| 05 Phương pháp mới MCSD (3 seed)                  | DONE                                                   | 9/9 run; **MCSD-g PASS trên val**; MCSD và H-mech INCONCLUSIVE; xem mục 7                   |
| 06 Xác nhận seed 3–4, bảng 5 seed                 | DONE — xác nhận MCSD-g PASS; xem mục 8                 | [06_confirm_5seeds.md](06_confirm_5seeds.md)                                                |
| 07 Kiểm toán cơ chế MCSD                          | Kế hoạch đã khóa; chỉ validation, không mở test        | [07_mechanism_audit.md](07_mechanism_audit.md)                                              |
| 08 Dataset thứ hai + khóa test                    | Chưa chạy; chọn dataset và protocol trước khi xem test | [07_mechanism_audit.md](07_mechanism_audit.md)                                              |

---

## 2. Exp00 — Review triển khai code (02/11/2026)

### 2.1 Những gì đã cài đặt, đối chiếu với kế hoạch

| Yêu cầu trong kế hoạch                                                                                                          | Cài đặt                                                                                        | Vị trí                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| CCMC-SR: `w = eps+(1-eps)p`, `w_norm = w/mean_window(w)`, `X_mod = X(1+λ(w_norm−1))`, rìa ngoài cửa sổ SR có hệ số 1, tính FP32 | `window_normalized_factor`                                                                     | [src/nets/ccmc_sr.py](../src/nets/ccmc_sr.py)                                                                |
| Chỉ sửa đầu vào SR conv dùng cho K/V; Q và residual giữ nguyên; stage 2–3; mọi block                                            | Forward pre-hook trên `attention.self.sr` (đã xác minh source HF 4.45.2: Q được tính trước SR) | `CCMCMiTEncoder`                                                                                             |
| Prior từ **output stage 1** (sau norm, dạng 4D), cấm điều biến stage 1                                                          | Vòng lặp encoder viết tường minh, giống hệt HF; dùng stage 1 thì báo lỗi                       | `CCMCMiTEncoder.forward`                                                                                     |
| Prior head khởi tạo đồng đều, `λ = sigmoid(a)`, init 0,05                                                                       | Conv cuối khởi tạo bằng 0 nên `p = 0,5` và `w_norm = 1`; mỗi stage có một `λ`                  | `PriorHead`, `lam_logit`                                                                                     |
| Aux loss soft occupancy 150×150, BCE, 0,3·L_aux, không detach F1                                                                | `aux_target` + `train.py`                                                                      | [src/train.py](../src/train.py)                                                                              |
| L0, L1, L2, L3, K1, K2, K2b, K3, K5, K6, C2, C3, C3-K5, C3-K6                                                                   | `--prior_mode`, `--injection`, `--prior_perturb`                                               | xem [doc/README.md](README.md)                                                                               |
| Oracle intervention trên **cùng** checkpoint (2×2), GT-Weed + nhiễu                                                             | `--prior_override`, `--prior_noise` (dilate, erode, shift, drop)                               | [src/evaluate_ccmc.py](../src/evaluate_ccmc.py)                                                              |
| Small-Weed Recall theo bin tam phân vị train, ghép one-to-one, IoU 0,5 (và 0,3)                                                 | Dùng instance chính thức nếu có, nếu không thì connected components (ghi rõ `source`)          | [src/ccmc_metrics.py](../src/ccmc_metrics.py), [make_instance_bins.py](../src/scripts/make_instance_bins.py) |
| Boundary IoU, theo ngày bay, ma trận nhầm lẫn, chất lượng prior                                                                 | Có trong `*_summary.json`                                                                      | `evaluate_ccmc.py`                                                                                           |
| Paired bootstrap theo cluster (ảnh hoặc tile), paired seed, gate tự động                                                        | `compare_runs.py --gate=h2/g4/screen/abs`                                                      | [compare_runs.py](../src/scripts/compare_runs.py)                                                            |
| Diag-D2 linear probe (avg trước SR, sau SR, max trước SR), AUROC/AP/recall@precision, contrast small − large có CI              | `probe_sr.py`                                                                                  | [probe_sr.py](../src/scripts/probe_sr.py)                                                                    |
| FLOPs có tính matmul attention, latency đồng bộ CUDA, trung vị/p10/p90, peak VRAM                                               | `profile_model.py` (FlopCounterMode)                                                           | [profile_model.py](../src/scripts/profile_model.py)                                                          |
| Test chỉ chạy một lần                                                                                                           | `--split=test` bị từ chối nếu thiếu `--confirm_test_locked=True`                               | `evaluate_ccmc.py`                                                                                           |
| Tái lập/resume phiên Kaggle                                                                                                     | `run_config.json` (git commit, split hash, versions), `history.jsonl`, `last.pth` + `--resume` | `train.py`                                                                                                   |

### 2.2 Quyết định thiết kế đăng ký trước (cần bạn xác nhận; nếu đổi thì phải đổi **trước** khi xem kết quả validation)

1. **K5 (gate trước SR, không chuẩn hóa cửa sổ):** `factor = 1 + λ(w/mean_ảnh(w) − 1)`. Chuẩn hóa theo toàn ảnh giúp K5 vẫn trùng baseline khi prior đồng đều và có cùng biên độ với CCMC-SR. Như vậy khác biệt duy nhất so với L3 là chuẩn hóa **cục bộ theo cửa sổ SR**. Gate thô `X·w` (không chuẩn hóa) không trùng baseline lúc khởi tạo và chỉ có thể làm yếu feature, nên không dùng.
2. **K6:** prior được avg-pool về lưới K/V và chuẩn hóa theo toàn lưới, rồi gate nhân vào token K/V **sau** LayerNorm của SR, trước phép chiếu `key`/`value`. _Sửa so với kế hoạch ("trước LayerNorm"):_ HF áp `layer_norm` theo từng token ngay sau `sr`, mà LayerNorm bất biến với phép nhân vô hướng trên token. Nếu đặt gate trước LayerNorm thì gate bị triệt tiêu, K6 gần như trùng L0 và trở thành đối chứng giả (L3 sẽ "thắng" K6 một cách vô nghĩa).
3. **Một `λ` cho mỗi stage**, dùng chung cho 2 block của stage đó. Nhật ký train ghi `λ` theo từng epoch.
4. **Resize prior:** 150→75 (stage 2) khớp chính xác 2×2; 150→38 (stage 3) dùng `adaptive_avg_pool2d` nên chỉ căn chỉnh xấp xỉ. Chưa đo sensitivity khi lệch 1 pixel (để ở ablation).
5. **Prior head:** Conv3×3(32→32)–BN–ReLU–Conv1×1(32→1), khoảng 9,3 nghìn tham số; tạo **sau** decoder để decoder của L3 được khởi tạo giống L0 với cùng seed. Thứ tự batch và augmentation dùng generator riêng, nên L0/L2/L3/K\* có cùng seed sẽ thấy cùng chuỗi batch (paired thật).
6. **Small-Weed Recall chính** = recall@IoU 0,5 của bin nhỏ. Recall@0,3 và coverage (≥ 50 % pixel của instance được dự đoán là Weed) là thước đo phụ. Coverage giúp phân biệt "bỏ sót" với "dính vào cỏ bên cạnh".
7. **Cluster bootstrap:** ~~theo tile~~ → **theo ảnh** (sửa sau Exp01, xem 3.6 mục 4: mã tile không nhất quán giữa các ngày, 25 tile cho 26 ảnh).
8. **K2 khi đánh giá:** phép hoán vị dùng seed cố định 0 để kết quả tái lập được.
9. **C2/C3:** checkpoint được chọn theo val mIoU trong khi vẫn dùng prior GT của val, nên có hai lớp leakage. Đây chỉ là sàng lọc, mọi output đều mang cờ `GT_LEAKAGE`.

**Rủi ro toán học cần biết trước (ảnh hưởng claim novelty):** vì LayerNorm sau SR bất biến với hệ số vô hướng trên từng token K/V, thành phần "tổng khối lượng của cửa sổ" gần như bị LayerNorm loại bỏ (chỉ còn khác biệt nhỏ qua bias của SR conv và qua phép trộn `1 + λ(·)`). Hệ quả: **K5 (gate trước SR, chuẩn hóa toàn ảnh) có thể cho kết quả rất gần L3**, vì trong mỗi cửa sổ hai toán tử có tỉ lệ trọng số tương đối gần như nhau. Dự đoán đăng ký trước: nếu L3 ≈ K5 thì đây là hệ quả toán học, **không** được claim đóng góp riêng của "window normalization". Khi đó novelty phải dựa vào việc _tái trọng số theo prior bên trong receptive field của SR_, chứng minh bằng L3 so với K6 (gate sau giảm độ phân giải), K1, K2, K3 và L3 so với L2.

### 2.3 Kiểm toán sơ bộ từ dữ liệu có sẵn trong repo (chưa cần chạy)

| Vấn đề                                                  | Phát hiện                                                                                                                                                                                                                                                                                                                                                                                                           | Hệ quả                                                                                                                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 82,16 so với 83,53 (không LLR)                          | [paper/ablation_results.csv](../paper/ablation_results.csv): `no_refine` gồm 5 seed **trên test** = 82,94 / 83,98 / 83,42 / 83,50 / 83,83, trung bình **83,53**, khớp con số trong kế hoạch. README gốc ghi "SegFormer + pretrained MiT, **CE**" đạt **82,07 % val**; còn CE + Dice (λ = 0,5, trọng số 0,8/1,0/1,5) đạt 82,62 % val. Vì vậy 82,16 nhiều khả năng đến từ cấu hình **chỉ dùng CE** và/hoặc khác split | Không gộp hai số. Exp01 chạy lại L0 (CE + Dice 0,3, trọng số 0,9/1,0/1,3, có fusion, không LLR) trong cùng pipeline, chỉ báo cáo **val**. Cần đối chiếu log gốc của Bảng 5 để xác nhận |
| FLOPs 3,49 G và 17,09 G                                 | `measure_model_stats` trong `train.py` đo bằng **THOP tại 512×512** (mặc định `input_size=(512,512)`), không phải 600×600, và THOP bỏ sót matmul của attention. Latency cũ đo bằng `time.time()` **không** có `cuda.synchronize`                                                                                                                                                                                    | Hai số này không dùng được làm bằng chứng hiệu quả. Exp01 đo lại ở 600×600 bằng FlopCounterMode, có đồng bộ CUDA                                                                       |
| `set_detect_anomaly(True)` luôn bật trong `train.py` cũ | Đã chuyển thành cờ `--detect_anomaly` (mặc định tắt). Cờ này chỉ thêm kiểm tra, không đổi giá trị số, nhưng làm train chậm đáng kể                                                                                                                                                                                                                                                                                  | Các lần chạy cũ và mới có thể so sánh về số liệu; thời gian train thì không                                                                                                            |

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

| Cell      | Nội dung                    | Trạng thái                                                                                                                                                 |
| --------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1.1      | Bin kích thước              | DONE: instance **chính thức** (PNG mode `L`, uint8), 156/156 ảnh; edges = **[45, 147] px**; val có 620/567/621 instance nhỏ/vừa/lớn, nên mọi bin đủ n ≥ 20 |
| E1.2      | L0 seed 0, 1                | DONE                                                                                                                                                       |
| E1.3      | LiteSent seed 0             | DONE                                                                                                                                                       |
| E1.4–E1.6 | Diag-D0, LiteSent so với L0 | DONE                                                                                                                                                       |
| E1.7      | Diag-D2 probe (2 seed)      | DONE                                                                                                                                                       |
| E1.8      | Profile                     | DONE                                                                                                                                                       |
| Diag-D1   | Quét `R`                    | NOT RUN (thăm dò, kế hoạch cho phép)                                                                                                                       |

### 3.2 Baseline và kiểm toán

| Mô hình (val) | mIoU  | Crop  | Weed  | best epoch | s/epoch |
| ------------- | ----- | ----- | ----- | ---------- | ------- |
| L0 s0         | 82,94 | 75,11 | 75,54 | 93         | 9,9     |
| L0 s1         | 82,79 | 74,74 | 75,46 | 94         | 10,0    |
| LiteSent s0   | 84,56 | 77,44 | 77,81 | 92         | 14,3    |

- **Tái lập được:** chênh lệch giữa 2 seed của L0 là 0,15 mIoU. LiteSent đạt 84,56, khớp con số 84,63 val trong README cũ.
- **Kiểm toán 82,16 so với 83,53:** L0 với CE + Dice đạt khoảng 82,9 val. Các dòng cũ là 82,07 (CE) và 82,62 (CE + Dice với λ = 0,5). Vậy 82,16 thuộc cấu hình khác loss (nhiều khả năng chỉ dùng CE). Tham chiếu chuẩn từ nay là **L0 trong pipeline này**. Số 83,53 là test của cấu hình tương tự L0, không dùng để so với val.
- **Lưu ý:** best epoch rơi vào 92–94/100, tức mô hình có thể chưa bão hòa. Protocol 100 epoch đã khóa và áp dụng như nhau cho mọi mô hình nên vẫn công bằng. Đây là giới hạn cần ghi trong bài, không đổi giữa chừng.
- Train mất khoảng **17 phút/run** (`t ≈ 0,3 GPU-giờ`). Năm seed của 3 mô hình chỉ tốn khoảng 5 GPU-giờ, nên quota không phải nút thắt.

### 3.3 Diag-D0: cỏ nhỏ có bị bỏ sót nhiều hơn không? → **CÓ, rõ ràng**

| L0 s0 (val)                             | small (≤ 45 px) | medium | large (> 147 px) |
| --------------------------------------- | --------------- | ------ | ---------------- |
| Coverage (≥ 50 % pixel dự đoán là Weed) | **63,5**        | 92,4   | 98,6             |
| Recall@IoU 0,5                          | 49,7            | 69,8   | 47,8             |
| Recall@IoU 0,3                          | 66,9            | 77,6   | 57,8             |

- `cov_large − cov_small` (bootstrap theo ảnh): L0 s0 **+35,0 [23,6; 44,2]**; L0 s1 **+40,2 [28,5; 49,3]**.
- **Hiệu ứng vẫn còn khi tách theo ngày**, tức không chỉ do ngày đầu có nhiều cỏ nhỏ: 2023-05-30 +17,2 [5,3; 30,0]; 2023-06-06 +28,9 [9,1; 39,2] (s0). s1 cho kết quả tương tự.
- **Phát hiện về metric:** recall@IoU 0,5 của bin _large_ thấp (khoảng 48 %) dù coverage đạt 98,6 %. Lý do là cỏ lớn chạm nhau bị dự đoán thành một khối dính liền, nên ghép one-to-one theo IoU thất bại; ngày 2023-06-15 chỉ có recall50_large = 10 % trong khi coverage là 97 %. Vì vậy phép so sánh `recall50 large − small` ở Cell E1.5 (≈ 0, CI rất rộng) **không đo được việc bỏ sót** và không dùng cho G1. Với bin small, recall50 vẫn hợp lệ: cỏ nhỏ ít dính nhau, và recall50_small (49,7) cùng chiều với cov_small (63,5).
- Ma trận nhầm lẫn: lỗi của Weed chủ yếu là **Weed → Background (6,8 %)**, lớn hơn Weed → Crop (2,4 %). Đây là kiểu lỗi "mất cỏ" mà CCMC-SR nhắm tới, không phải lỗi phân biệt Crop/Weed.
- Theo ngày: 2023-05-25 có Weed IoU chỉ 37,5 % và chứa 283/620 cỏ nhỏ của val. Đây là nhóm có nhiều dư địa cải thiện nhất (dự đoán T3 trong kế hoạch).

**LiteSent so với L0 (seed 0, paired bootstrap):** LLR tăng Small-Weed Recall **+14,5 [+11,4; +17,7]**, cov_small +8,4 [+4,6; +12,4], nhưng recall50_large chỉ +2,1 và mIoU +1,6. Lợi ích của chi tiết độ phân giải cao **tập trung vào cỏ nhỏ**. Điều này ủng hộ giả thuyết "mất chi tiết không gian", nhưng LLR tác động ở decoder nên **không** chứng minh riêng SR là nguyên nhân.

### 3.4 Diag-D2: linear decodability trước và sau SR

| Stage (R) | AUROC small: pre_avg → post_sr | pre_max   | Contrast (small − large), s0 | s1                          |
| --------- | ------------------------------ | --------- | ---------------------------- | --------------------------- |
| 1 (8)     | 0,605 → 0,566                  | 0,694     | −0,019 [−0,049; +0,008]      | −0,051 [−0,094; −0,018]     |
| **2 (4)** | **0,845 → 0,816**              | **0,904** | **+0,028 [+0,014; +0,042]**  | **+0,025 [+0,007; +0,040]** |
| 3 (2)     | 0,941 → 0,944                  | 0,945     | −0,001 [−0,006; +0,003]      | −0,004 [−0,007; −0,001]     |

Diễn giải:

- **Stage 2:** cỏ nhỏ mất khả năng giải mã qua SR (AUROC giảm khoảng 0,028) trong khi cỏ lớn gần như không mất (0,001). Contrast dương với CI > 0 **ở cả 2 seed**. Đây là bằng chứng chính cho H1.
- **Stage 1:** cỏ nhỏ gần mức ngẫu nhiên ngay cả _trước_ SR (AUROC 0,61), tức hiệu ứng sàn. Contrast âm vì cỏ lớn có nhiều thứ để mất hơn, nên stage 1 không diễn giải được cho H1. Dù sao, stage 1 cũng không nằm trong pilot vì prior lấy từ output stage 1.
- **Stage 3:** SR 2×2 không làm mất thông tin, nên **điều biến stage 3 có thể không cần**.
- **Phát hiện quan trọng cho thiết kế:** `pre_max` luôn cao hơn `pre_avg` và `post_sr` đối với cỏ nhỏ ở stage 1–2 (stage 2: 0,904 so với 0,816). Tín hiệu cỏ nhỏ **có mặt** trong cửa sổ trước SR, nhưng phép gộp tuyến tính gần với trung bình đã làm loãng nó. Đây đúng là cơ chế CCMC-SR nhắm tới: tái trọng số bên trong cửa sổ để tín hiệu cỏ không bị pha loãng.
- Độ lớn hiệu ứng nhỏ (ΔAUROC ≈ 0,03) và đây là bằng chứng tương quan, không phải nhân quả.

### 3.5 Chi phí tính toán (600×600, fp32, batch 1, T4, FlopCounter có tính matmul attention)

| Mô hình      | Params           | GFLOPs (GMACs)            | Latency trung vị (p10–p90)           | Peak VRAM |
| ------------ | ---------------- | ------------------------- | ------------------------------------ | --------- |
| L0           | 3,456 M          | 13,27 (6,63)              | 14,85 ms (14,59–14,89)               | 145 MB    |
| L3 / K5 / K6 | 3,465 M (+9,3 k) | 13,68 (6,84), **+3,1 %**  | 15,35 / 15,50 / 15,53 ms, **+3,4 %** | 151 MB    |
| LiteSent     | 3,662 M          | 50,41 (25,21), **+280 %** | 24,68 ms, +66 %                      | 299 MB    |

- Các số cũ 3,49 G và 17,09 G thực chất là **MACs của THOP ở 512×512**, chỉ tính conv. Không dùng lại hai số này trong bài.
- Ý nghĩa cho paper: LLR mua +14,5 pp Small-Weed Recall với giá +280 % FLOPs. L3 chỉ tốn +3 %. Nếu L3 lấy lại được một phần đáng kể khoảng cách này với CI rõ thì đó là kết quả quality/compute có giá trị.
- Số `modulation_elementwise_Mops` của K6 đang ước lượng theo kích thước đầu vào SR nên chưa chính xác; con số nhỏ nên không ảnh hưởng kết luận.

### 3.6 Quyết định G1: **PASS có điều kiện**

Theo quy tắc đăng ký trước, D0 cho thấy nhóm nhỏ suy giảm rõ (coverage, CI > 0, kể cả trong từng ngày) **và** D2 cho contrast dương với CI > 0 ở ít nhất một stage (stage 2, cả 2 seed). Các điều kiện kèm theo:

1. Bằng chứng mang tính tương quan; hiệu ứng ở stage 2 nhỏ. Trong bài chỉ được viết "consistent with", không viết "SR causes".
2. Stage 3 không có dấu hiệu mất thông tin. Cấu hình pilot vẫn giữ stage 2–3 như đã khóa, nhưng **ablation "chỉ stage 2" được nâng lên ưu tiên cao** (Exp04/05).
3. **Sửa đổi metric trước khi có bất kỳ kết quả nào của phương pháp:**
   - **(a)** Gate vẫn dùng `SmallWeedRecall` (recall@0,5, bin nhỏ), giữ nguyên như đã đăng ký. `cov_small` được nâng thành _đồng chính_: chỉ kết luận PASS khi hai chỉ số **cùng dấu**.
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

| Mục                                          | Trạng thái                                                                                        |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 8 run train (C2, C3, C3K5, C3K6 × seed 0, 1) | DONE                                                                                              |
| Đánh giá lại L0 s0/s1, LiteSent (BIoU 3 px)  | DONE                                                                                              |
| E2.4 nhiễu prior trên C3 s0                  | DONE                                                                                              |
| C3 so với C2, C3K5, C3K6                     | DONE                                                                                              |
| C3 so với L0 và C2 so với L0                 | **Lỗi** `Missing columns pq_*` (L0 không có cột chất lượng prior), đã sửa trong `compare_runs.py` |

### 4.2 Kết quả (val, trung bình 2 seed)

| Mô hình                       | SmallWeedRecall       | cov_small | mIoU  | Weed IoU | λ cuối (stage 2; 3) |
| ----------------------------- | --------------------- | --------- | ----- | -------- | ------------------- |
| L0                            | 48,23 (49,68 / 46,77) | 60,89     | 82,87 | 75,50    | —                   |
| C3 (GT-Weed)                  | 48,15                 | 62,66     | 82,82 | 75,44    | 0,051; 0,053        |
| C2 (GT-Veg)                   | 49,84                 | 64,68     | 82,83 | 75,49    | 0,050; 0,052        |
| C3-K5                         | 44,68                 | 57,02     | 82,66 | 75,32    | 0,052; 0,054        |
| C3-K6                         | 47,74                 | 62,26     | 82,56 | 75,08    | 0,052; 0,054        |
| LiteSent (tham chiếu, 1 seed) | 64,19                 | 71,94     | 84,56 | 77,81    | —                   |

### 4.3 Chẩn đoán: **cơ chế chưa từng hoạt động**, Exp02 không kiểm định được giả thuyết

1. **λ bị kẹt về mặt toán học.** Adam dịch mỗi tham số khoảng ≤ lr mỗi bước. Với lr 3e-4 và 100 epoch × 13 iter = 1 300 bước, logit `a` đổi được nhiều nhất khoảng 0,39, nên λ ≤ sigmoid(−2,94 + 0,39) ≈ 0,07. Quỹ đạo thực tế là 0,050 → 0,054. Khởi tạo λ = 0,05 trong kế hoạch là để đảm bảo trùng baseline, nhưng phần đó đã được prior head đồng đều đảm bảo, nên λ nhỏ là không cần thiết và làm tắt cơ chế. _Đây là lỗi thiết kế tối ưu hóa, không phải bằng chứng về giả thuyết._
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
- Chưa sang L2/L3 cho tới khi Exp03 trả lời được câu hỏi: _khi λ thực sự học được, prior GT có tác động lên đầu ra không?_

### 4.6 Code đã sửa sau Exp02

- `train.py`: thêm `--lam_lr` (param group riêng cho logit λ). Exp03 dùng `--lam_init=0.5 --lam_lr=0.01`; khởi tạo vẫn trùng baseline với L2/L3 vì prior head đồng đều. Cấu hình λ cố định dùng `--learn_lambda=False --lam_init=0.999`.
- `compare_runs.py`: bootstrap phân tầng theo seed, yêu cầu ≥ 3 seed, chỉ dùng cột chung.

### 4.7 Quy tắc dừng đăng ký trước cho Exp03 (chống "đào dữ liệu")

| Kết quả Exp03                                                             | Quyết định                                                                                                                     |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Activity: \|C3a(GT) − C3a(uniform)\| < 1 pp Small-Weed Recall ở cả 3 seed | Toán tử **không có đòn bẩy**, kể cả với nhãn thật. Dừng CCMC-SR dạng hiện tại                                                  |
| Activity rõ, nhưng C3a **và** C3f đều không vượt L0 (CI chứa 0 hoặc âm)   | Toán tử thay đổi đầu ra nhưng không có lợi. Dừng CCMC-SR; không chuyển sang L3                                                 |
| C3a hoặc C3f > L0, nhưng C3a ≤ C2a                                        | G2 NO-GO cho tính _Weed-specific_. Có thể chuyển hướng sang prior thực vật (L2 làm phương pháp chính) và phải ghi rõ trong bài |
| C3a > L0 và C3a > C2a (SCREEN-PASS, 3 seed)                               | Sang Exp04: L2, L3 với cùng cấu hình λ (init 0,5, lam_lr 0,01)                                                                 |

### 4.8 Về yêu cầu "cải tiến thật, chưa có paper công bố"

- **CCMC-SR không bỏ bớt gì của mô hình pretrained.** Toàn bộ trọng số MiT-B0 được giữ nguyên; chỉ **đầu vào của phép giảm K/V** được tái trọng số theo prior lớp. Đây là sửa đổi toán tử bên trong backbone, không phải cắt bớt hay ghép mô hình.
- **Không thể bảo đảm tuyệt đối "chưa có paper nào công bố".** Điều làm được là rà soát có hệ thống: tìm kiếm có ghi lại từ khóa, cơ sở dữ liệu, ngày, và đối chiếu công thức với các công trình gần nhất ở mục 4 của kế hoạch (MLSRS-SegFormer APW, Dynamic Spatial Sparsification, ALGM, VLTP, SegMAN). Việc này nằm ở P7, nhưng **chỉ có ý nghĩa nếu Exp03 cho thấy toán tử có tác dụng**. Một phương pháp mới mà không có hiệu quả thì cũng không đăng Q1 được.
- Nếu Exp03 rơi vào hai dòng đầu của bảng 4.7, phương án dự phòng **vẫn nằm trong kế hoạch** (mục 4.3): "dual-statistic SR". Phương án này dùng một nhánh gộp có trọng số theo prior và một nhánh SR chuẩn, chung conv và pha trộn bằng hệ số học được. Lý do được dẫn từ Diag-D2: `pre_max` giữ tín hiệu cỏ nhỏ tốt hơn hẳn `post_sr`. Phương án này chỉ được mở khi có lý do ghi rõ trong báo cáo, không mở tự động.

---

## 5. Exp03 — Oracle với λ hoạt động, 3 seed (review 03/11/2026)

Artifact: [doc/exp03_results/results/exp03](exp03_results/results/exp03). 13/13 run DONE. So sánh dùng bootstrap phân tầng (seed paired + ảnh).

### 5.1 λ đã học được (sửa lỗi Exp02 thành công)

| Mô hình                        | λ cuối (stage 2; 3), theo seed       |
| ------------------------------ | ------------------------------------ |
| C3a (GT-Weed, CCMC-SR)         | 0,85; 0,82 · 0,80; 0,86 · 0,87; 0,80 |
| C2a (GT-Veg, CCMC-SR)          | 0,50; 0,62 · 0,52; 0,59 · 0,61; 0,54 |
| C3K5a (GT-Weed, gate trước SR) | 0,83; 0,89 · 0,78; 0,93 · 0,86; 0,91 |
| C3f                            | 0,999 (cố định)                      |

Khi prior là GT-Weed, λ tăng mạnh hơn GT-Veg. Mô hình **muốn** dùng prior Weed nhiều hơn prior thực vật, đây là một quan sát mô tả.

### 5.2 Kết quả chính (Small-Weed Recall, val)

| Mô hình                       | s0 / s1 / s2          | Trung bình         | So với L0 [CI 95 %]                  |
| ----------------------------- | --------------------- | ------------------ | ------------------------------------ |
| **L0**                        | 49,68 / 46,77 / 52,90 | **49,78** (SD 3,1) | —                                    |
| C3a                           | 45,97 / 48,87 / 50,48 | 48,44              | −1,34 [−4,74; +2,30]                 |
| C3f (λ = 1)                   | 46,94 / 46,29 / 49,52 | 47,58              | −2,20 [−5,14; +0,35]                 |
| C2a                           | 46,61 / 47,58 / 42,26 | 45,48              | −4,30 [−10,32; +0,92]                |
| C3K5a                         | 49,52 / 40,97 / 50,81 | 47,10              | (C3a − C3K5a = +1,34 [−3,78; +7,54]) |
| LiteSent (1 seed, tham chiếu) | 64,19                 | —                  | +14,5 (Exp01)                        |

mIoU của mọi mô hình nằm trong khoảng 82,4–82,9, không khác L0.

### 5.3 Cơ chế **có** hoạt động: đổi prior trên cùng checkpoint

| Checkpoint | GT-Weed so với prior đồng đều (Small-Weed Recall) | cov_small              |
| ---------- | ------------------------------------------------- | ---------------------- |
| C3a        | **+4,14 [+1,84; +6,69]**                          | +5,48 [+2,92; +8,27]   |
| C3f        | **+9,78 [+5,27; +15,67]**                         | +14,09 [+9,21; +19,53] |

Diễn giải: mô hình **dựa vào** prior. Bỏ prior đi thì tụt, ví dụ C3f với prior đồng đều chỉ còn 37,8 so với L0 49,8. Tuy vậy, dùng prior cũng **không vượt** L0. Toán tử chỉ khiến mạng _chuyển phụ thuộc_ sang prior, không bổ sung năng lực. Ngay cả nhãn thật đưa qua K/V SR của stage 2–3 cũng không tăng recall cỏ nhỏ.

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

| Phần của paper                                                               | Trạng thái  | Dùng được không                                              |
| ---------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| Baseline tái lập, kiểm toán số liệu cũ, FLOPs/latency đo đúng                | DONE        | Có                                                           |
| Chẩn đoán cỏ nhỏ (D0: coverage theo bin và theo ngày; lỗi Weed → Background) | DONE        | Có                                                           |
| Probe SR (D2): mất thông tin ở stage 2, max-pool giữ tốt hơn                 | DONE        | Có (phân tích)                                               |
| Oracle CCMC-SR: âm tính có kiểm soát (cơ chế hoạt động nhưng không lợi)      | DONE        | Có (kết quả âm tính/ablation, **không** phải đóng góp chính) |
| Phương pháp thống kê: A/A test, bootstrap phân tầng                          | DONE        | Có                                                           |
| **Phương pháp mới vượt baseline có ý nghĩa**                                 | **Chưa có** | Đây là phần quyết định Q1                                    |
| Rà soát novelty, 5 seed, test một lần, dataset thứ hai                       | Chưa        | —                                                            |

### 5.7 Việc tiếp theo: Exp04, định vị nút thắt trước khi thiết kế phương pháp mới

[04_bottleneck_localization.md](04_bottleneck_localization.md): so LiteSent (decoder 1/2 + băng thô) với **D½-none** (decoder 1/2, không có chi tiết) và **D½-F1** (decoder 1/2, chi tiết lấy từ F1), mỗi cấu hình 3 seed. Thí nghiệm này cũng trả lời trực tiếp hai đối chứng thầy yêu cầu. Bảng diễn giải đã khóa trước trong file Exp04. Phương pháp mới chỉ được thiết kế _sau_ khi biết nút thắt nằm ở:

- (i) độ phân giải decoder,
- (ii) patch embedding stage 1 của MiT, hay
- (iii) cách decoder dùng F1.

Nếu là (ii), hướng ưu tiên là **sửa toán tử patch embedding stage 1 bên trong MiT**: giữ trọng số pretrained, không ghép mô hình. Việc này phải rà soát novelty có ghi lại trước khi code.

Code đã thêm: `--llr_source {raw, none, f1}` cho `SegFormer5Band`. Mặc định là `raw` nên checkpoint LiteSent cũ vẫn tải được.

---

## 6. Exp04 — Định vị nút thắt (review 03/11/2026)

Artifact: [doc/exp04_results/results/exp04](exp04_results/results/exp04). 8/8 run DONE. Mọi mô hình có 3 seed, so sánh bằng bootstrap phân tầng.

### 6.1 Kết quả (val, trung bình 3 seed)

| Mô hình                               | Small-Weed Recall (s0/s1/s2) | cov_small | mIoU  | Weed IoU | BIoU Weed | GFLOPs | Latency T4 |
| ------------------------------------- | ---------------------------- | --------- | ----- | -------- | --------- | ------ | ---------- |
| L0 (1/4 + bilinear)                   | 49,78 (49,7/46,8/52,9)       | 62,47     | 82,77 | 75,25    | 52,04     | 13,27  | 14,85 ms   |
| D½-none (decoder 1/2, không chi tiết) | 61,08 (59,5/61,0/62,7)       | 70,81     | 83,63 | 76,39    | 55,72     | 40,07  | 21,16 ms   |
| D½-F1 (decoder 1/2 + F1)              | 58,06                        | 67,10     | 83,72 | 76,52    | 55,04     | 50,30  | 23,42 ms   |
| LiteSent (decoder 1/2 + băng thô)     | 63,55 (64,2/58,1/68,4)       | 71,40     | 84,50 | 77,77    | 57,90     | 50,41  | 23,54 ms   |

| So sánh                                            | Small-Weed Recall                                    | mIoU                     | BIoU Weed                |
| -------------------------------------------------- | ---------------------------------------------------- | ------------------------ | ------------------------ |
| LiteSent − L0 (tổng lợi ích LLR, giờ đã có 3 seed) | **+13,76 [+10,26; +17,78]**                          | +1,73 [+1,19; +2,56]     | +5,86                    |
| D½-none − L0 (chỉ tăng độ phân giải dự đoán)       | **+11,29 [+8,11; +14,82]**                           | +0,86 [+0,23; +1,67]     | +3,68                    |
| LiteSent − D½-none (giá trị thêm của băng thô)     | +2,47 [−2,99; +7,44], không có ý nghĩa               | **+0,87 [+0,48; +1,25]** | **+2,18 [+0,03; +3,93]** |
| D½-F1 − D½-none                                    | −3,01 [−6,97; +1,05]; cov_small −3,71 [−7,07; −0,46] | +0,09                    | −0,68                    |
| LiteSent − D½-F1                                   | +5,48 [−2,88; +13,01]                                | **+0,78 [+0,52; +1,07]** | **+2,86**                |

### 6.2 Diễn giải theo bảng đã khóa trong Exp04: **dòng 1 (D½-none ≈ LiteSent cho cỏ nhỏ)**

- **Nút thắt của cỏ nhỏ là độ phân giải của bước dự đoán.** Chỉ cần dự đoán ở 1/2 thay vì 1/4, không thêm thông tin nào, đã lấy lại **82 %** lợi ích cỏ nhỏ của LLR (11,29/13,76). Thông tin về cỏ nhỏ **vẫn còn** trong feature 1/4 của decoder; nó bị mất ở bước "một phân phối lớp cho mỗi ô 4×4 px rồi bilinear ×4".
- **Băng thô đóng góp cho biên và pixel IoU**: mIoU +0,87, BIoU Weed +2,18, có ý nghĩa. Phần này không giúp phát hiện thêm cỏ nhỏ.
- **F1 không thay được băng thô**, thậm chí làm giảm coverage cỏ nhỏ. Chi tiết hữu ích cho biên nằm trong đầu vào gốc, không nằm trong F1.
- **Trả lời hai yêu cầu của thầy** (Ưu tiên 2): lợi ích của nhánh refinement **chủ yếu đến từ việc tăng độ phân giải decoder** (đối với cỏ nhỏ), còn nhánh raw-input góp phần biên/mIoU và tốt hơn dùng F1. Bảng 6.1 dùng thẳng được trong paper.
- **Chi phí:** D½-none tăng +26,8 GFLOPs (×3,0) và +6,3 ms chỉ cho một conv 3×3 128 kênh trên lưới 300×300.
- Phương sai seed của LiteSent rất lớn (58,1–68,4). Các so sánh 1 seed trước đây (bài cũ, README cũ) không đáng tin, đúng với A/A test ở Exp03.

### 6.3 Phương pháp mới đề xuất: **MCSD (Mass-Conserving Sub-cell Decoding)**

**Chuỗi lập luận từ bằng chứng sang thiết kế:**

1. Cỏ nhỏ ≤ 45 px, tức chỉ khoảng 1–3 ô 4×4.
2. Classifier 1/4 + bilinear + argmax chỉ đưa ra **một** phân phối cho mỗi ô. Nếu Weed chỉ chiếm 15 % ô thì argmax = Background, và cỏ bị xóa. Đây là lỗi chính Weed → Background 6,8 % ở Exp01.
3. D½-none chứng minh thông tin vẫn nằm trong feature 1/4 (Exp04).
4. Diag-D2 cho thấy phép trung bình làm loãng bằng chứng cỏ nhỏ.

Vì vậy cần một bước giải mã **bảo toàn khối lượng lớp trong mỗi ô** và đặt khối lượng đó vào đúng sub-pixel, với chi phí nhỏ hơn nhiều so với decoder 1/2 dày đặc.

**Toán tử** ([src/nets/mcsd.py](../src/nets/mcsd.py)): trên feature 1/4 của L0 có hai head 1×1:

- **occupancy** `q_c` = tỉ lệ lớp của ô, học bằng soft-CE với tỉ lệ GT;
- **placement** `s_cj` = điểm lớp c tại sub-pixel j (16 sub-pixel mỗi ô).

Xác suất sub-pixel là kế hoạch vận chuyển tối ưu entropic `P = diag(u)·exp(s/τ)·diag(v)` với biên hàng `16·q_c` và biên cột 1, giải bằng 5 vòng Sinkhorn log-domain. Toàn bộ khả vi, không có tham số ở bước ghép, và là một log-prob hợp lệ nên dùng thẳng CE + Dice. Giữ nguyên toàn bộ MiT-B0 pretrained và decoder 1/4; chỉ thay classifier 1×1 + bilinear. Chi phí head ước tính khoảng 0,15 GMAC, so với +13,4 GMAC của D½-none.

**Đối chứng bắt buộc:** PS (cùng head, cùng occupancy loss, bỏ ràng buộc Sinkhorn) để chứng minh lợi ích đến từ **bảo toàn khối lượng**, chứ không phải từ việc có thêm head sub-pixel. Gate đã khóa trong [05_mcsd.md](05_mcsd.md).

**Liên kết với câu chuyện của paper:** ý tưởng "bảo toàn khối lượng" của kế hoạch gốc được giữ, nhưng **chuyển từ K/V SR** (nơi chẩn đoán và oracle cho thấy không phải nút thắt) **sang bước giải mã dự đoán** (nơi Exp04 định vị nút thắt). Paper có mạch: chẩn đoán → loại trừ giả thuyết SR bằng oracle → định vị nút thắt → toán tử nhắm đúng nút thắt.

### 6.4 Rà soát novelty sơ bộ (03/11/2026, ghi lại để tái lập)

| Nguồn / truy vấn                                                                                      | Kết quả                                                                                                                     |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| arXiv API: `Sinkhorn AND upsampling AND segmentation`                                                 | 0 kết quả                                                                                                                   |
| arXiv API: `"optimal transport" AND upsampling AND segmentation`                                      | 0                                                                                                                           |
| arXiv API: `"mass conserving" AND upsampling`; `"mass conservation" AND segmentation AND upsampling`  | 0 / 0                                                                                                                       |
| arXiv API: `"label proportion" AND segmentation AND resolution`; `"sub-pixel" AND "class proportion"` | 0 / 0                                                                                                                       |
| arXiv API: `"subpixel mapping"`                                                                       | UnMixMamba (arXiv 2610.38507, 10/2026): subpixel mapping cây trồng từ MODIS, dùng tỉ lệ lớp để phân bổ ngân sách refinement |

**Công trình gần nhất phải định vị và trích dẫn:**

- **SparseRefine** (ECCV 2024, arXiv 2407.19014): entropy selector + refine thưa. Khác MCSD: chọn pixel theo entropy, không bảo toàn khối lượng.
- **PointRend** (CVPR 2020): refine điểm bất định.
- **CARAFE / DySample / FADE / SAPA**: upsampler theo nội dung của feature, không có ràng buộc tỉ lệ lớp.
- **DUC / PixelShuffle head**: tương ứng đối chứng PS.
- **Sub-pixel mapping trong viễn thám** (Atkinson; pixel-swapping; Hopfield SPM; deep SPM; UnMixMamba): **gần nhất về ý tưởng** (phân bổ sub-pixel theo tỉ lệ lớp). SPM cổ điển là hậu xử lý từ ảnh tỉ lệ (unmixing). MCSD là lớp giải mã khả vi, end-to-end trong mạng phân đoạn, ghép occupancy và placement bằng OT/Sinkhorn.

**Kết luận trung thực:** chưa thấy công bố trùng trong lần tìm sơ bộ này, nhưng arXiv API **không** thay được rà soát đầy đủ. Trước khi nộp phải tìm trên Scopus, IEEE Xplore và Google Scholar với các cụm: "sub-pixel mapping deep learning end-to-end", "Sinkhorn segmentation decoder", "optimal transport upsampling", "proportion-constrained decoding", "area-preserving segmentation". Tuyệt đối không claim "first" nếu chưa làm xong bước này.

### 6.5 Code đã thêm

- [src/nets/mcsd.py](../src/nets/mcsd.py): `sinkhorn_subcell`, `MCSDSegFormer` (`head ∈ {bilinear, mcsd, pixelshuffle}`, `spectral_guide`), `occupancy_loss`.
- `segformer.py`: tách `decode_features()` (dùng chung decoder 1/4).
- `train.py`: `--head`, `--sinkhorn_iters`, `--mcsd_tau`, `--spectral_guide`, `--occ_weight`.
- `ccmc_common.py` / `evaluate_ccmc.py`: hỗ trợ đánh giá và profile MCSD.
- 6 unit test mới: biên Sinkhorn, xếp sub-pixel, gradient, occupancy loss, forward/backward, decoder khởi tạo trùng L0.

---

## 7. Exp05 — MCSD, PS, MCSD-g, 3 seed (review 03/11/2026)

Artifact: [doc/exp05_results/results/exp05](exp05_results/results/exp05). 9/9 run DONE; mọi so sánh dùng bootstrap phân tầng.

### 7.1 Kết quả (val, trung bình 3 seed)

| Mô hình                                 | Small-Weed Recall | cov_small | mIoU      | Crop IoU  | Weed IoU  | BIoU Weed | GFLOPs | Latency T4 |
| --------------------------------------- | ----------------- | --------- | --------- | --------- | --------- | --------- | ------ | ---------- |
| L0 (1/4 + bilinear)                     | 49,78             | 62,47     | 82,77     | 74,92     | 75,25     | 52,04     | 13,27  | 14,85 ms   |
| PS (head sub-pixel, không bảo toàn)     | 54,03             | 65,75     | **80,93** | **71,23** | 73,56     | 49,39     | 13,54  | 14,52 ms   |
| MCSD                                    | 54,09             | 64,19     | 83,29     | 75,47     | 76,14     | 55,00     | 13,54  | 16,41 ms   |
| **MCSD-g** (MCSD + placement từ 5 băng) | **55,59**         | 63,12     | **83,68** | 76,06     | **76,66** | **55,95** | 14,10  | 17,75 ms   |
| D½-none (decoder 1/2 dày)               | 61,08             | 70,81     | 83,63     | 76,20     | 76,39     | 55,72     | 40,07  | 21,16 ms   |
| LiteSent (decoder 1/2 + băng thô)       | 63,55             | 71,40     | 84,50     | 77,34     | 77,77     | 57,90     | 50,41  | 23,54 ms   |

### 7.2 Gate (đã khóa trong 05_mcsd.md)

| Gate                                  | Kết quả                                                                                                                                                                                                                                 | Quyết định                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| H-MCSD: MCSD − L0                     | +4,30 [−0,80; +9,80]; cov_small +1,72 (không ý nghĩa); Weed IoU +0,89, BIoU Weed +2,95 (có ý nghĩa)                                                                                                                                     | **INCONCLUSIVE**                                                                         |
| H-mech: MCSD − PS (Small-Weed Recall) | +0,05 [−3,90; +4,23]                                                                                                                                                                                                                    | **INCONCLUSIVE**. Không claim rằng bảo toàn khối lượng làm tăng recall cỏ nhỏ            |
| Crop guard (đăng ký trước, ≤ 1 pp)    | PS − L0: Crop **−3,69 [−6,98; −1,15]**, mIoU −1,84. MCSD − L0: Crop +0,55                                                                                                                                                               | **PS VI PHẠM; MCSD ĐẠT**                                                                 |
| (mô tả) MCSD-g − L0                   | **+5,81 [+3,25; +8,48]**, cov_small +0,65 (cùng dấu), mIoU +0,91 [+0,12; +1,84], Weed IoU +1,41, BIoU Weed +3,90, Crop +1,15                                                                                                            | **PASS** (3 seed), nhưng là biến thể được chọn sau khi xem val, nên cần xác nhận (Exp06) |
| H-eff: MCSD-g so với D½-none          | Lấy lại **51 %** lợi ích cỏ nhỏ (5,81/11,29) với **3,1 %** FLOPs thêm (+0,83 so với +26,8 G); latency thêm 46 % (+2,9 so với +6,3 ms). mIoU, Weed IoU, BIoU Weed **ngang D½-none** (83,68 / 76,66 / 55,95 so với 83,63 / 76,39 / 55,72) | Đạt mục tiêu Q1 về FLOPs; **không** đạt về latency                                       |

### 7.3 Diễn giải trung thực

1. **Vai trò thật của bảo toàn khối lượng là ổn định hóa, không phải tăng recall.** Head sub-pixel thuần (PS) cũng tăng recall cỏ nhỏ (+4,25 [+1,25; +7,31]) nhưng phá độ chính xác pixel: Crop −3,7, mIoU −1,8, một seed sập (PS s2: Crop 67,7). Ràng buộc Sinkhorn giữ được lợi ích sub-pixel mà **không** làm hỏng Crop/mIoU: MCSD − PS cho mIoU **+2,36 [+1,05; +4,10]**, Crop **+4,24**, BIoU +5,6–5,9. Claim đúng trong paper: _"mass conservation makes sub-cell decoding usable (stable and accuracy-preserving)"_, không phải _"mass conservation increases small-weed recall"_.
2. **Placement từ 5 băng (NIR/RE) góp phần thêm:** MCSD-g − MCSD cho mIoU +0,39 [+0,12; +0,81] và Crop +0,60, có ý nghĩa; Small-Weed Recall +1,5 không có ý nghĩa. Đây là điểm gắn phương pháp với dữ liệu đa phổ, hữu ích cho câu chuyện ứng dụng.
3. **Lợi ích của MCSD-g chủ yếu là hình dạng/định vị cỏ nhỏ tốt hơn:** recall@0,5 +5,8, recall@0,3 +3,9, BIoU +3,9, trong khi coverage chỉ +0,65. Nó chưa "tìm thêm" nhiều cỏ bị bỏ sót hoàn toàn. Phần coverage vẫn thua D½-none/LiteSent (−7 pp). Phải ghi đây là giới hạn.
4. **Bảo toàn khối lượng chỉ gần đúng với 5 vòng:** sai số tỉ lệ lớp lớn nhất trên mỗi ảnh là 0,073 (trung bình 0,059), và khối lượng Weed dự đoán bằng 1,10 lần GT (MCSD s0). Trong paper phải viết "approximately mass-conserving (5 Sinkhorn iterations)". Exp06 đo thêm với 1 và 20 vòng.
5. **Latency:** FLOPs chỉ +6 % nhưng latency +19,5 % (17,75 so với 14,85 ms). Sinkhorn là các phép elementwise/logsumexp trên tensor 5 chiều, bị giới hạn bởi bộ nhớ chứ không bởi FLOPs. Vẫn nhanh hơn D½-none (21,2 ms) và LiteSent (23,5 ms). Có thể tối ưu sau (ít vòng hơn, kernel gộp).
6. **Vị trí so với LiteSent:** MCSD-g vẫn thấp hơn LiteSent (Small-Weed Recall −8, mIoU −0,8) nhưng tốn **28 %** FLOPs và **75 %** latency. Hai kịch bản khả thi cho paper: (a) **Pareto hiệu quả**: MCSD-g ngang D½-none về pixel metrics với 35 % FLOPs; (b) **bổ trợ**: gắn MCSD lên decoder 1/2 của LiteSent (ô 2×2), để Exp07 nếu Exp06 xác nhận.

### 7.4 Vấn đề chọn biến thể sau khi xem val

MCSD-g ban đầu đăng ký là "mô tả", và việc chọn nó thay MCSD sau khi xem val là một lựa chọn hậu kiểm (2 biến thể). Để không thiên lệch: (i) **khóa** MCSD-g ngay bây giờ, không tinh chỉnh thêm; (ii) **xác nhận trên seed mới 3–4** (Exp06) bằng quy tắc đã đăng ký trước; (iii) báo cáo cả MCSD và MCSD-g trong paper.

### 7.5 Tiến độ gói bằng chứng Q1 (ước lượng)

| Phần                                                                         | Trạng thái             |
| ---------------------------------------------------------------------------- | ---------------------- |
| Chẩn đoán, oracle âm tính, định vị nút thắt, phương pháp thống kê            | DONE                   |
| Phương pháp mới có tín hiệu dương (val, 3 seed) + đối chứng cơ chế + chi phí | DONE                   |
| Xác nhận seed mới, bảng 5 seed                                               | Exp06                  |
| Test một lần; ablation (số vòng, τ, occ_weight, ô 2×2 trên LiteSent)         | Exp07                  |
| Dataset thứ hai (đề xuất WeedMap, RedEdge 5 băng, cùng bộ băng R,G,B,NIR,RE) | Exp08, cần viết loader |
| Rà soát novelty đầy đủ (Scopus/IEEE/Scholar), viết bài                       | Song song              |

Ước lượng: khoảng **60 %** gói bằng chứng, nếu Exp06 xác nhận.

### 7.6 Code đã thêm

- [src/scripts/mass_check.py](../src/scripts/mass_check.py): đo sai số bảo toàn khối lượng theo số vòng Sinkhorn (thay script viết tạm bằng `%%writefile`).

---

## 8. Exp06 — Xác nhận MCSD-g, bảng 5 seed và ablation inference (04/11/2026)

Artifact: [doc/exp06_results/results/exp06](exp06_results/results/exp06). Có **12/12 run train seed 3–4**, đủ metadata `run_config.json`, cùng commit `52a8843`, split hash `675667edf3f35f16`, 100 epoch. Đánh giá validation có đủ seed 0–4 cho các mô hình chính và hai biến thể inference 1/20 vòng. Bootstrap phân tầng seed + ảnh, 10.000 lần, 26 ảnh. **Không chạy test.**

### 8.1 Gate xác nhận MCSD-g: **PASS trên seed mới 3–4**

| Chỉ số, MCSD-g − L0 | Seed 3–4, delta [CI 95%] | Seed 0–4, delta [CI 95%] |
| ------------------- | -----------------------: | -----------------------: |
| Small-Weed Recall   | **+10,89 [5,80; 15,56]** |  **+7,84 [4,46; 11,66]** |
| `cov_small`         |  **+6,77 [1,46; 11,70]** |      +3,10 [−0,38; 7,20] |
| mIoU                |       +0,97 [0,46; 1,73] |       +0,93 [0,35; 1,78] |
| Crop IoU            |       +1,20 [0,11; 2,35] |       +1,17 [0,15; 2,28] |
| Weed IoU            |       +1,53 [0,96; 2,87] |       +1,46 [0,60; 3,01] |
| Weed Boundary IoU   |       +4,21 [3,11; 6,49] |       +4,03 [2,45; 6,25] |

- Cặp xác nhận seed 3–4 thỏa điều kiện đã đăng ký: recall dương với CI trên 0, `cov_small` cùng chiều và Crop không giảm. Delta Small-Weed Recall của MCSD-g so với L0 đều dương trên cả **5/5 seed** (+5,97; +5,32; +6,13; +13,39; +8,39 pp). Vì vậy MCSD-g được giữ làm cấu hình chính để tiếp tục kiểm toán và lập protocol test.
- Diễn giải cần chặt: trên 5 seed, bằng chứng recall là rõ; độ bất định của `cov_small` vẫn bao gồm 0. Có thể kết luận coverage **có xu hướng tăng**, không nên viết rằng coverage đã được xác nhận chắc chắn trên 5 seed.
- Xác nhận 2 seed là đúng thiết kế đã đăng ký nhưng không thay cho tái lập đa seed lớn hơn. Kết luận mạnh hơn đến từ việc gộp 5 seed, và kết quả vẫn chỉ là validation trên một dataset.

### 8.2 Cơ chế và trade-off

| So sánh validation        | Small-Weed Recall delta [CI 95%] | `cov_small` delta [CI 95%] |      mIoU delta [CI 95%] |  Crop IoU delta [CI 95%] |
| ------------------------- | -------------------------------: | -------------------------: | -----------------------: | -----------------------: |
| MCSD-g − MCSD, 5 seed     |              +3,03 [−2,04; 7,69] |        +0,52 [−4,31; 5,17] |   **+0,39 [0,13; 0,89]** |   **+0,54 [0,21; 1,01]** |
| MCSD − PS, 5 seed         |              −1,26 [−5,99; 3,01] |        −2,90 [−8,55; 2,60] |   **+1,82 [0,88; 3,08]** |   **+3,13 [1,33; 5,73]** |
| PS − L0, 5 seed           |           **+6,06 [2,86; 9,54]** |     **+5,48 [2,30; 9,04]** | **−1,28 [−2,35; −0,27]** | **−2,51 [−4,97; −0,60]** |
| MCSD-g − D½-none, 5 seed  |              −3,45 [−7,90; 0,68] |        −4,84 [−9,55; 0,21] |      −0,05 [−0,49; 0,38] |      −0,26 [−1,01; 0,54] |
| MCSD-g − LiteSent, 5 seed |            −6,84 [−10,94; −2,73] |      −6,58 [−11,08; −2,51] |     −0,84 [−1,17; −0,50] |     −1,20 [−1,88; −0,50] |

1. PS củng cố nhận định Exp05: head sub-pixel có thể tăng recall, nhưng làm giảm mIoU/Crop. MCSD so với PS không tăng recall có ý nghĩa; lợi ích rõ của Sinkhorn hiện tại là giữ chất lượng phân đoạn và Crop tốt hơn. Claim nên là **ràng buộc occupancy giúp sub-cell decoding bảo toàn độ chính xác tốt hơn**, không phải “Sinkhorn làm tăng Small-Weed Recall”.
2. MCSD-g cải thiện mIoU/Crop so với MCSD; mức tăng recall/coverage giữa hai biến thể chưa chắc chắn. Spectral guide hiện có bằng chứng tốt hơn cho chất lượng pixel/biên so với phát hiện thêm instance nhỏ.
3. MCSD-g ngang D½-none trong các metric pixel chính nhưng recall/coverage thấp hơn về điểm ước lượng. Nó vẫn là phương án hiệu quả tính toán theo profile Exp05, không phải phương án có recall cao nhất. So với LiteSent còn khoảng cách rõ về recall và pixel metrics.

### 8.3 Sinkhorn iterations — chỉ inference trên cùng checkpoint

| MCSD-g inference, so với 5 vòng |       Small-Weed Recall |              `cov_small` |                     mIoU |                 Crop IoU |
| ------------------------------- | ----------------------: | -----------------------: | -----------------------: | -----------------------: |
| 1 vòng, 5 seed                  | **+9,77 [6,67; 12,76]** | **+10,00 [6,97; 12,80]** | **−0,58 [−1,04; −0,33]** | **−1,03 [−1,55; −0,67]** |
| 20 vòng, 5 seed                 |    −0,61 [−1,31; −0,09] |     −0,48 [−1,09; −0,09] |     −0,12 [−0,22; −0,07] |     −0,25 [−0,44; −0,14] |

Trên MCSD-g s0, sai số marginal đầu ra so với occupancy **dự đoán** giảm theo số vòng: 1 vòng max 0,3553, mean 0,005754; 5 vòng max 0,1077, mean 0,0004323; 20 vòng max 0,01035, mean 0,00000209. Tỉ lệ tổng Weed probability / GT Weed pixel lần lượt 1,086 / 1,092 / 1,093. Mean ở đây là trung bình trị tuyệt đối trên các lớp và ô; max là sai số cực đại trên validation.

Kết quả 1 vòng cho thấy một **trade-off suy luận** rõ: recall/coverage tăng trong khi mIoU/Crop giảm, cùng lúc marginal constraint kém chính xác hơn. Đây là quan sát trên checkpoint đã train ở 5 vòng; **không chứng minh** rằng train với 1 vòng sẽ có cùng hiệu ứng hoặc rằng sai số marginal là nguyên nhân duy nhất làm recall tăng. Cần kiểm toán theo lớp và occupancy trước khi diễn giải cơ chế.

**Lưu ý định nghĩa:** `pred_weed_mass_over_gt` là tổng xác suất Weed dự đoán chia tổng pixel Weed GT. Nó không đo việc Sinkhorn có khớp occupancy dự đoán hay không. Chỉ số marginal error mới là chênh giữa trung bình xác suất sub-pixel đầu ra và `q` từ occupancy head. Hai đại lượng trả lời hai câu hỏi khác nhau; không gọi tỷ lệ 1,09 là “sai số bảo toàn khối lượng”.

### 8.4 Quyết định và việc tiếp theo

- **Exp06 hoàn tất; MCSD-g qua gate confirmatory và được giữ nguyên cấu hình 5 vòng, τ=1, occupancy loss 0,3.** Chưa mở test, chưa chọn 1/20 vòng thay cấu hình chính.
- Tiếp theo là **Exp07 — kiểm toán cơ chế**, theo protocol mới trong [07_mechanism_audit.md](07_mechanism_audit.md): thêm occupancy-conditioned baseline không Sinkhorn; đo marginal error theo từng lớp và occupancy GT nhỏ; phân tích ảnh/instance nơi 1 vòng tăng recall nhưng giảm accuracy. Không điều chỉnh cấu hình chính từ kết quả thăm dò.
- Song song rà soát novelty có ghi log, ưu tiên fractional class occupancy, sub-pixel mapping, constrained decoding và optimal transport/Sinkhorn. Chuẩn bị Exp08 cho WeedMap hoặc dataset đa phổ tương đương bằng giao thức được định trước; xác nhận số băng, GSD, nhãn, split và định nghĩa small-object trước khi chạy.
- Chỉ sau khi Exp07, lựa chọn dataset thứ hai, rà soát novelty và quyết định phương pháp hoàn tất mới khóa cấu hình/protocol cuối cùng. Khi khóa xong mới mở test **một lần**, không dùng test để chọn model hay hyperparameter.

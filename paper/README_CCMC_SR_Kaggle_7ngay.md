# Weed-Aware Spatially Weighted K/V Reduction cho SegFormer MiT-B0 trên ảnh UAV đa phổ

> Tên làm việc: **WeedAware-SR**. Phát triển từ LiteSent (SegFormer MiT-B0 5 băng + LowLevelRefine).
> **Phiên bản 2.1.** Trạng thái: **kế hoạch nghiên cứu, chưa có kết quả nào của phương pháp đề xuất.** Mọi mục `[ ]` chưa thực hiện.

## 0. Thay đổi so với phiên bản 1

| Nội dung | v1 | v2 |
|---|---|---|
| Giả thuyết | Một giả thuyết H1 gộp | Tách thành H1 (mất thông tin ở SR), H2 (prior Weed hơn prior thực vật), H3 (prior học được tái tạo oracle) |
| Công thức | `K,V = SR(X ⊙ ŵ)` | Thêm hệ số `λ` khởi tạo 0 (khởi đầu trùng baseline) và giới hạn `ŵ` |
| Prior head | Không nêu rõ vị trí | Một head ở stage 1, resize cho các stage sau (ablation riêng cho stage 2–3) |
| Oracle | Mỗi oracle dùng checkpoint riêng | Can thiệp lúc suy luận trên cùng checkpoint, **kèm** ma trận 2×2 và mô hình ceiling để tránh thiên lệch |
| K2 | Hoán vị prior toàn ảnh | Hoán vị **trong từng cửa sổ SR** |
| G1 | Chỉ sửa `sr_ratio` | Có chuyển đổi trọng số SR, thêm probe thông tin |
| Quyết định | Chưa nói rõ | Mọi cổng quyết định và chọn siêu tham số trên **validation**; test chỉ chạy một lần khi đã khóa cấu hình |
| Tuyên bố FLOPs | "FLOPs thấp hơn rõ rệt" | Chỉ so với LiteSent; so với baseline không LLR, FLOPs **không giảm** |
| Related work | 6 tài liệu | Bảng đối chiếu có cột trạng thái xác minh |
| Tài liệu tham khảo | Có tên tác giả viết theo trí nhớ | Chỉ giữ thông tin đã xác minh; các mục khác ghi rõ "cần xác minh" |

**v2 → v2.1** (sau góp ý về tính nhất quán toán học và protocol):

| Nội dung | v2 | v2.1 |
|---|---|---|
| Giới hạn `c` | Clip `ŵ ≤ c` | Bỏ ở pilot (cách A). Sau clip, trung bình cửa sổ không còn bằng 1; nếu cần clip phải chuẩn hóa lại (cách B) |
| `lambda` | Khởi tạo đúng 0 | `lam = sigmoid(a)` trong [0,1], khởi tạo nhỏ dương; prior head khởi tạo đồng đều để vẫn tương đương baseline; tránh gradient bằng 0 qua nhánh SR |
| D1 (quét `R`) | Bằng chứng cho G1 | Hạ xuống thí nghiệm thăm dò; G1 dựa chủ yếu vào D0 và D2 |
| D2 (probe) | So probe trước/sau SR chưa rõ | Định nghĩa lại: pooling cùng chiều `C`, cùng probe, AUROC/AP; gọi là linear decodability |
| Kích thước feature map | Chưa nêu | Bảng cho ảnh 600 × 600, lưu ý SR bỏ hàng/cột cuối, kiểm tra shape |
| FLOPs và latency | Chỉ nêu công cụ | Quy ước đo thống nhất, phân biệt MACs/FLOPs |
| G2 | Cần learned head | Bước 1 dùng mô hình ceiling C2/C3 (không cần learned head) |
| Bảng quyết định | Chưa có | Thêm bảng theo kết quả K1–K3 |

## 1. Bối cảnh

LiteSent đạt **85,49 ± 0,30 % mIoU** (test WeedsGalore, 5 seeds), 3,66 M tham số, 17,09 GFLOPs, 15,97 ms/ảnh trên Tesla T4. Ảnh đầu vào 600 × 600.

| Cấu hình | mIoU | Crop IoU | Weed IoU | Params | FLOPs | Nguồn |
|---|---|---|---|---|---|---|
| LiteSent (fusion + LLR) | 85,49 ± 0,30 | 77,78 ± 0,62 | 80,06 ± 0,45 | 3,66 M | 17,09 G | Bảng 3, 5 (bài báo) |
| Không LLR (có multi-scale fusion) | 83,53 ± 0,40 | 75,39 ± 0,95 | 76,89 ± 0,31 | — | — | Bảng 3 (n = 5) |
| SegFormer MiT-B0 5 băng, không LLR | 82,16 | — | — | 3,46 M | 3,49 G | Bảng 5 |

**Việc cần làm trước:** hai dòng "không LLR" cho kết quả khác nhau 1,37 điểm mIoU. Cần xác nhận hai cấu hình có giống nhau không (decoder, số seed, pipeline). Trong README này tạm dùng **83,53 ± 0,40** làm tham chiếu L0 vì có 5 seeds; khoảng cách cần thu hẹp là **1,96 điểm mIoU** và **3,17 điểm Weed IoU**.

Hạn chế khi hướng tới Q1:

- LLR về bản chất là đường tắt đặc trưng mức thấp (cùng họ với DeepLabV3+, FG-UNet), không đủ làm đóng góp chính.
- LLR làm FLOPs tăng từ 3,49 G lên 17,09 G.
- Chưa có chứng minh *vì sao* MiT-B0 bỏ sót cỏ dại nhỏ.

## 2. Giả thuyết

Efficient Self-Attention của MiT-B0 nén K/V bằng conv có kernel và stride bằng `R` (`R = 8, 4, 2, 1` ở 4 stage). Phép nén đồng đều này có thể làm tín hiệu cỏ nhỏ bị trộn với nền chiếm ưu thế (Background IoU 98,63 %).

| ID | Giả thuyết | Bác bỏ nếu |
|---|---|---|
| **H1** | Cá thể Weed nhỏ bị suy giảm tín hiệu ở SR các stage đầu nhiều hơn cá thể lớn | Không có chênh lệch theo kích thước, hoặc giảm `R` không cải thiện Weed recall nhóm nhỏ |
| **H2** | Prior Weed bảo toàn thông tin có ích cho Weed tốt hơn prior thực vật (Crop ∪ Weed) | GT-Weed không vượt GT-Veg ở Small Weed Recall |
| **H3** | Prior Weed **dự đoán được** tái tạo phần đáng kể lợi ích của oracle mà không cần GT | Hệ số khôi phục HR dưới ngưỡng đã đăng ký |

Nếu H1 không có bằng chứng thì không kết luận SR là nguyên nhân chính. Nếu H2 không đạt thì dừng hướng Weed-specific.

## 3. Phương pháp đề xuất

**Đối tượng:** lớp Spatial Reduction (SR) trong Efficient Self-Attention của MiT-B0, **stage 1–3**; stage 4 (`R = 1`) giữ nguyên.

```
5 băng -> patch embedding -> F1 (H/4)
                               |
                     Weed prior head (có giám sát)
                               |
                         p = P(Weed), soft
                               |
          resize theo stage + chuẩn hóa theo cửa sổ R x R
                               |
   Feature (stage i) -> điều biến -> SR conv pretrained -> LayerNorm -> K/V
```

Công thức cho stage `i` (cửa sổ `R_i x R_i`; `AvgPool_R` dùng đúng kernel và stride của SR conv, không padding):

```
p      = sigmoid( g(F1) )                       # prior Weed, resize về độ phân giải stage i
w      = eps + (1 - eps) * p                    # sàn eps bảo vệ vùng prior thấp
w_hat  = w / (AvgPool_R(w) + delta)             # chuẩn hóa theo cửa sổ (pilot: không clip)
lam_i  = sigmoid(a_i)                           # lambda_i trong [0,1], khởi tạo nhỏ (ví dụ 0,05)
X'     = X * ( 1 + lam_i * (w_hat - 1) )
Z      = SRConv_i(X')                           # trọng số pretrained giữ nguyên
```

Tính chất cần nhớ:

- **Khởi tạo tương đương baseline:** lớp cuối của prior head khởi tạo để `p` đồng đều, suy ra `w_hat ≈ 1` và `X' = X` với mọi `lam`. Nhờ vậy dùng được `lam` khởi tạo **dương nhỏ**, và prior head nhận gradient từ segmentation loss ngay từ đầu. Nếu để `lam = 0` đúng thì đạo hàm theo `w_hat` bằng 0: head chỉ học từ aux loss ở bước đầu, còn cấu hình không có aux loss (K3) bị kẹt ở điểm yên ngựa. `lam = 0` chỉ dùng cho unit test (đầu ra phải trùng baseline).
- Chuẩn hóa theo cửa sổ làm trung bình của `w_hat` trong cửa sổ xấp xỉ 1: cơ chế chỉ **phân phối lại đóng góp bên trong cửa sổ**, không loại cả cửa sổ nền. Cửa sổ có prior thấp đều nhau thì `w_hat ≈ 1`, giữ nguyên ngữ cảnh Background.
- Một pixel Weed (`p = 1`) trong cửa sổ 8 × 8 với `eps = 0,1` cho `w_hat ≈ 8,77`, khá lớn nếu `lam` cao; vì vậy `lam` bị chặn trong [0, 1] và phải báo cáo giá trị `lam_i` sau huấn luyện của từng stage.
- **Về giới hạn trên `c`:** chuẩn hóa chỉ giữ trung bình `w_hat` bằng 1 **trước** khi clip. Sau `min(·, c)` tính chất này mất, cửa sổ có thể giảm đóng góp tổng thể thay vì chỉ phân phối lại. Bản pilot **bỏ clip** (cách A). Nếu về sau cần clip (cách B), phải chuẩn hóa lại sau clip (hoặc chiếu lên tập trọng số bị chặn có tổng cố định) và kiểm tra giới hạn trên còn đúng.
- **Không phải token pruning:** số K/V tokens không đổi. Tên mô tả chính xác: *Weed-Aware Spatially Weighted K/V Reduction*.
- **Rủi ro cơ chế:** SR vẫn là ánh xạ tuyến tính `R²·C -> C`. Điều biến chỉ đổi **ưu tiên**, không tăng dung lượng. Phải chứng minh bằng đo trực tiếp (probe ở 6.1), không chỉ bằng mIoU.
- **Rủi ro prior:** head dùng đặc trưng stage 1 (mức thấp) nên khó phân biệt Crop và Weed. Nếu G3 không đạt, thử head cho stage 2–3 lấy đầu vào từ đầu ra stage trước (ablation 6.5).
- **Hiệu quả:** số K/V tokens và `R` không đổi, nên FLOPs của MiT **không giảm** so với baseline không LLR (head và phép nhân thêm một lượng nhỏ). Lợi ích FLOPs chỉ có khi **thay LLR** (17,09 G) bằng cơ chế này.

### 3.1 Ghi chú triển khai (ảnh 600 × 600)

Kích thước tính từ tham số conv của MiT-B0 (patch embedding 7×7/stride 4/pad 3, các stage sau 3×3/stride 2/pad 1; SR conv kernel = stride = `R`, không padding). **Cần kiểm lại bằng forward thật.**

| Stage | Feature map | `R` | K/V map |
|---|---|---|---|
| 1 | 150 × 150 | 8 | 18 × 18 |
| 2 | 75 × 75 | 4 | 18 × 18 |
| 3 | 38 × 38 | 2 | 19 × 19 |
| 4 | 19 × 19 | 1 | 19 × 19 |

- Stage 1 và 2 không chia hết (150 = 18·8 + 6; 75 = 18·4 + 3): SR conv không padding nên **bỏ hàng/cột cuối** khỏi K/V. Không hard-code `H/R, W/R`; sinh weight map theo kích thước feature map thực, dùng `AvgPool2d` cùng kernel, stride, padding và `ceil_mode=False` với SR conv, rồi `assert` shape đầu ra trùng nhau.
- Nếu huấn luyện với crop khác kích thước suy luận (ví dụ 512 và 600), kiểm tra cả hai.
- Unit test bắt buộc: (1) `lam = 0` và (2) prior đồng đều đều cho đầu ra trùng baseline; (3) backward không NaN; (4) FLOPs đúng quy ước mục 5.

## 4. Related work, research gap và giới hạn tuyên bố novelty (rà soát 02/11/2026)

> **Mức xác minh:** các nhận xét dưới đây dựa trên abstract/trang công bố chính thức và nội dung README, **chưa phải systematic review hoặc đối chiếu toàn bộ công thức trong full-text**. Không được tuyên bố “first”, “chưa ai làm”, hoặc Q1-ready ở giai đoạn này. Link/DOI để truy vết ở mục 4.6.

### 4.1. Đối chiếu với công trình gần nhất

| Paper | Đã được xác nhận từ nguồn | Trùng với ý tưởng hiện tại | Khoảng khác biệt *cần kiểm chứng* |
|---|---|---|---|
| SegFormer (NeurIPS 2021) | MiT sử dụng spatial-reduction attention, decoder đa tỉ lệ | Toán tử SR conv và backbone | Prior theo lớp tác động vào đóng góp **trong cửa sổ SR**, không thay đổi số token K/V |
| Dynamic Spatial Sparsification (TPAMI 2023; sửa mốc 2022 arXiv trong README cũ) | Dynamic importance, xử lý bất đối xứng vị trí, mở rộng sang dense prediction | Spatial importance thích nghi | Không hướng đến class-conditioned *within-SR convolution weighting*; cần kiểm toán code/toán tử |
| ALGM (CVPR 2024) | Merge token theo similarity cục bộ và toàn cục trong segmentation | Thay đổi cách giữ/gộp token | Không dùng class-conditioned weighted SR convolution; mục tiêu tăng throughput khác với bảo toàn Weed |
| DTMFormer (AAAI 2024) | Attention-guided token merging và token reconstruction cho medical segmentation | Gộp token thích nghi | Khác phép chiếu K/V SR cố định và không dùng Weed prior |
| VLTP (WACV 2025) | Token pruning có vision-language task guidance | Task-guided token importance | Prune token theo task, không tái phân phối trọng số trong cửa sổ SR; chứng minh task guidance không tự nó mới |
| MLSRS-SegFormer (MethodsX 2025) | Adaptive patch weighting, Hausdorff-bias attention, positional encoding | Adaptive spatial weighting và sửa SegFormer | **Đối thủ novelty quan trọng**: cần so toán tử adaptive patch weighting, vị trí áp dụng và cơ chế bias |
| SegMAN (CVPR 2025) | Phân tích mất fine detail do downsampled attention; dùng local attention và state-space | Cùng vấn đề mất chi tiết | Không nên claim đầu tiên phát hiện fine-detail loss trong attention; chỉ claim giả thuyết cụ thể tại SR của MiT-B0 nếu được chứng minh |
| WeedsGalore (WACV 2025) | Dataset 5-band, multitemporal, instance+semantic labels; RGB so với multispectral | Cùng dataset và tác vụ | Thêm NIR/RE không phải novelty; tận dụng instance labels để chứng minh lỗi theo kích thước |
| Dual-Branch CNN–Transformer U-Net (AECE 2026) | WeedsGalore, fusion CNN/Transformer, báo 85.22% IoU trong protocol riêng | Cùng benchmark | Không so chéo 85.22 với 85.49 nếu split/metric khác; mục tiêu sửa một toán tử MiT thay vì ghép model |
| VISA/BAWSeg (Remote Sensing 2026) | Two-stream radiance/index, spectral attention, đánh giá cross-year | Cỏ nhỏ và đa phổ | Không thể claim novelty chỉ từ spectral prior/gating; phương pháp hiện tại phải vượt đối chứng spatial/spectral gating |
| Unveiling the hidden (Smart Agricultural Technology 2026) | RGB–NIR fusion, weed segmentation trên WeedsGalore và bộ khác | Multispectral giúp cỏ dại | RGB+NIR không còn là đóng góp chính |

**Cần đọc full-text tiếp trước khi nộp:** PDAViT (2026), ASVLB-Net (2026), FG-UNet (2024), DynamicViT (2021), các nghiên cứu *class-guided attention*, *weighted spatial reduction attention*, *content-adaptive pooling* và *foreground-aware token pooling*. Không được suy ra không tồn tại paper trùng chỉ vì chưa tìm thấy ở lần rà soát này.

### 4.2. Research gap có thể bảo vệ (không phải khẳng định tuyệt đối)

**Gap A — Task-relevant information retention:** Các công trình token pruning/merging chủ yếu thích nghi **số lượng hoặc lựa chọn token**; SegFormer dùng SR convolution cố định. Chưa xác nhận trong tập paper đã rà soát một phép **tái phân phối trọng số theo class prior bên trong mỗi receptive field SR, vẫn giữ nguyên số K/V tokens**. Đây là khác biệt toán tử cần đối chiếu sâu hơn, không phải bằng chứng chắc chắn về novelty.

**Gap B — Causal mechanism:** Có nhận xét phổ biến về mất chi tiết khi downsampling, nhưng README V1 chưa đo trực tiếp lỗi cỏ nhỏ qua SR. Cần chứng minh thay đổi tại SR có ích hơn cùng prior dùng ngoài SR, và không chỉ do auxiliary supervision.

**Gap C — Failure robustness:** Prior Weed có thể sai khi Crop/Weed giống nhau. Chưa thể claim cơ chế hữu ích nếu không kiểm tra prior nhiễu, lệch vị trí, false-positive và prior gần đồng đều.

**Gap D — Deployment trade-off:** Giữ số K/V token không có nghĩa giảm FLOPs so với SegFormer thuần. Claim khả thi là cải thiện quality/compute so với LLR nặng **nếu số liệu thực nghiệm xác nhận**.

### 4.3. Ý tưởng đề xuất để code: Class-Conditioned Mass-Conserving SR (CCMC-SR)

**Chỉ là working name, không claim hoàn toàn mới.** Cải tiến trực tiếp `sr` trong Efficient Self-Attention của MiT-B0, không ghép backbone khác.

- **Prototype an toàn luồng dữ liệu:** sinh `p_weed` từ output stage 1 rồi chỉ áp dụng CCMC-SR cho stage 2–3. Stage 1 không được dùng output của chính stage 1 để điều biến attention trong stage 1 (circular dependency). Biến thể sau: prior từ patch embedding trước block attention stage 1.
- Tại stage i, `w = eps + (1-eps)*p_weed`, `mu = AvgPool2d(w, kernel=R_i, stride=R_i, padding=0)`, `w_norm = w / clamp_min(upsample_repeat(mu, R_i), eps)` **trong vùng các cửa sổ SR đầy đủ**. Phần rìa không thuộc cửa sổ giữ trọng số 1. Dùng `delta` chỉ như numerical floor nhỏ hơn `eps`, không cộng `delta` vào mẫu số nếu muốn bảo toàn mean chính xác.
- `X_mod = X * (1 + sigmoid(a_i)*(w_norm-1))`; dùng pretrained SR conv và LayerNorm cũ. Init prior head đồng đều, `sigmoid(a_i)` khoảng 0.05; không clip trong pilot. Assert mean của `w_norm` theo từng cửa sổ xấp xỉ 1 và output trùng baseline khi prior đồng đều/`lambda=0`.
- **Không nhầm mass-conservation với feature-conservation:** SR conv có kernel learned không đều, nên mean trọng số bằng 1 không đảm bảo tổng đóng góp feature hay norm K/V được bảo toàn. Đo activation norms và feature distortion.
- **Cảnh báo kích thước:** 600x600 -> stage 1 150x150 R8 và stage 2 75x75 R4: SR bỏ rìa; không repeat rồi crop một cách vô tình điều biến phần rìa; kiểm tra output `sr` thực tế.
- **Hướng nâng cấp nếu gate thường ngang CCMC-SR:** thử `dual-statistic SR` trong **cùng một block**: nhánh weighted SR và SR chuẩn dùng chung conv, sau đó học hệ số pha trộn nhỏ (tăng tính toán, cần profile); chỉ triển khai nếu thí nghiệm chứng minh lỗi phân biệt crop/weed xuất phát từ prior nhiễu. Không tự động thêm nhánh nếu prototype thất bại.

### 4.4. Bằng chứng tối thiểu phải chạy (ưu tiên theo thứ tự)

| ID | Thí nghiệm | So sánh/đo | Tiêu chí quyết định trên validation |
|---|---|---|---|
| N0 | Baseline reproducibility | MiT-B0 5-band no LLR, LiteSent LLR; 5 seeds cuối | Giải thích chênh 82.16 và 83.53 của README cũ trước khi claim gain |
| N1 | Small-weed diagnostics | Instance recall theo quantile kích thước train, stage-wise linear probes | Có dấu hiệu lỗi cỏ nhỏ liên quan SR; không coi probe là causal proof |
| N2 | Correctness | CCMC-SR stage 2–3, unit tests shape/gradient/equivalence/window mean | Tất cả test pass; không NaN; cùng pretrained weights |
| N3 | Prior specificity | GT-Weed vs GT-Veg oracle **trên validation**, cùng checkpoint + ceiling riêng | Weed-specific thực sự hơn vegetation prior; không dùng GT trong model báo cáo |
| N4 | Mechanism controls | CCMC-SR vs aux-only vs same-prior spatial gate **trước SR nhưng không window normalization**, gate **sau SR**, shuffle prior trong cửa sổ, random prior, no-aux | CCMC-SR hơn đối chứng gần nhất ở small-weed recall/Weed IoU, không đánh đổi Crop IoU quá ngưỡng đăng ký trước |
| N5 | Controlled compute | Params, MACs/FLOPs (attention matmul included), latency T4 batch1, peak VRAM | Báo cáo Pareto, không claim giảm compute so baseline B0 nếu không có |
| N6 | Robustness | prior corruption (dilate/erode, shift, false-positive, dropout), từng ngày bay | Mức suy giảm theo noise, không chỉ kết quả oracle đẹp |
| N7 | External validity | Một dataset đa phổ khác có nhãn phù hợp; PhenoBench RGB chỉ làm transfer/ablation riêng | Không gọi train độc lập trên PhenoBench là cross-dataset generalization |

**Thiết kế N4 công bằng:** tất cả đối chứng cùng backbone, input, train/val split, augmentation, optimizer, loss chính, số epoch và 5 seeds ở bảng cuối. Đối chứng cùng prior phải dùng **cùng head + cùng auxiliary loss**; thay duy nhất vị trí/toán tử gating. Report paired delta và CI bootstrap theo ô/vị trí, không chỉ mean ± std seed.

**Go/no-go:** nếu CCMC-SR không hơn same-prior gate, **không claim SR-specific novelty**; nếu GT-Weed không hơn GT-Veg, bỏ claim Weed-specific; nếu prior dự đoán không tạo gain ổn định, dừng trước khi mở rộng kiến trúc. Chỉ khóa hyperparameter bằng validation, test chính thức sau cùng.

### 4.5. Claim viết bài nếu (và chỉ nếu) N1–N7 ủng hộ

> We investigate class-conditioned, window-normalized spatial reduction in a hierarchical transformer for small-weed segmentation, and evaluate whether redistributing within-window feature contributions improves instance-level weed retention without increasing the number of reduced key/value tokens.

Đây là **research question/conditional contribution**, không phải kết luận đã được chứng minh. Không gọi phương pháp “first”, “novel” tuyệt đối hay “efficient” nếu không có evidence về toán tử và runtime.

### 4.6. Nguồn đã đối chiếu (đường dẫn công bố)

- WeedsGalore (WACV 2025): https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html
- Dynamic Spatial Sparsification (TPAMI 2023): https://pubmed.ncbi.nlm.nih.gov/37030709/ ; DOI 10.1109/TPAMI.2023.3263826
- ALGM (CVPR 2024): https://www.tue-mps.org/ALGM/
- DTMFormer (AAAI 2024): https://ojs.aaai.org/index.php/AAAI/article/view/28394
- VLTP (WACV 2025): https://ieeexplore.ieee.org/abstract/document/10944127
- MLSRS-SegFormer (MethodsX 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12637274/
- SegMAN (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.pdf
- Dual-Branch CNN–Transformer U-Net (2026): https://aece.ro/abstractplus.php?article=5&number=2&year=2026
- VISA / BAWSeg (Remote Sensing 2026): https://www.mdpi.com/2072-4292/18/6/915
- Unveiling the hidden (2026): https://www.sciencedirect.com/science/article/pii/S2772375526000997

## 5. Protocol

| Hạng mục | Quy định |
|---|---|
| Dữ liệu chính | WeedsGalore, split chính thức theo không gian (104/26/26), 3 lớp: Background, Crop, Weed |
| Baseline | Huấn luyện lại **toàn bộ** trong cùng protocol; không so với số liệu của bài khác (split và cách tính khác nhau) |
| Thiết lập chung | Cùng crop size, augmentation, optimizer, scheduler, số epoch, quy tắc chọn checkpoint (validation mIoU cao nhất) |
| **Chọn cấu hình** | Stage, `eps`, `c`, trọng số loss, ngưỡng cổng G1–G4: chọn **trên validation**. **Cấm** dùng test để chọn kiến trúc hoặc siêu tham số |
| **Test** | Chạy sau khi đã khóa cấu hình cuối. Không thử nhiều cấu hình trên test rồi chọn cấu hình đẹp nhất |
| Seeds | 5 seeds (0–4); báo cáo từng seed và trung bình |
| Thống kê | Bootstrap theo **vị trí/ô ruộng** (cluster) nếu metadata cho phép, vì ảnh cùng vị trí khác ngày không độc lập; nếu không thì theo ảnh. Gộp 5 seeds: báo cáo riêng phương sai giữa các seed và CI theo cluster trên dự đoán đã trung bình seed |
| Dữ liệu phụ | Một tập đa phổ khác (ví dụ WeedMap); PhenoBench (RGB) làm đối chứng |
| GT trong suy luận | **Cấm** trong mô hình báo cáo. GT chỉ dùng trong các phân tích oracle, ghi rõ riêng |

**Metric:**

- mIoU, IoU từng lớp, Boundary-IoU, ECE sau Temperature Scaling.
- **Weed recall mức cá thể theo kích thước:** dùng mask instance chính thức; bin theo tam phân vị diện tích tập train; ghép **one-to-one** (một thành phần dự đoán chỉ được tính phát hiện cho tối đa một cá thể GT); ngưỡng IoU 0,5 (ablation 0,3).
- Proxy "tiếp giáp tán cây": tỉ lệ chu vi cá thể Weed tiếp giáp mask Crop sau giãn nở. Đây là proxy, không phải nhãn che khuất.
- **Params, FLOPs, latency (quy ước thống nhất):** cùng độ phân giải, batch size, precision và phạm vi model được profile cho mọi mô hình; ghi rõ MACs hay FLOPs; tính cả phép nhân điều biến và prior head. THOP đếm theo module nên có thể bỏ sót các phép matmul của attention: cần quy tắc tùy chỉnh hoặc công cụ khác (ví dụ fvcore) và kiểm tra chéo trước khi dùng 3,49 G và 17,09 G làm bằng chứng hiệu quả. Latency: batch 1 trên Tesla T4, warm-up, đồng bộ CUDA, lặp nhiều lần, báo trung vị và độ phân tán.

## 6. Thiết kế thí nghiệm

### 6.1 Chẩn đoán (P0–P1), kiểm chứng H1

| ID | Thí nghiệm | Mục đích |
|---|---|---|
| D0 | Weed recall theo bin kích thước, ma trận nhầm lẫn, lỗi theo ngày bay, trên checkpoint LiteSent và baseline không LLR | Cỏ nhỏ có bị bỏ sót nhiều hơn không |
| D1 | Quét `R` ở stage 1–2 (ví dụ 8→4→2): nội suy kernel SR, bảo toàn tổng trọng số, fine-tune cùng protocol | **Thăm dò (exploratory), không phải bằng chứng nhân quả.** Đổi `R` đồng thời đổi kích thước kernel, số K/V tokens, phạm vi gom và biểu diễn pretrained, rồi còn fine-tune |
| D2 | Linear decodability probe (protocol bên dưới) | Cỏ nhỏ có khó giải mã tuyến tính hơn sau SR không |

**Protocol D2:**

1. Nhãn theo cửa sổ SR của từng stage: dương = cửa sổ chứa Weed thuộc bin nhỏ; âm = cửa sổ không có Weed (biến thể: âm gồm cả cửa sổ chỉ có Weed lớn).
2. Đặc trưng cùng chiều `C`: (a) trước SR, average pooling trong cửa sổ; (b) sau SR, vector tại đúng cửa sổ đó; (c) đối chứng: max pooling trước SR.
3. Cùng một cấu hình linear probe; huấn luyện trên train, đánh giá trên validation.
4. Metric: AUROC, Average Precision, recall tại precision cố định.
5. Diễn giải: đây là phân tích **khả năng giải mã tuyến tính**, không phải phép đo tuyệt đối lượng thông tin mất. Pooling trước SR cũng là một phép nén nên (b) so với (a)/(c) chỉ mang tính tương đối; so sánh sạch nhất là cùng protocol giữa L0 và L3, và giữa các bin kích thước.

### 6.2 Các mô hình huấn luyện (bậc thang prior)

| ID | Prior | Huấn luyện | Đánh giá | GT lúc test |
|---|---|---|---|---|
| L0 | Đồng đều | Segmentation loss | Đồng đều | Không |
| L1 | NDVI cố định | Segmentation loss | NDVI | Không |
| L2 | Vegetation học được | Aux loss, nhãn Crop ∪ Weed | Dự đoán | Không |
| **L3** | **Weed học được** | Aux loss, nhãn Weed | Dự đoán | Không |

### 6.3 Can thiệp oracle (chỉ chẩn đoán, kiểm chứng H2 và H3)

Thay prior dự đoán bằng GT **lúc suy luận** trên chính checkpoint đã huấn luyện, nên khác biệt chỉ đến từ loại prior, không đến từ trọng số khác nhau.

Mỗi checkpoint chỉ từng thấy prior cùng loại với lúc huấn luyện (L3 chưa từng thấy prior thực vật dày), nên can thiệp trên một checkpoint duy nhất **có thể thiên lệch** về phía loại prior đó. Vì vậy báo cáo cả ma trận 2×2 và mô hình ceiling:

| Checkpoint \ Prior can thiệp | Dự đoán | GT-Veg | GT-Weed | GT-Weed + nhiễu |
|---|---|---|---|---|
| L2 | | | | |
| L3 | | | | |

| ID | Mô hình ceiling (huấn luyện với GT prior, **chỉ chẩn đoán**) |
|---|---|
| C2 | Huấn luyện và đánh giá với GT-Veg |
| C3 | Huấn luyện và đánh giá với GT-Weed |

Diễn giải H2: GT-Weed vượt GT-Veg ở cả hai hàng (L2 và L3) và C3 vượt C2 thì ủng hộ mạnh. Nếu chỉ thắng ở hàng L3 thì có thể là hiệu ứng thiên lệch, cần dựa vào C2 so C3. Nhiễu cho GT-Weed gồm xói mòn, giãn nở, xóa ngẫu nhiên cá thể.

### 6.4 Đối chứng cơ chế

| ID | Cấu hình | Điều cần chứng minh |
|---|---|---|
| K1 | Head và aux loss giữ nguyên, **không nối** vào SR | Lợi ích không chỉ do auxiliary supervision |
| K2 | Prior bị hoán vị **trong từng cửa sổ SR** (giữ trung bình và phân phối giá trị của cửa sổ, phá tương ứng vị trí) | Cần đúng vị trí cỏ, không chỉ cần map có cùng phân phối. K2b (tùy chọn): hoán vị toàn ảnh làm placebo yếu hơn |
| K3 | Cùng head, **chỉ nhận gradient từ segmentation loss**, không có nhãn Weed phụ; `lam` khởi tạo dương như L3 (nếu `lam = 0` thì head không học được) | "Weed-aware" khác với bất kỳ spatial attention map nào |
| K4 | L3 + LLR | Bổ trợ hay thay thế; **không bắt buộc** để chấp nhận mô hình |

### 6.5 Ablation (pilot 1 seed trên validation, chạy tuần tự, không full factorial)

| Thứ tự | Nhóm | Các mức |
|---|---|---|
| 1 | Nguồn prior stage 1 | 5 băng thô · đầu ra patch embedding |
| 2 | Nguồn prior stage 2–3 | Resize prior stage 1 · head riêng trên đầu ra stage trước |
| 3 | Stage áp dụng | 1 · 1–2 · 1–3 |
| 4 | Cơ chế | `lam` ∈ {cố định nhỏ, học được bị chặn trong [0,1] (init 0,05)}; chuẩn hóa cửa sổ có/không; sàn `eps` ∈ {0,05, 0,1, 0,3}. Clip `c` **không** nằm trong pilot; chỉ thử ở cách B (clip rồi chuẩn hóa lại) sau khi cơ chế chính hoạt động |
| 5 | Nhãn và loss | Mục tiêu: soft occupancy (average pooling mask Weed) · max-pool presence; aux loss: BCE · BCE + Dice; trọng số 0,1/0,3/1,0 |

Mặc định dùng soft occupancy; max-pool presence là đối chứng, không giả định nó tốt hơn.

### 6.6 Bảng so sánh chính (5 seeds, sau khi khóa cấu hình)

| Mô hình | mIoU | Crop IoU | Weed IoU | Small-Weed Recall | Boundary-IoU | Params | FLOPs | Latency |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B0 5 băng, không LLR (L0) | | | | | | | | |
| LiteSent (B0 + LLR) | 85,49 ± 0,30* | 77,78* | 80,06* | | | 3,66 M | 17,09 G | 15,97 ms |
| WeedAware-SR L3, không LLR | | | | | | | | |
| WeedAware-SR L3 + LLR (K4) | | | | | | | | |
| ResNet50–DeepLabV3+ | | | | | | | | |
| SegFormer-B1/B2 (tham chiếu dung lượng) | | | | | | | | |

\* Số liệu từ bài báo; phải **chạy lại trong cùng pipeline** trước khi dùng để so sánh.

**Hai kết quả có thể xảy ra, cần đăng ký trước:** (a) L3 không LLR đạt mIoU ≥ LiteSent ở FLOPs gần 3,5 G thì claim hiệu quả; (b) L3 không đạt mức LiteSent nhưng L3 + LLR vượt LiteSent thì claim bổ trợ. Cả hai đều chấp nhận được nếu giả thuyết H1–H3 được ủng hộ.

### 6.7 Phân tích thời điểm bay (bổ trợ, không phải đóng góp chính)

| ID | Protocol | Vai trò |
|---|---|---|
| T1 | Phân tầng test chính thức theo ngày bay | Hiệu quả có đổi theo giai đoạn sinh trưởng không. Đây là phân tích phân tầng, **không** phải kiểm tra tổng quát hóa thời gian |
| T2 | Hold-out ngày cuối | Chỉ làm phân tích giới hạn: ít ảnh và có thể nhiễu bởi herbicide |
| T3 | Recall theo kích thước trong từng ngày | Prior có ích nhất ở nhóm nào. Dự đoán kiểm chứng: lợi ích tập trung ở cỏ nhỏ, ngày đầu |

## 7. Cổng quyết định (đăng ký trước, quyết định trên validation)

| Cổng | Câu hỏi | Điều kiện đi tiếp | Nếu không đạt |
|---|---|---|---|
| G1 | SR có phải nút thắt cho cỏ nhỏ? | Dựa chủ yếu vào **D0 và D2**: nhóm nhỏ suy giảm rõ ở D0 và khó giải mã hơn sau SR ở D2. D1 chỉ hỗ trợ, không bắt buộc | Không giả định SR là nguyên nhân chính |
| G2 | GT-Weed có hơn GT-Veg? | Bước 1 (rẻ, chưa cần learned head): C3 vượt C2 và L0 ở Small-Weed Recall (pilot 1 seed, validation). Bước 2: xác nhận bằng ma trận can thiệp 6.3. CI bootstrap ủng hộ, Crop IoU không giảm quá ngưỡng đặt trước | Bỏ giả thuyết Weed-specific |
| G3 | Prior học được có đủ tốt? | `HR = (M_L3 − M_L0) / (M_O3 − M_L0) ≥ 50 %` (chỉ tính khi mẫu số > 0), `M` là Small-Weed Recall | Cải thiện prior head hoặc dừng |
| G4 | Lợi ích đến từ việc nối prior vào SR? | L3 vượt K1, K2, K3 ở Small-Weed Recall, không đánh đổi nghiêm trọng mIoU/Crop IoU | Chưa chứng minh được cơ chế |

Ngưỡng 50 % là tiêu chí của dự án, không phải ngưỡng khoa học đã được công nhận; phải ghi cố định trước khi xem kết quả validation. mIoU không tăng ở G2 vẫn chấp nhận được nếu Small-Weed Recall tăng rõ mà Crop IoU không hỏng. Ngược lại, mIoU tăng nhẹ mà Small-Weed Recall không tăng thì **chưa** chứng minh giả thuyết trung tâm.

**Bảng quyết định theo kết quả:**

| Kết quả | Quyết định |
|---|---|
| D0/D2 không ủng hộ H1 | Không tiếp tục claim SR là bottleneck |
| C3 không hơn C2 (GT-Weed không hơn GT-Veg) | Dừng hướng Weed-specific |
| Ceiling tốt nhưng prior học được kém (HR thấp) | Nghiên cứu lại prior head |
| L3 không hơn K1 | Chưa chứng minh tác dụng của việc nối prior vào SR |
| L3 hơn K1 nhưng không hơn K2 | Chưa chứng minh cần đúng vị trí Weed |
| L3 hơn K1, K2 nhưng không hơn K3 | Chưa chứng minh "Weed-aware" khác content-adaptive |
| L3 hơn K1, K2, K3 | Đủ cơ sở chuyển sang ablation và 5 seeds |

## 8. Lộ trình

**Deadline cứng: 7 ngày lịch** để có gói kết quả validation và quyết định go/no-go. GPU-hours thực tế phụ thuộc quota còn lại, GPU được cấp và thời gian mỗi run; không giả định có sẵn 30 giờ hay 10× quota. Làm P0–P3 trước, chỉ đầu tư K1–K4 và 5 seeds khi các cổng đạt. Xem lịch chi tiết tại mục 8.1.

| Giai đoạn | Công việc | Kết quả cần có |
|---|---|---|
| [ ] P0 | Chẩn đoán checkpoint hiện có (D0); xác nhận hai baseline không-LLR | Recall theo bin, lỗi theo ngày, ma trận nhầm lẫn, baseline L0 chốt |
| [ ] P1 | D1 và D2 (quét `R` có chuyển trọng số, probe) | Bằng chứng hoặc bác bỏ H1, cổng G1 |
| [ ] P2 | Cài `WeightedSR` (có thể viết song song với P1); unit test mục 3.1 | Shape đúng ở các kích thước ảnh; `lam = 0` và prior đồng đều trùng baseline; backward không NaN; FLOPs đúng quy ước |
| [ ] P3a | L0 và mô hình ceiling C2, C3 (GT prior, 1 seed, validation) | Cổng G2 bước 1, chưa cần learned head |
| [ ] P3b | L1–L3 và can thiệp oracle 6.3 | Cổng G2 bước 2, G3 |
| [ ] P4 | K1–K4 và ablation 6.5 | Cổng G4 |
| [ ] P5 | Khóa cấu hình, 5 seeds, bootstrap, bảng 6.6 | Kết quả chính, CI, hiệu quả tính toán |
| [ ] P6 | Phân tích ngày bay (6.7) và tập dữ liệu thứ hai | Bằng chứng bổ trợ về độ bền |
| [ ] P7 | Đọc full paper các mục "cần đọc"; viết bài | Related work, đóng góp, giới hạn |

**Thứ tự code:** bắt đầu bằng 4 notebook độc lập (P0, P1, P2, P3) rồi mới tách thành repo. Không huấn luyện L3 (P3b) trước khi có kết quả P0–P1 và P3a. Mức sẵn sàng hiện tại: P0–P2 có thể bắt đầu; chưa chạy toàn bộ thí nghiệm trước khi qua G1 và G2.

### 8.1. Sprint Kaggle bắt buộc 7 ngày (D1–D7)

**Mục tiêu bắt buộc đến cuối D7:** (i) baseline có thể tái hiện; (ii) CCMC-SR stage 2–3 chạy đúng và có checkpoint; (iii) ít nhất một so sánh **paired** với same-prior gate trên validation; (iv) bảng Weed IoU, Crop IoU, mIoU, Small-Weed Recall, thời gian suy luận; (v) quyết định **go/no-go** kèm hạn chế. **Không hứa hoàn thành 5 seeds cho mọi mô hình, external validation hay bài Q1 trong 7 ngày.** Nếu gate không đạt, kết quả âm tính + báo cáo cơ chế là đầu ra hợp lệ.

**Quy tắc Kaggle và 10 tài khoản:** 10 tài khoản **không đồng nghĩa** 10 GPU song song hoặc được phép cộng dồn quota. Chỉ dùng tài khoản do từng người sở hữu, có quyền sử dụng, tuân thủ điều khoản Kaggle; **không luân phiên tài khoản để né hạn mức/biện pháp kiểm soát**, không chia sẻ mật khẩu, không dùng tài khoản phụ nhằm vượt quota. Có thể chia nhóm theo **người phụ trách độc lập** (code, phân tích, kiểm thử, huấn luyện khi được cấp tài nguyên hợp lệ). Nếu tài nguyên không đủ, giảm phạm vi theo thứ tự ưu tiên bên dưới, hoặc dùng GPU được cấp hợp lệ từ nguồn khác. Quota/loại GPU/thời lượng phiên thay đổi theo tài khoản và thời điểm, phải kiểm tra trong Kaggle trước khi lên lịch.

| Ngày | Việc bắt buộc và thứ tự ưu tiên | GPU / công việc song song | Deliverable cuối ngày; cổng dừng |
|---|---|---|---|
| **D1** | Chốt official split, preprocessing 5 bands, metric instance/bin, seed; chạy baseline L0 1 seed hoặc xác minh checkpoint cùng protocol; kiểm toán sai lệch 82,16 vs 83,53 | 1 phiên GPU nếu cần; song song CPU: đọc full paper gần nhất, kiểm tra leakage, viết test metric | `configs/baseline.yaml`, `splits_hash.json`, `results/d1_baseline.json`, log môi trường; **không sang D3 nếu metric sai** |
| **D2** | D0 Small-Weed Recall, Crop/Weed IoU, theo ngày; D2 probe tối giản nếu đủ thời gian; cài `WeightedSR` và unit tests | GPU cho inference/probe; CPU viết operator + synthetic tests song song | `results/d2_diagnostics.csv`, unit tests shape/gradient/identity/window mean; nếu D0 không ủng hộ, ghi **G1 fail**, chỉ chạy pilot phản chứng |
| **D3** | Tích hợp CCMC-SR stage 2–3; chạy smoke test 1 batch → 1 epoch → pilot L3 1 seed; cùng lúc chạy K1 aux-only nếu GPU cho phép | GPU ưu tiên L3; CPU review code, loss, profiler | `checkpoints/l3_seed0.pt`, learning curves, no-NaN, equivalence test; **fail correctness → dừng train** |
| **D4** | Chạy **same-prior spatial gate** với cùng prior head/aux loss/seed/split/epoch; chạy oracle C2/C3 pilot chỉ nếu còn ngân sách | GPU ưu tiên đối chứng gate; CPU viết bootstrap, phân tích prior | `results/d4_l3_vs_gate.csv`, checkpoint gate, báo cáo G2 sơ bộ; nếu gate ngang/vượt L3, **không claim SR-specific novelty** |
| **D5** | Chạy K2 within-window shuffle, K3 no-aux hoặc prior random (chọn tối thiểu 1 placebo); corruption thử nghiệm trên validation; profile latency batch 1 | GPU 1–2 pilot ngắn tùy quota; CPU dựng bảng paired delta | `results/d5_mechanism.csv`, `profile_t4.json`, bảng lỗi; **G3/G4 không đạt → không mở ablation rộng** |
| **D6** | Nếu G1–G4 có tín hiệu: lặp **L0/L3/gate với seed thứ 2**, ưu tiên cùng seed giữa các model; nếu quota thiếu: chỉ chạy L3 vs gate seed 2; đóng băng config | GPU ưu tiên xác nhận paired; CPU bootstrap theo image/cluster và audit | `results/d6_paired.csv`, config frozen, bảng mean/delta sơ bộ; không lấy test để chọn mô hình |
| **D7** | Đánh giá validation thống nhất, kiểm tra crop/weed trade-off, tổng hợp biểu đồ, chạy test chính thức **một lần duy nhất nếu đã khóa mô hình**; viết báo cáo quyết định | GPU inference/test + profiling khi hợp lệ; CPU viết report và đóng gói notebook | `results/summary_7days.md`, `results/metrics.csv`, `results/decision.json`, `README` cập nhật, notebook reproducible; danh sách việc còn thiếu cho paper |

**Ước lượng công suất (phải benchmark D1, không phải cam kết):** ký hiệu `t` = thời gian train **một** model/seed đến budget epoch đã khóa, `q` = tổng GPU-giờ **thực sự được cấp hợp lệ** trong 7 ngày. Kịch bản tối thiểu gồm L0, L3, same-prior gate và 1 placebo: khoảng `4t + t_diagnostics + t_profile` GPU-giờ. Kịch bản xác nhận 2 seeds cho L0/L3/gate: thêm khoảng `3t`. Năm seeds của 3 model cần xấp xỉ `15t` (chưa tính ablation), **chỉ thực hiện nếu** `q` đủ. D1 chạy benchmark 5–10 epoch, đo thời gian/epoch và ghi `ETA = epochs_remaining × median_epoch_time`; thêm biên dự phòng 20–30% cho startup, checkpoint, lỗi phiên. Không tự suy ra GPU-hours từ số tài khoản.

**Điều phối phiên và checkpoint:** mỗi job có `run_id = model__seed__split_hash__git_commit`; đặt seed Python/NumPy/PyTorch và log versions; lưu checkpoint **theo epoch** hoặc mỗi 10–15 phút, gồm model/optimizer/scheduler/scaler/RNG/epoch/config, và xuất ra Kaggle Dataset/Artifact hoặc kho được phép truy cập để có thể resume giữa các phiên hợp lệ. Không phụ thuộc vào filesystem tạm của notebook; kiểm tra checkpoint đọc lại được ngay D1. Một job chỉ có một người/phiên ghi vào output để tránh overwrite. Ghi `manifest.csv`: run_id, chủ sở hữu, notebook version, GPU, start/end, wall time, GPU time, seed, trạng thái, đường dẫn artifact, hash. Không trộn checkpoints của các split/config khác nhau.

**Chia việc cho nhóm có tối đa 10 người/tài khoản hợp lệ (không yêu cầu 10 GPU):** nhóm A (2 người) data/metrics + split audit; B (2) WeightedSR + tests; C (2) training/baseline + logging; D (2) gate/oracle + controls; E (2) profiling, thống kê, related work và tổng hợp. Một người có thể kiêm nhiều vai nếu nhóm nhỏ hơn. Khi chỉ có **1 GPU khả dụng**, hàng đợi ưu tiên: **L0 → L3 → same-prior gate → K1/K2 → seed 2 → oracle → ablation → seed 3–5 → external**. Các job CPU triển khai song song, còn GPU chỉ chạy những job không xung đột tài nguyên.

**Định nghĩa hoàn thành 7 ngày:** phân biệt `DONE` (đã chạy và có artifact), `PARTIAL` (chưa đủ seed/CI) và `NOT RUN` (thiếu quota/thời gian). Không đưa số liệu từ pilot 1 seed thành bằng chứng kết luận Q1; không báo cáo test lặp lại để chọn hyperparameter. Sau D7, nếu kết quả tốt, tiếp tục 5 seeds, external dataset, robust checks và hoàn thiện related work trong giai đoạn tiếp theo.

## 9. Cấu trúc repo (dự kiến)

Giai đoạn đầu làm bằng notebook; chỉ tách thành repo sau khi P2 hoàn tất.

```
.
├── configs/                 # L0-L3, K1-K4, ablation
├── data/
│   ├── weedsgalore/         # dataset + split chính thức
│   └── instance_bins.json   # bin kích thước (tam phân vị train)
├── src/
│   ├── models/
│   │   ├── mit_weighted_sr.py    # Efficient Self-Attention với prior_mode
│   │   ├── weed_prior_head.py
│   │   └── litesent.py           # baseline B0 + LLR
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── metrics/
│       ├── instance_recall.py    # one-to-one matching, theo bin
│       ├── boundary_iou.py
│       └── ece.py
├── scripts/
│   ├── diagnose.py          # D0
│   ├── sweep_sr_ratio.py    # D1 (có chuyển đổi trọng số SR)
│   ├── probe_sr_info.py     # D2
│   ├── oracle_intervention.py  # 6.3
│   └── bootstrap_ci.py
└── results/
```

`prior_mode ∈ {uniform, ndvi, learned_veg, learned_weed, gt_veg, gt_weed, gt_weed_noisy, permute_window}`; các chế độ `gt_*` chỉ được bật trong `scripts/oracle_intervention.py`.

## 10. Môi trường

- Kaggle Notebook, NVIDIA Tesla T4, CUDA 12.8, PyTorch 2.10.0+cu128.
- Khởi tạo: `nvidia/mit-b0`. Ảnh 600 × 600.
- Adam, lr 3e-4, batch size 8, tối đa 100 epoch; `L = 0,7·CE (trọng số lớp 0,9/1,0/1,3) + 0,3·SoftDice`; augmentation: xoay, lật, nhiễu Gaussian.

## 11. Rủi ro

- Giả thuyết H1–H3 chưa được kiểm chứng; nếu G1 hoặc G2 không đạt thì hướng này bị bác bỏ. Phương án dự phòng: stem phổ thích nghi, hoặc mã hóa hướng phổ bất biến theo cường độ sáng.
- Điều biến feature trước SR có thể không bảo toàn thêm thông tin vì dung lượng SR không đổi; D2 và K1–K3 kiểm tra điểm này.
- Prior từ đặc trưng stage 1 có thể quá yếu để phân biệt Crop và Weed.
- Prior sai có thể tự củng cố lỗi (cỏ bị bỏ sót thì bị giữ ít hơn): kiểm soát bằng `eps`, `c` và can thiệp GT-Weed nhiễu.
- Test chỉ có 26 ảnh, một ruộng, một mùa: bắt buộc bootstrap theo cluster và tập dữ liệu thứ hai.
- Ngày cuối ít ảnh và có thể bị nhiễu bởi herbicide.
- Con số FLOPs 3,49 G và 17,09 G có thể phụ thuộc cách đếm (THOP bỏ sót matmul của attention); phải đo lại cùng điều kiện trước khi dùng làm bằng chứng hiệu quả.
- SR không padding bỏ hàng/cột cuối ở stage 1 và 2, nên vùng sát biên không đóng góp vào K/V ở cả baseline lẫn mô hình đề xuất; ghi nhận khi phân tích lỗi gần biên ảnh.
- Tính mới chưa được xác nhận: cần đọc full paper Dynamic Spatial Sparsification, ALGM, VLTP.

## 12. Tài liệu tham khảo

**Đã xác minh hoặc biết chắc:**

1. Celikkan et al., *WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields*, WACV 2025 (arXiv:2502.13103).
2. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
3. Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, ECCV 2018.
4. Rao et al., *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification*, NeurIPS 2021.

**Theo rà soát của nhóm, cần xác minh tên tác giả, venue và nội dung trước khi trích dẫn:** Dynamic Spatial Sparsification (2022); ALGM (CVPR 2024); VLTP (WACV 2025); PDAViT (Neurocomputing 2026); ASVLB-Net (Pest Management Science 2026); VISA / BAWSeg (Remote Sensing 2026); FG-UNet (Pest Management Science 2024); Multispectral Feature Decomposition via Dual-Branch CNN–Transformer U-Net (2026).

## 13. Nhóm thực hiện

Nhóm 6 — Khoa Công nghệ Thông tin, HUIT.

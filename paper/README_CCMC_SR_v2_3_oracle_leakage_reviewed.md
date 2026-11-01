# CCMC-SR: Class-Conditioned Mass-Conserving Spatial Reduction cho SegFormer MiT-B0 trên ảnh UAV đa phổ

> Tên thống nhất: **CCMC-SR** (tên cũ/alias: WeedAware-SR). Phát triển từ LiteSent (SegFormer MiT-B0 5 băng + LowLevelRefine).
> **Phiên bản 2.2 (02/11/2026).** Trạng thái: **kế hoạch nghiên cứu, chưa có kết quả nào của phương pháp đề xuất.** Mọi mục `[ ]` chưa thực hiện.

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
| D1 (quét `R`) | Bằng chứng cho G1 | Hạ xuống thí nghiệm thăm dò; G1 dựa chủ yếu vào Diag-D0 và Diag-D2 |
| D2 (probe) | So probe trước/sau SR chưa rõ | Định nghĩa lại: pooling cùng chiều `C`, cùng probe, AUROC/AP; gọi là linear decodability |
| Kích thước feature map | Chưa nêu | Bảng cho ảnh 600 × 600, lưu ý SR bỏ hàng/cột cuối, kiểm tra shape |
| FLOPs và latency | Chỉ nêu công cụ | Quy ước đo thống nhất, phân biệt MACs/FLOPs |
| G2 | Cần learned head | Bước 1 dùng mô hình ceiling C2/C3 (không cần learned head) |
| Bảng quyết định | Chưa có | Thêm bảng theo kết quả K1–K3 |

**Changelog v2.2:** thống nhất prior output stage 1 → stage 2–3; sửa chuẩn hóa chính xác; thêm đặc tả HuggingFace, K5/K6, ánh xạ ID, thứ tự oracle trước L3, metric bin/seed và sprint 7 ngày nhất quán. Các thiết kế mới vẫn là đề xuất chưa kiểm thử.

**Changelog v2.3 (03/11/2026):** kiểm toán leakage oracle; G2 chuyển thành gate loại bỏ một chiều; H2 kiểm bằng L3–L2; thêm C3-K5/C3-K6 và paired interaction; HR oracle chỉ báo cáo thăm dò; đặc tả hook HF có điều kiện xác minh mã pin; thống nhất ID/tên và timeline. Không có kết quả thực nghiệm mới.

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
| **H1** | Cá thể Weed nhỏ có mức suy giảm khả năng giải mã tuyến tính qua SR cao hơn cá thể lớn ở ít nhất một stage (D2 đo cả stage 1–3) | Không có chênh lệch theo kích thước, hoặc giảm `R` không cải thiện Weed recall nhóm nhỏ |
| **H2** | **Prior Weed dự đoán được** mang lại lợi ích hơn prior Vegetation dự đoán được khi giữ nguyên SR, kiến trúc và ngân sách train | L3 không vượt L2 theo paired seeds/ảnh; C3/C2 chỉ dùng sàng lọc vì chứa GT leakage |
| **H3** | Prior Weed **dự đoán được** đem lại hiệu ứng có ý nghĩa và ổn định, không chỉ hưởng lợi từ aux supervision hoặc spatial gate | L3 không vượt L0/K1/K5/K6 với CI phù hợp; HR oracle chỉ là mô tả phụ, không dùng làm gate |

Nếu H1 không có bằng chứng thì không kết luận SR là nguyên nhân chính. **H2 quyết định bằng L3 vs L2, không quyết định bằng C3 vs C2.** C3/C2 chứa thông tin nhãn thật ở validation nên chỉ là stress test loại bỏ một chiều; không phải bằng chứng H2 hay upper bound toán học.

## 3. Phương pháp đề xuất

**Pilot khóa thiết kế:** prior lấy từ **output stage 1** (sau toàn bộ block stage 1); chỉ điều biến **K/V Spatial Reduction ở stage 2 và stage 3, trong mọi attention block**; stage 1 và 4 giữ nguyên. **Biến thể nghiên cứu riêng:** prior lấy từ output patch embedding trước các attention block stage 1 để thử stage 1–3, không dùng trong pilot. Không lấy output stage 1 để tác động ngược stage 1 (vòng phụ thuộc).

```text
5 bands → patch embedding stage 1 → attention stage 1 → F1 (output stage 1)
                                                       ├→ Weed prior head → p
                                                       └→ stage 2 (K/V SR được điều biến bằng p)
                                                               └→ stage 3 (K/V SR được điều biến bằng p)
Q và residual path: không điều biến; stage 4: giữ nguyên.
```

Với mỗi cửa sổ SR đầy đủ, `w = eps + (1-eps)*p`, `eps > 0`, `mu = mean_window(w) >= eps`, `w_norm = w/mu` (không cộng `delta`), `X_mod = X*(1+lambda_i*(w_norm-1))`, rồi `SRConv_i(X_mod) → LayerNorm → K/V`. `lambda_i = sigmoid(a_i)` khởi tạo gần 0,05. Vùng rìa không nằm trong cửa sổ SR có hệ số 1. **Mean theo từng cửa sổ của `w_norm` bằng 1 trong sai số floating-point**, và mean của hệ số điều biến cũng bằng 1. Đây là bảo toàn tổng **trọng số**, không phải bảo toàn tổng feature. Với mixed precision, tính `mu` và phép chia bằng FP32 trước khi cast lại dtype của feature. Nếu `eps` được chọn quá nhỏ, đo độ ổn định gradient; không dùng `clamp(mu, min=eps)` như một thay thế cho kiểm thử số học, dù clamp không đổi giá trị trong số học chính xác.

**Chú ý:** K/V SR của stage 3 có cửa sổ 2×2, chỉ bốn vị trí được phân phối lại; stage 2 là 4×4. Stage 1 có 8×8 nhưng chưa có prior từ output stage 1. Effective stride × R tại stage 1/2/3 lần lượt `4×8 = 8×4 = 16×2 = 32 px` trên ảnh gốc (bỏ qua sai khác do padding/biên). Đây là lý do chẩn đoán cả ba stage, **không** là bằng chứng H1 đã đúng.

Tính chất cần nhớ:

- **Khởi tạo tương đương baseline:** lớp cuối của prior head khởi tạo để `p` đồng đều, suy ra `w_norm = 1` và `X' = X` với mọi `lam`. Nhờ vậy dùng được `lam` khởi tạo **dương nhỏ**, và prior head nhận gradient từ segmentation loss ngay từ đầu. Nếu để `lam = 0` đúng thì đạo hàm theo `w_norm` bằng 0: head chỉ học từ aux loss ở bước đầu, còn cấu hình không có aux loss (K3) bị kẹt ở điểm yên ngựa. `lam = 0` chỉ dùng cho unit test (đầu ra phải trùng baseline).
- Chuẩn hóa theo cửa sổ làm trung bình của `w_norm` trong cửa sổ bằng 1 (trong sai số số học): cơ chế chỉ **phân phối lại đóng góp bên trong cửa sổ**, không loại cả cửa sổ nền. Cửa sổ có prior thấp đều nhau thì `w_norm = 1`, giữ nguyên ngữ cảnh Background.
- Một pixel Weed (`p = 1`) trong cửa sổ 8 × 8 với `eps = 0,1` cho `w_norm ≈ 8,77`, khá lớn nếu `lam` cao; vì vậy `lam` bị chặn trong [0, 1] và phải báo cáo giá trị `lam_i` sau huấn luyện của từng stage.
- **Về giới hạn trên `c`:** chuẩn hóa chỉ giữ trung bình `w_norm` bằng 1 **trước** khi clip. Sau `min(·, c)` tính chất này mất, cửa sổ có thể giảm đóng góp tổng thể thay vì chỉ phân phối lại. Bản pilot **bỏ clip** (cách A). Nếu về sau cần clip (cách B), phải chuẩn hóa lại sau clip (hoặc chiếu lên tập trọng số bị chặn có tổng cố định) và kiểm tra giới hạn trên còn đúng.
- **Không phải token pruning:** số K/V tokens không đổi. Tên mô tả chính xác: *Weed-Aware Spatially Weighted K/V Reduction*.
- **Rủi ro cơ chế:** SR vẫn là ánh xạ tuyến tính `R²·C -> C`. Điều biến chỉ đổi **ưu tiên**, không tăng dung lượng. Phải chứng minh bằng đo trực tiếp (probe ở 6.1), không chỉ bằng mIoU.
- **Rủi ro prior:** head dùng **output stage 1 (stride 4, sau các attention block của stage 1)** nên khó phân biệt Crop và Weed. Nếu G3 không đạt, thử head cho stage 2–3 lấy đầu vào từ đầu ra stage trước (ablation 6.5).
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

- **Pilot thống nhất với mục 3:** sinh `p_weed` từ output stage 1 rồi chỉ áp dụng CCMC-SR cho stage 2–3. Stage 1 không được dùng output của chính stage 1 để điều biến attention trong stage 1 (circular dependency). Biến thể sau: prior từ patch embedding trước block attention stage 1.
- Tại stage i, `w = eps + (1-eps)*p_weed`, `mu = AvgPool2d(w, kernel=R_i, stride=R_i, padding=0)`, `w_norm = w / upsample_repeat(mu, R_i)` **trong vùng các cửa sổ SR đầy đủ**. Phần rìa không thuộc cửa sổ giữ trọng số 1. Không dùng `delta` cộng mẫu số; `mu >= eps` trong số học chính xác.
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
| Diag-D0 | Weed recall theo bin kích thước, ma trận nhầm lẫn, lỗi theo ngày bay, trên checkpoint LiteSent và baseline không LLR | Cỏ nhỏ có bị bỏ sót nhiều hơn không |
| Diag-D1 | Quét `R` ở stage 1–2 (ví dụ 8→4→2): nội suy kernel SR, bảo toàn tổng trọng số, fine-tune cùng protocol | **Thăm dò (exploratory), không phải bằng chứng nhân quả.** Đổi `R` đồng thời đổi kích thước kernel, số K/V tokens, phạm vi gom và biểu diễn pretrained, rồi còn fine-tune |
| Diag-D2 | Linear decodability probe (protocol bên dưới) | Cỏ nhỏ có khó giải mã tuyến tính hơn sau SR không |

**Protocol Diag-D2:**

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

**Cảnh báo oracle leakage:** C2/C3 sử dụng GT mask của chính ảnh đang đánh giá để tạo trọng số, nên có thể tiết lộ sự hiện diện và vị trí lớp mục tiêu ngay cả khi số token K/V không đổi. C3 > C2 có thể chỉ phản ánh GT-Weed gần với nhãn dự đoán hơn GT-Veg; Small-Weed Recall đặc biệt dễ bị thổi phồng. **Không coi C3 là upper bound của learned prior, không gọi C3−C2 là hiệu ứng bảo toàn thông tin SR.** C3/C2 là kiểm tra loại bỏ một chiều: nếu GT-Weed vẫn không tốt hơn GT-Veg thì thiếu động lực cho prior Weed; nếu tốt hơn thì **chưa đủ điều kiện GO**.

**Đối chứng leakage có cùng nguồn nhãn:** C3-K5 = huấn luyện/đánh giá GT-Weed qua gate trước SR không window normalization; C3-K6 = GT-Weed gate sau SR (tùy ngân sách). So sánh paired `C3(SR) − C3-K5` để **kiểm tra hiệu ứng vị trí/toán tử trong điều kiện cùng leakage**, không được tự gọi là hiệu ứng SR thuần túy nếu hai cơ chế chưa match về biên độ gate và train budget. Báo cáo interaction theo difference-in-differences `[(C3_SR−C3_K5)−(C2_SR−C2_K5)]` như phân tích bổ trợ; vẫn chịu GT leakage và không thay thế L3 vs K5. K2 shuffle trong cửa sổ **giữ lại cờ có Weed ở cấp cửa sổ**, chỉ kiểm tra định vị trong cửa sổ; K7 random/frozen prior là placebo không mang nhãn, nhưng cần match phân phối để tránh confounding.

### 6.4 Đối chứng cơ chế

| ID | Cấu hình | Điều cần chứng minh |
|---|---|---|
| K1 | Head và aux loss giữ nguyên, **không nối** vào SR | Lợi ích không chỉ do auxiliary supervision |
| K2 | Prior bị hoán vị **trong từng cửa sổ SR** (giữ trung bình và phân phối giá trị của cửa sổ, phá tương ứng vị trí) | Cần đúng vị trí cỏ, không chỉ cần map có cùng phân phối. K2b (tùy chọn): hoán vị toàn ảnh làm placebo yếu hơn |
| K3 | Cùng head, **chỉ nhận gradient từ segmentation loss**, không có nhãn Weed phụ; `lam` khởi tạo dương như L3 (nếu `lam = 0` thì head không học được) | "Weed-aware" khác với bất kỳ spatial attention map nào |
| K4 | L3 + LLR | Bổ trợ hay thay thế; không bắt buộc trong pilot |
| **K5** | **Cùng prior head, aux loss, lambda và training budget với L3; gate trực tiếp trước SR nhưng KHÔNG chuẩn hóa trong cửa sổ** | Kiểm tra giá trị tăng thêm của window normalization so với spatial gating thông thường |
| **K6** | **Cùng prior head và aux loss; gate trên feature sau SR, trước LayerNorm/K/V projection; chuẩn hóa theo lưới K/V nếu phù hợp** | Kiểm tra lợi ích do vị trí can thiệp bên trong cửa sổ SR so với gating sau giảm độ phân giải |
| **C3-K5 / C3-K6** | **GT-Weed oracle qua gate trước/sau SR**, cùng split/seed/budget với C3 | Định lượng phần lợi thế oracle do cơ chế can thiệp, khi leakage GT được giữ chung; không phải chứng cứ deployable |
| K7 (phụ) | Frozen/random prior, kiểm soát prior head và phân phối trọng số | Kiểm tra cơ chế có thật sự cần prior đã học |

### 6.5 Ablation (pilot 1 seed trên validation, chạy tuần tự, không full factorial)

| Thứ tự | Nhóm | Các mức |
|---|---|---|
| 1 | Nguồn prior stage 1 | **output stage 1 (pilot)** · 5 băng thô (ablation) · đầu ra patch embedding (biến thể stage 1–3) |
| 2 | Nguồn prior stage 2–3 | Resize prior stage 1 · head riêng trên đầu ra stage trước |
| 3 | Stage áp dụng | **pilot: 2–3**; ablation hợp lệ với prior output stage 1: 2 · 3 · 2–3; **chỉ với prior từ patch embedding:** 1 · 1–2 · 1–3 |
| 4 | Cơ chế | `lam` ∈ {cố định nhỏ, học được bị chặn trong [0,1] (init 0,05)}; chuẩn hóa cửa sổ có/không; sàn `eps` ∈ {0,05, 0,1, 0,3}. Clip `c` **không** nằm trong pilot; chỉ thử ở cách B (clip rồi chuẩn hóa lại) sau khi cơ chế chính hoạt động |
| 5 | Nhãn và loss | Mục tiêu: soft occupancy (average pooling mask Weed) · max-pool presence; aux loss: BCE · BCE + Dice; trọng số 0,1/0,3/1,0 |

Mặc định dùng soft occupancy; max-pool presence là đối chứng, không giả định nó tốt hơn.

### 6.6 Bảng so sánh chính (5 seeds, sau khi khóa cấu hình)

| Mô hình | mIoU | Crop IoU | Weed IoU | Small-Weed Recall | Boundary-IoU | Params | FLOPs | Latency |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B0 5 băng, không LLR (L0) | | | | | | | | |
| LiteSent (B0 + LLR) | 85,49 ± 0,30* | 77,78* | 80,06* | | | 3,66 M | 17,09 G | 15,97 ms |
| CCMC-SR L3, không LLR | | | | | | | | |
| CCMC-SR L3 + LLR (K4) | | | | | | | | |
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
| G1 | SR có phải nút thắt cho cỏ nhỏ? | Dựa chủ yếu vào **Diag-D0 và Diag-D2**: nhóm nhỏ suy giảm rõ ở D0 và khó giải mã hơn sau SR ở D2. Diag-D1 chỉ hỗ trợ, không bắt buộc | Không giả định SR là nguyên nhân chính |
| G2 | Oracle có bác bỏ động lực Weed-specific? | **Gate loại bỏ một chiều:** nếu C3 không vượt C2 trên validation (paired seed, metric/CI), ghi NO-GO/INCONCLUSIVE cho giả thuyết oracle. Nếu C3 > C2 chỉ ghi SCREEN-PASS, **không phải H2 PASS**; yêu cầu L3 vs L2 và K5/K6 mới có GO khoa học | Không dùng C3>C2 để khẳng định H2 |
| G3 | Learned Weed prior có hơn learned Vegetation prior? | **H2 chính:** `Δ_H2 = M_L3 − M_L2` (Small-Weed Recall) ≥ 3 điểm phần trăm, paired 95% CI không cắt 0; Crop IoU giảm không quá 1 điểm; báo cáo Weed IoU/mIoU và prior quality. HR oracle chỉ báo cáo exploratory | Không claim Weed-specific superiority |
| G4 | Lợi ích đến từ việc nối prior vào SR? | L3 vượt K1, K2, K3 **và từng K5, K6** ở Small-Weed Recall, không đánh đổi nghiêm trọng mIoU/Crop IoU | Chưa chứng minh được cơ chế |

**Không dùng HR = (M_L3−M_L0)/(M_C3−M_L0) làm gate**: mẫu số có thể bị phóng đại do oracle leakage hoặc không dương. Nếu báo cáo HR, ghi rõ *oracle-relative descriptive ratio*, kèm mẫu số và CI, không gán ngưỡng 50%. Có thể báo cáo thêm tỉ lệ theo C3-K5 nhưng phép trừ leakage **không nhận dạng được** một upper bound sạch; không thay bằng mẫu số điều chỉnh tự chế. mIoU không tăng ở G3 vẫn có thể chấp nhận nếu Small-Weed Recall tăng rõ mà Crop IoU không hỏng. Ngược lại, mIoU tăng nhẹ mà Small-Weed Recall không tăng thì **chưa** chứng minh giả thuyết trung tâm.

**Bảng quyết định theo kết quả:**

| Kết quả | Quyết định |
|---|---|
| Diag-D0/Diag-D2 không ủng hộ H1 | Không tiếp tục claim SR là bottleneck |
| C3 không hơn C2 | Không có bằng chứng oracle ủng hộ; xem lại thiết kế hoặc dừng; không dùng để chứng minh H2 |
| C3 hơn C2 nhưng L3 không hơn L2 | **Không** xác nhận H2: chênh lệch oracle có thể là label leakage |
| C3(SR) không hơn C3-K5 | Không có bằng chứng SR tốt hơn gate ngay cả khi cùng GT leakage |
| Oracle tốt nhưng L3 kém | Kiểm tra prior head, calibration và GT leakage; không dùng HR làm lý do duy nhất |
| L3 không hơn K1 | Chưa chứng minh tác dụng của việc nối prior vào SR |
| L3 hơn K1 nhưng không hơn K2 | Chưa chứng minh cần đúng vị trí Weed |
| L3 hơn K1, K2 nhưng không hơn K3 | Chưa chứng minh "Weed-aware" khác content-adaptive |
| L3 hơn K1, K2, K3 nhưng không hơn K5 hoặc K6 | Chưa chứng minh đóng góp riêng của toán tử SR; không claim SR-specific novelty |
| L3 hơn K1, K2, K3, K5, K6 với CI paired đủ rõ | Có cơ sở tiếp tục ablation và 5 seeds |

## 8. Lộ trình

**Deadline cứng: 7 ngày lịch** để có gói kết quả validation và quyết định go/no-go. GPU-hours thực tế phụ thuộc quota còn lại, GPU được cấp và thời gian mỗi run; không giả định có sẵn 30 giờ hay 10× quota. Làm P0–P3 trước, chỉ đầu tư K1–K7 và 5 seeds khi các cổng đạt. Xem lịch chi tiết tại mục 8.1.

| Giai đoạn | Công việc | Kết quả cần có |
|---|---|---|
| [ ] P0 | Chẩn đoán checkpoint hiện có (Diag-D0); xác nhận hai baseline không-LLR | Recall theo bin, lỗi theo ngày, ma trận nhầm lẫn, baseline L0 chốt |
| [ ] P1 | Diag-D1 và Diag-D2 (quét `R` có chuyển trọng số, probe) | Bằng chứng hoặc bác bỏ H1, cổng G1 |
| [ ] P2 | Cài `WeightedSR` (có thể viết song song với P1); unit test mục 3.1 | Shape đúng ở các kích thước ảnh; `lam = 0` và prior đồng đều trùng baseline; backward không NaN; FLOPs đúng quy ước |
| [ ] P3a | L0 và mô hình ceiling C2, C3 (GT prior, 1 seed, validation) | G2 chỉ sàng lọc một chiều, chưa kiểm chứng H2 |
| [ ] P3b | L1–L3 và can thiệp oracle 6.3 | G3/H2 qua L3 vs L2; kiểm tra leakage C3-K5 và G4 |
| [ ] P4 | K1–K7 (ưu tiên K5/K6) và ablation 6.5 | Cổng G4 |
| [ ] P5 | Khóa cấu hình, 5 seeds, bootstrap, bảng 6.6 | Kết quả chính, CI, hiệu quả tính toán |
| [ ] P6 | Phân tích ngày bay (6.7) và tập dữ liệu thứ hai | Bằng chứng bổ trợ về độ bền |
| [ ] P7 | Đọc full paper các mục "cần đọc"; viết bài | Related work, đóng góp, giới hạn |

**Thứ tự code:** bắt đầu bằng 4 notebook độc lập (P0, P1, P2, P3) rồi mới tách thành repo. **Không tích hợp/train L3 (P3b) trước khi có kết quả P0–P1, P2 và P3a (C2/C3), trừ khi ghi rõ pilot kỹ thuật không dùng để quyết định khoa học.** Mức sẵn sàng hiện tại: P0–P2 có thể bắt đầu; chưa chạy toàn bộ thí nghiệm trước khi qua G1 và G2.

### 8.1. Sprint Kaggle bắt buộc 7 ngày (D1–D7)

**Mục tiêu bắt buộc đến cuối D7 (G2 chỉ sàng lọc, không thay L3–L2):** (i) baseline 2 seeds có thể tái hiện; (ii) C2/C3 và kết quả G2-screen có artifact hoặc ghi NOT RUN với lý do; (iii) nếu G1 đạt và G2 không bác bỏ, CCMC-SR stage 2–3 có checkpoint và ít nhất một so sánh **paired** với K5/K6 trên validation; (iv) bảng Weed IoU, Crop IoU, mIoU, Small-Weed Recall, thời gian suy luận; (v) quyết định **go/no-go** kèm hạn chế. **Không hứa hoàn thành 5 seeds cho mọi mô hình, external validation hay bài Q1 trong 7 ngày.** Nếu gate không đạt, kết quả âm tính + báo cáo cơ chế là đầu ra hợp lệ.

**Quy tắc Kaggle và 10 tài khoản:** 10 tài khoản **không đồng nghĩa** 10 GPU song song hoặc được phép cộng dồn quota. Chỉ dùng tài khoản do từng người sở hữu, có quyền sử dụng, tuân thủ điều khoản Kaggle; **không luân phiên tài khoản để né hạn mức/biện pháp kiểm soát**, không chia sẻ mật khẩu, không dùng tài khoản phụ nhằm vượt quota. Có thể chia nhóm theo **người phụ trách độc lập** (code, phân tích, kiểm thử, huấn luyện khi được cấp tài nguyên hợp lệ). Nếu tài nguyên không đủ, giảm phạm vi theo thứ tự ưu tiên bên dưới, hoặc dùng GPU được cấp hợp lệ từ nguồn khác. Quota/loại GPU/thời lượng phiên thay đổi theo tài khoản và thời điểm, phải kiểm tra trong Kaggle trước khi lên lịch.

| Ngày | Việc bắt buộc và thứ tự ưu tiên | GPU / công việc song song | Deliverable cuối ngày; cổng dừng |
|---|---|---|---|
| **D1** | Chốt official split, preprocessing 5 bands, metric instance/bin, seed; chạy baseline L0 **seed 1/2** hoặc xác minh checkpoint cùng protocol; kiểm toán sai lệch 82,16 vs 83,53 | 1 phiên GPU nếu cần; song song CPU: đọc full paper gần nhất, kiểm tra leakage, viết test metric | `configs/baseline.yaml`, `splits_hash.json`, `results/d1_baseline.json`, log môi trường; **không sang D3 nếu metric sai** |
| **D2** | Diag-D0 Small-Weed Recall, Crop/Weed IoU, theo ngày; **L0 seed 2**, Diag-D2 probe stage 1–3; unit test WeightedSR **trên tensor tổng hợp, chưa tích hợp HF** | GPU cho inference/probe; CPU viết operator + synthetic tests song song | `results/d2_diagnostics.csv`, unit tests shape/gradient/identity/window mean; nếu D0 không ủng hộ, ghi **G1 fail**, chỉ chạy pilot phản chứng |
| **D3** | **P3a: tích hợp tối thiểu đường truyền GT prior để chạy C2 và C3 (ceiling), cùng seed/protocol; chưa train L3**. Nếu chưa xong, tiếp tục P0/P1 và tests | GPU ưu tiên C2/C3; CPU review tính hợp lệ oracle, profiler | `results/d3_c2_c3.csv`, checkpoint C2/C3, G2 sơ bộ; **nếu chưa có C2/C3 thì L3 = NOT RUN** |
| **D4** | Hoàn tất C2/C3 nếu chưa đủ; kiểm tra G1/G2-screen; **chỉ khi đạt và đã chốt 6 điểm thiết kế**, tích hợp HF CCMC-SR stage 2–3, smoke test và **L2/L3 paired pilot** | GPU ưu tiên C2/C3 còn thiếu → L2/L3 paired pilot nếu G1 và G2-screen cho phép; CPU viết bootstrap, phân tích prior | `results/d4_gates.csv`, `checkpoints/l2_seed0.pt`, `checkpoints/l3_seed0.pt` nếu đủ điều kiện; nếu G2 loại bỏ, **không chạy L3 như thí nghiệm khẳng định** |
| **D5** | Chạy **K5 same-prior gate trước SR và K6 gate sau SR** với paired seed; nếu thiếu thời gian ưu tiên K5, đánh dấu K6 NOT RUN và **không claim SR-specific novelty**; profile latency | GPU 1–2 pilot ngắn tùy quota; CPU dựng bảng paired delta | `results/d5_k5_k6.csv`, `profile_t4.json`, bảng lỗi; **G3/G4 không đạt → không mở ablation rộng** |
| **D6** | Nếu G1–G4 có tín hiệu, đặc biệt L3>L2: lặp **L2/L3/K5/K6 cùng seed thứ 2**; thêm K1/K2/K3 và corruption khi đủ thời gian; đóng băng config | GPU ưu tiên xác nhận paired; CPU bootstrap theo image/cluster và audit | `results/d6_paired.csv`, config frozen, bảng mean/delta sơ bộ; không lấy test để chọn mô hình |
| **D7** | Đánh giá validation thống nhất, kiểm tra crop/weed trade-off, tổng hợp biểu đồ, chạy test chính thức **một lần duy nhất nếu đã khóa mô hình**; viết báo cáo quyết định | GPU inference/test + profiling khi hợp lệ; CPU viết report và đóng gói notebook | `results/summary_7days.md`, `results/metrics.csv`, `results/decision.json`, `README` cập nhật, notebook reproducible; danh sách việc còn thiếu cho paper |

**Ước lượng công suất (phải benchmark D1, không phải cam kết):** ký hiệu `t` = thời gian train **một** model/seed đến budget epoch đã khóa, `q` = tổng GPU-giờ **thực sự được cấp hợp lệ** trong 7 ngày. Kịch bản tối thiểu sau khi qua G1/G2 gồm L0 (2 seeds), C2, C3, L3, K5, K6: khoảng `7t + t_diagnostics + t_profile` GPU-giờ. Kịch bản xác nhận 2 seeds cho L0/L3/gate: thêm khoảng `3t`. Năm seeds của 3 model cần xấp xỉ `15t` (chưa tính ablation), **chỉ thực hiện nếu** `q` đủ. D1 chạy benchmark 5–10 epoch, đo thời gian/epoch và ghi `ETA = epochs_remaining × median_epoch_time`; thêm biên dự phòng 20–30% cho startup, checkpoint, lỗi phiên. Không tự suy ra GPU-hours từ số tài khoản.

**Điều phối phiên và checkpoint:** mỗi job có `run_id = model__seed__split_hash__git_commit`; đặt seed Python/NumPy/PyTorch và log versions; lưu checkpoint **theo epoch** hoặc mỗi 10–15 phút, gồm model/optimizer/scheduler/scaler/RNG/epoch/config, và xuất ra Kaggle Dataset/Artifact hoặc kho được phép truy cập để có thể resume giữa các phiên hợp lệ. Không phụ thuộc vào filesystem tạm của notebook; kiểm tra checkpoint đọc lại được ngay D1. Một job chỉ có một người/phiên ghi vào output để tránh overwrite. Ghi `manifest.csv`: run_id, chủ sở hữu, notebook version, GPU, start/end, wall time, GPU time, seed, trạng thái, đường dẫn artifact, hash. Không trộn checkpoints của các split/config khác nhau.

**Chia việc cho nhóm có tối đa 10 người/tài khoản hợp lệ (không yêu cầu 10 GPU):** nhóm A (2 người) data/metrics + split audit; B (2) WeightedSR + tests; C (2) training/baseline + logging; D (2) gate/oracle + controls; E (2) profiling, thống kê, related work và tổng hợp. Một người có thể kiêm nhiều vai nếu nhóm nhỏ hơn. Khi chỉ có **1 GPU khả dụng**, hàng đợi ưu tiên: **L0 (2 seeds) → C2/C3 → L2/L3 → K5/K6 → K1/K2/K3 → seed 2 → ablation → seed 3–5 → external**. Các job CPU triển khai song song, còn GPU chỉ chạy những job không xung đột tài nguyên.

**Định nghĩa hoàn thành 7 ngày:** phân biệt `DONE` (đã chạy và có artifact), `PARTIAL` (chưa đủ seed/CI) và `NOT RUN` (thiếu quota/thời gian). Không đưa số liệu từ pilot 1 seed thành bằng chứng kết luận Q1; không báo cáo test lặp lại để chọn hyperparameter. Sau D7, nếu kết quả tốt, tiếp tục 5 seeds, external dataset, robust checks và hoàn thiện related work trong giai đoạn tiếp theo.

### 8.2. Điều kiện khóa thiết kế và quy tắc thống kê (bắt buộc)

- **Chưa sửa code huấn luyện L3:** ưu tiên P0 (metric instance/bin, baseline L0 2 seeds, kiểm toán 82,16 vs 83,53) và P2 (unit test toán tử tensor tổng hợp). C2/C3 cần đường truyền GT prior nhưng không cần learned prior head; chỉ triển khai khi test toán tử đạt.
- **Metric instance/bin:** dùng instance labels gốc nếu có; nếu suy ra connected components từ semantic mask thì ghi rõ không tương đương instance GT. Định nghĩa kích thước trên **diện tích pixel ảnh gốc**, bins và matching rule cố định trước khi nhìn validation. Báo cáo số instance `n` mỗi bin trên train/val/test, recall `TP_instance/n_GT_instance`, và CI cluster-bootstrap theo ảnh (không bootstrap pixel như độc lập). Bin có `n < 20` trên validation/test chỉ **exploratory**, không dùng một mình cho gate.
- **Baseline audit:** kiểm tra split hash, decoder/fusion, weights, epoch selection, augmentation, cách tính mIoU và số seeds; không gộp 82,16 với 83,53 khi chưa xác nhận cùng protocol. L0 2 seeds để đo phương sai pilot, nhưng không thay thế 5 seeds xác nhận.
- **Ngưỡng đặt trước:** `L3−L0 ≥ +3 điểm phần trăm` Small-Weed Recall (hiệu quả tuyệt đối); `L3−L2 ≥ +3 điểm phần trăm` cho H2/G3; và `L3−Kj > 0` với **từng** đối chứng chính `j∈{1,5,6}` (paired CI 95% không cắt 0) cho G4. Không yêu cầu +3 điểm với từng đối chứng nếu không đủ power; phân biệt hiệu ứng tối thiểu thực tiễn và ý nghĩa thống kê. Các ngưỡng này phải khóa trước khi xem kết quả validation; oracle G2 không dùng ngưỡng +3 làm bằng chứng H2. Đồng thời không giảm Crop IoU quá **1 điểm phần trăm** trên validation, kèm paired bootstrap 95% CI cho delta; **đây là ngưỡng quản trị pilot đề xuất, phải khóa trước khi xem kết quả, không phải ngưỡng phổ quát**. Nếu CI cắt 0 thì ghi INCONCLUSIVE dù delta điểm đạt. Với mIoU, không kết luận từ một run có chênh lệch dưới 0,30 điểm (độ lệch chuẩn LiteSent đã báo); so sánh paired theo seed và ảnh. HR chỉ là mô tả exploratory, không phải điều kiện G3.
- **D7:** chỉ đánh giá test sau khi chốt model/config từ validation; nếu chưa đạt các gate, test để nguyên chưa mở. Ghi DONE/PARTIAL/NOT RUN từng thí nghiệm; deadline 7 ngày không cho phép bỏ gate mà vẫn claim novelty.

### 8.3. Ánh xạ ID để tránh hai hệ thí nghiệm song song

| Sprint ID | Protocol nghiên cứu | Ý nghĩa |
|---|---|---|
| N0 / D1 | P0, L0, Diag-D0 | Baseline, metric, audit |
| N1 / D2 | P1, Diag-D1–Diag-D2 | Diagnostic/probe, cả stage 1–3 |
| N2 / D3 | P2, P3a, C2/C3 | Unit tests, oracle ceiling |
| N3 / D4 | P3b, L3 | Learned Weed prior, chỉ sau G1/G2 |
| N4 / D5 | P4, K5/K6 | Same-prior gating controls |
| N5 / D6 | P4, K1/K2/K3 | Mechanism/placebo nếu đủ GPU |
| N6 / D6 | P5, paired seeds | Xác nhận và bootstrap |
| N7 / D7 | G1–G4, P5 | Tổng hợp, quyết định và test một lần nếu khóa cấu hình |

**Ưu tiên tên chuẩn:** D1–D7 = ngày sprint; `Diag-D0/1/2` = diagnostic ID trong mục 6.1. N0–N7 là tên nhóm bằng chứng lịch sử, **không phải thí nghiệm độc lập mới**.

### 8.4. Đặc tả triển khai HuggingFace (khóa API trước khi tích hợp)

**Chưa xác minh bằng `inspect.getsource` trên môi trường Kaggle đã pin:** môi trường soạn README hiện không cài `transformers`; các đường dẫn module sau là **giả thuyết API cần kiểm chứng**, không được xem là cam kết tương thích mọi version. D1 ghi `transformers.__version__`, `torch.__version__`, commit và xuất source của `SegformerEncoder.forward`, `SegformerEfficientSelfAttention.forward`, `SegformerOverlapPatchEmbeddings.forward`, đồng thời in `model.named_modules()` để xác nhận tên/shape.

1. **Prior capture (pilot):** dự kiến `forward_pre_hook` ở `model.segformer.encoder.patch_embeddings[1]` nhận input `F1` dạng `(B,C,H,W)` ngay trước patch embedding stage 2. Hook này **chỉ đúng nếu source đã pin** cho thấy encoder chuyển output stage 1 (sau layer norm và reshape) sang patch embedding stage 2. Assert `F1.ndim == 4`, stride 4, và hook gọi trước mọi SR stage 2. Tạo prior head từ **F1**, không dùng GT lúc inference. Không lấy output patch embedding stage 2 làm nhầm F1.
2. **SR interception:** dự kiến `forward_pre_hook` ở `.encoder.block[stage_idx][block_idx].attention.self.sr` (hoặc đường dẫn tương đương xác nhận bằng `named_modules`) cho **mọi block stage 2 và 3**; input SR phải là `(B,C,H_i,W_i)`. Hook trả tuple `(X_mod,)`, chỉ thay input của SRConv dùng cho **K/V**; Q và residual đi từ tensor gốc. Kiểm tra module có `sr` hay `sequence_reduction` theo version, không hard-code đường dẫn chưa xác minh. **Lưu ý:** hook trên `sr` là trước SR convolution, không phải trước toàn bộ attention; kiểm tra gradient/identity bằng code thật.
3. **Resize và chuẩn hóa:** `adaptive_avg_pool2d(p, (H_i,W_i))` cho prior 150→75/38; dùng `AvgPool2d(kernel_size=R,stride=R,ceil_mode=False)` cho mean **trong cửa sổ SR**, không dùng adaptive pool để thay window mean. Đối chiếu alignment 150→38 với thực tế patch embedding và đo sensitivity với lệch 1 pixel. Bỏ vùng rìa ngoài SR window; tại đó hệ số 1.
4. **Head/loss:** prior head xuất logits tại `F1` (150×150 cho input 600×600); `sigmoid` ra p. Target soft occupancy = `adaptive_avg_pool2d(mask_weed.float(), (150,150))`; `BCEWithLogitsLoss` ở độ phân giải đó, pilot `L_seg + 0.3 L_aux`. **Không detach F1** ở pilot; detach là ablation riêng. L2 dùng target Crop ∪ Weed cùng pipeline. Với C2/C3 dùng GT prior cùng phép downsample, không có aux head; chỉ là diagnostic GT-leaking.
5. **An toàn hook:** xóa cache theo forward, không global mutable state giữa concurrent forwards; hook closure per-model và thử gradient checkpointing, DDP, torch.compile riêng trước khi dùng. Hook có thể được gọi lại khi checkpoint recompute; ưu tiên wrapper/subclass tường minh nếu lifecycle không chắc. Test zero prior/dense prior, lambda=0, mean weight, forward/backward, batch>1, input 512/600, train/eval và autocast. Test thay đổi **chỉ** SR input của stage 2–3; Q và stage 1/4 phải bất biến khi cùng weights.
6. **Oracle leakage:** C3 sử dụng GT Weed của chính ảnh validation/test, tức là đưa nhãn đích vào mô hình qua K/V, không thể gọi kết quả là năng lực phân đoạn deployable hay upper bound sạch. K2 window shuffle giữ tín hiệu hiện diện cỏ theo cửa sổ; C3-K5/C3-K6 dùng cùng GT để định lượng đối chứng có leakage. **Không hiệu chỉnh HR bằng phép trừ C3-K5 một cách tùy tiện**; cần learned-prior L3 vs L2 và L3 vs K5/K6 để kết luận.

**Lệnh xác minh trên notebook Kaggle (chạy trước khi đăng ký hook):**

```python
import inspect, transformers
from transformers.models.segformer.modeling_segformer import (
    SegformerEncoder, SegformerEfficientSelfAttention, SegformerOverlapPatchEmbeddings
)
print(transformers.__version__)
for cls in (SegformerEncoder, SegformerEfficientSelfAttention, SegformerOverlapPatchEmbeddings):
    print(cls.__name__, inspect.getsource(cls.forward))
for name, module in model.named_modules():
    if 'patch_embeddings.1' in name or name.endswith('.sr'):
        print(name, type(module).__name__)
```

### 8.5. Tuyên bố novelty có thể kiểm chứng (không cam kết chưa từng công bố)

**Đóng góp ứng viên:** một phép **class-conditioned, per-SR-window normalized reweighting** của đầu vào SRConv trong K/V projection, bảo toàn mean trọng số trong cửa sổ và giữ số token K/V, dùng prior Weed **dự đoán từ cùng encoder**. Không gọi đây là token pruning, không khẳng định bảo toàn feature/information, không gọi việc thêm attention/gating là mới. Điểm khác biệt phải được chứng minh bằng đối chứng *same-prior same-budget* K5/K6, L3 vs L2, test lệch prior và chẩn đoán SR; nếu không thắng gate thông thường, novelty SR-specific không đứng vững.

**Khoảng trống chỉ là giả thuyết sau rà soát tài liệu:** phải đọc full text và công thức/implementation của các paper liên quan (đặc biệt adaptive patch weighting, SR attention, class-conditioned token selection) trước khi viết claim “first”. Mục tiêu Q1 không được bảo đảm bằng tên phương pháp; cần hiệu quả, kiểm chứng cơ chế, robustness, paired multi-seed và generalization.

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
- Tính mới chưa được xác nhận: đã xác minh metadata/abstract Dynamic Spatial Sparsification (TPAMI 2023), ALGM (CVPR 2024), MLSRS-SegFormer (MethodsX 2025); vẫn cần đối chiếu full method/code và tìm thêm prior art. Không xem việc chưa thấy toán tử trùng là bằng chứng tuyệt đối.

## 12. Tài liệu tham khảo

**Đã xác minh metadata/abstract trên nguồn xuất bản (không đồng nghĩa đã kiểm toán toàn bộ code):**

1. Celikkan et al., *WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields*, WACV 2025 (arXiv:2502.13103).
2. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
3. Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, ECCV 2018.
4. Rao et al., *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification*, NeurIPS 2021.
5. Rao et al., *Dynamic Spatial Sparsification for Efficient Vision Transformers and Convolutional Neural Networks*, IEEE TPAMI 45(9), 2023, DOI: 10.1109/TPAMI.2023.3263826; https://pubmed.ncbi.nlm.nih.gov/37030709/ .
6. Norouzi et al., *ALGM: Adaptive Local-then-Global Token Merging for Efficient Semantic Segmentation with Plain Vision Transformers*, CVPR 2024; https://openaccess.thecvf.com/content/CVPR2024/html/Norouzi_ALGM_Adaptive_Local-then-Global_Token_Merging_for_Efficient_Semantic_Segmentation_with_CVPR_2024_paper .
7. *Multi-level spatio-relational SegFormer (MLSRS-SegFormer)*, MethodsX 15 (2025), 103693, DOI: 10.1016/j.mex.2025.103693; https://pmc.ncbi.nlm.nih.gov/articles/PMC12637274/ . APW nằm tại PatchEmbedding, Hausdorff bias thêm vào attention logits; không thấy mô tả CCMC-SR trong phần phương pháp đã kiểm tra, nhưng không khẳng định paper không có phép tương tự ở nơi khác.

**Chưa xác minh đầy đủ trong lượt rà soát này; không dùng làm chứng cứ chắc chắn về novelty:** Dynamic Spatial Sparsification (**TPAMI 2023, đã xác minh DOI**); ALGM (**CVPR 2024, đã xác minh**); VLTP (WACV 2025); PDAViT (Neurocomputing 2026); ASVLB-Net (Pest Management Science 2026); VISA / BAWSeg (Remote Sensing 2026); FG-UNet (Pest Management Science 2024); Multispectral Feature Decomposition via Dual-Branch CNN–Transformer U-Net (2026).

## 13. Nhóm thực hiện

Nhóm 6 — Khoa Công nghệ Thông tin, HUIT.

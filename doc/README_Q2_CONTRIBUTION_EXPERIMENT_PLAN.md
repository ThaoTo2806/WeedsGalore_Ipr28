# README — Q2 Contribution & Experiment Plan (WeedsGalore)

> **Quy tắc phạm vi:** Giữ nguyên kết quả/artefact Exp00–07 và protocol Exp08 hiện có. `K8.3-WG-only` đã `ABORTED_AFTER_MARKER_BEFORE_COMPLETED_EVALUATION`, `rerun_authorized=false`. Không xóa/đổi marker `test_started.json`, không chạy lại Cell D hay chạy test một lần nữa; bất kỳ release mới nào cần **protocol deviation được người phụ trách phê duyệt**, audit evaluator và manifest mới có truy vết. WeedMap giữ `NOT_AUTHORIZED`.

## 0. Quyết định nghiên cứu

- **Bài Word cũ:** bài thực nghiệm ứng dụng SegFormer MiT-B0 tiền huấn luyện, ảnh UAV 5-band (R,G,B,NIR,RE), multi-scale fusion + nhánh Low-Level Feature Refinement (LLR); mục tiêu cải thiện Crop/Weed mIoU và hiệu quả tính toán so với công trình WeedsGalore. **Không phải bằng chứng test cho model OccCond-g/MCSD-g mới.**
- **Exp00–07:** chuyển từ _tăng mIoU nhờ ghép kiến trúc_ sang _giải thích và kiểm tra cơ chế mất cỏ nhỏ ở decoder_. **Ứng viên contribution chính:** occupancy-conditioned sub-cell decoding (OccCond-g), đối chứng với pixelshuffle/PS và MCSD-g (Sinkhorn).
- **Cơ hội novelty chính xác:** tách dự đoán _class occupancy_ theo ô coarse khỏi _spatial placement_ tại sub-cell, kiểm chứng với decoder/upsampler có cùng backbone và độ phân giải; Sinkhorn là biến thể ràng buộc/đánh đổi chứ không mặc định vượt OccCond-g.
- **Đích gần deadline:** chốt **claim + code audit + một đối chứng sát nhất nếu đủ ngân sách**, không thiết kế lại toàn bộ kiến trúc, không dùng test để chọn module. Q2 là mục tiêu xuất bản chứ không phải bảo đảm chấp nhận.

## 1. Bài Word cũ đã thực hiện gì? (KHÔNG trộn với Exp00–08)

Nguồn: `(NHÓM 6_UPDATE) CẢI THIỆN PHÂN ĐOẠN CÂY NGÔ VÀ CỎ DẠI TRÊN ẢNH UAV ĐA PHỔ(1).docx` (bản thảo người dùng cung cấp). Đây là những **kết quả được báo cáo trong tài liệu**, cần giữ provenance checkpoint và giao thức khi tái sử dụng.

| Mục               | Bài Word đã báo cáo                                                                                                    | Cách dùng khi xây dựng paper Q2 mới                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Dữ liệu           | WeedsGalore UAV maize 5-band, 156 ảnh, train/val/test 104/26/26, 3 lớp Background/Crop/Weed                            | Nền tảng dataset, **không** là dataset mới                                                                                      |
| Kiến trúc         | Pretrained SegFormer/MiT-B0, mở lớp input RGB→5-band, multi-scale feature fusion, LLR từ ảnh 5-band ở độ phân giải H/2 | Mốc bài ứng dụng/related prior work nội bộ; khác với sub-cell occupancy head                                                    |
| Test của bài Word | 85,49 ± 0,30% mIoU (5 seed), so mốc paper WeedsGalore DeepLabV3+ 82,90% và baseline tái hiện 82,88%                    | **Chỉ gán cho model Word cũ.** Không dùng làm test của OccCond-g/MCSD-g; so khác thiết lập phải ghi rõ                          |
| Ablation          | RGB-only 81,56 ± 0,67% test mIoU; 5-band 85,49 ± 0,30%; bỏ refinement 83,53 ± 0,40%; bỏ fusion 83,87 ± 0,76%           | Tín hiệu thực nghiệm của spectral/refinement, **không** khẳng định novelty thuật toán                                           |
| Compute           | 3,66M params; 17,09 “GFLOPs” (THOP) cho model cũ; latency 15,97ms T4                                                   | Cần lưu ý **THOP 512×512, thiếu matmul attention** theo review sau; không đối chiếu trực tiếp với FLOPs mới 600×600/FlopCounter |
| Calibration       | Temperature Scaling fit validation; test ECE 5-band 0,00473 ± 0,00031                                                  | Nhánh phụ của bài cũ, không phải đóng góp decoder mới                                                                           |
| PhenoBench        | Huấn luyện độc lập trên ảnh RGB sugar beet; validation mIoU 85,12% ở 512 và 89,83% ở 1024, cùng checkpoint             | **Không phải** cross-domain transfer của OccCond-g/MCSD-g, không so công bằng với baseline official test                        |

**Rủi ro phương pháp luận cần công khai:** Bài Word cũ đã sử dụng tập test WeedsGalore trong quá trình phát triển nghiên cứu trước; vì vậy không được ngầm khẳng định toàn bộ lịch sử Exp00–08 “chưa từng tiếp xúc test”. Kết quả Exp00–07 vẫn thuộc **validation**.

## 2. Exp00–07 đã hoàn tất — inventory evidence

Nguồn: `REVIEW_REPORT(5).md` và `README_Q1_PAPER_ROADMAP.md` do người dùng cung cấp. Đường dẫn dưới đây là vị trí **dự kiến trong repository Kaggle**; nếu khác phải dò nguồn gốc, không tạo file PASS giả.

| Exp | Trạng thái                               | Kết quả chốt / ý nghĩa                                                                                                                                      | Artifact tham chiếu                                                |
| --- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| E00 | DONE                                     | Unit/integration baseline, source/test guards, công cụ reproducibility                                                                                      | `doc/` và source audit                                             |
| E01 | DONE                                     | Cỏ nhỏ bị mất coverage; stage-2 SR probe có suy giảm decodability nhỏ, có tính tương quan                                                                   | `doc/exp01_results/results/exp01`                                  |
| E02 | DONE nhưng bằng chứng oracle **vô hiệu** | λ gần như không học; không dùng để ủng hộ hypothesis                                                                                                        | `doc/exp02_results/results/exp02`                                  |
| E03 | DONE, **negative result**                | Oracle CCMC-SR hoạt động nhưng không vượt L0 → dừng hướng SR; A/A cho thấy cần seed bootstrap                                                               | `doc/exp03_results/results/exp03`                                  |
| E04 | DONE                                     | D½-none tăng Small-Weed Recall +11,29 pp so L0; nút thắt quan trọng là **prediction resolution**                                                            | `doc/exp04_results/results/exp04`                                  |
| E05 | DONE                                     | PS, MCSD, MCSD-g; PS tăng recall nhưng làm giảm Crop/mIoU; occupancy/Sinkhorn bảo vệ độ chính xác hơn PS                                                    | `doc/exp05_results/results/exp05`                                  |
| E06 | DONE                                     | MCSD-g 5 seed vs L0: recall +7,84 [4,46;11,66] pp; mIoU +0,93 [0,35;1,78] pp; 1/5/20-iteration inference audits                                             | `doc/exp06_results/results/exp06`                                  |
| E07 | DONE                                     | OccCond-g 5 seed vs L0: recall +10,55 [6,39;14,84] pp; mIoU +0,88 [0,42;1,54] pp. Sinkhorn marginal fidelity tăng nhưng không chứng minh mIoU hơn OccCond-g | `doc/exp07_results/results/exp07`                                  |
| E08 | **ABORTED/PARTIAL**                      | WeedMap NOT_AUTHORIZED; WG-only K8.3 marker đã tạo, 0 logs/outputs đánh giá; không có test result của model mới                                             | `results/exp08/test_started.json`, abort audit, closeout, manifest |

**Đối chiếu E07 (MCSD-g − OccCond-g, validation, 5 seeds):** recall −2,71 [−6,02;+0,40] pp; `cov_small` −4,45 [−7,56;−1,32] pp; mIoU +0,06 [−0,17;+0,36] pp; Weed BIoU +1,10 [+0,16;+2,17] pp; Weed precision +0,98 [+0,28;+2,19] pp. Mean absolute marginal residual ≈0,000435 (MCSD-g 5 iters) so ≈0,007369 (OccCond-g), tổng hợp theo ô, **chưa phải CI chênh lệch**.

**Không đồng nhất metric:** pixel Weed precision ≠ số false-positive instance; `cov_small` ≠ recall@IoU 0,5; marginal fidelity với `q_pred` ≠ độ chính xác occupancy so với `q_GT`.

## 3. Lỗ hổng khoa học thực sự cần xoáy vào

### G1 — Resolution bottleneck (đã có bằng chứng)

- **Đã biết:** decoder 1/2 không thêm raw detail cải thiện mạnh recall cỏ nhỏ so với 1/4 + bilinear (E04).
- **Khoảng trống:** làm sao trả lại phân bố class ở sub-cell mà không phải chạy conv 1/2 dày đặc?
- **Điểm nhấn:** decoder sub-cell hiệu quả về FLOPs _nhưng cần báo cáo latency riêng_.

### G2 — Occupancy vs placement (novelty ứng viên, chưa chứng minh tính độc nhất)

- **Đã biết:** có PS không occupancy, OccCond-g và MCSD-g; OccCond-g > L0 trên validation.
- **Khoảng trống:** lợi ích nào do spatial sub-cell output, lợi ích nào do occupancy conditioning, lợi ích nào do spectral guide? Các phép toán deep subpixel mapping và class proportion đã tồn tại trước đó.
- **Trọng tâm code:** audit phép điều kiện hóa **đúng như mã hiện có**, sau đó ablation sạch, không thay loss/compute cùng lúc.

### G3 — “Better constraints ≠ better detection” (đã có bằng chứng trade-off)

- **Đã biết:** Sinkhorn ép marginal gần q_pred, tăng Weed BIoU/precision nhưng giảm `cov_small` so OccCond-g; không cải thiện rõ mIoU.
- **Khoảng trống:** khi q_pred đánh giá thấp q_GT, ép conservation có làm cỏ nhỏ biến mất? Chưa chứng minh nhân quả.
- **Trọng tâm code:** đo `q_pred - q_GT` theo bin occupancy và đối chiếu _cùng instance IDs_; không dùng GT occupancy để tạo predictions thật ở validation rồi chọn model.

### G4 — Đối chứng gần nhất về upsampling (chưa đủ)

- **Đã biết:** đã so PS, D½-none, LiteSent; chưa có đối chứng DySample/PointRend được triển khai trong cùng pipeline.
- **Khoảng trống:** reviewer có thể nói rằng tăng resolution bằng một upsampler đã có cũng đạt cùng lợi ích với ngân sách tương đương.
- **Trọng tâm code:** đúng **một** baseline gần nhất, với backbone/split/training/seed/metrics cố định; nếu không đủ thời gian thì đánh dấu chưa chạy và giảm claim hiệu quả hơn các phương pháp khác.

### G5 — Đánh giá độc lập và dữ liệu khác miền (đang bị chặn)

- **WeedsGalore test:** K8.3 abort sau marker, chưa có kết quả; phải phê duyệt deviation trước khi release mới.
- **WeedMap:** unresolved mask polarity, radiometry, evaluator; `NOT_AUTHORIZED`.
- **Carrot-Weed/CWFID:** RGB/robot mặt đất, không phải dữ liệu UAV 5-band tương thích trực tiếp. Hoãn full experiment; chỉ metadata/licence/mask-class audit nếu còn thời gian. Không dùng RGB làm giả NIR/RE.

## 4. Novelty: những gì là tiền lệ và claim nào còn hợp lý?

| Prior work (cần đọc full text kỹ)     | Đã có trong literature                                    | Điều phải phân biệt                                                     | Mức xác minh                                              |
| ------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------- |
| SegFormer (NeurIPS 2021)              | MiT + multiscale fusion + lightweight decoding            | Không nhận multiscale fusion/pretrained backbone là novelty             | Công bố chính thức                                        |
| PointRend (CVPR 2020)                 | Point-wise adaptive fine prediction                       | OccCond-g xuất xác suất trên lưới sub-cell; không chọn uncertain points | Công bố chính thức                                        |
| DySample (ICCV 2023)                  | Learned sampling offsets / feature grid sampling          | OccCond-g học class placement, không học tọa độ lấy mẫu feature         | Công bố chính thức                                        |
| CARAFE, DUC/PixelShuffle              | Content-aware feature upsampling / subpixel rearrangement | Cần PS + upsampler matched-head controls                                | Công bố chính thức                                        |
| Sinkhorn Label Allocation (ICML 2021) | Entropic OT với ràng buộc tỷ lệ label ở cấp dataset/batch | MCSD đặt row/column marginals ở cấp **ô coarse × sub-cell**             | Công bố chính thức; không nhận Sinkhorn là thuật toán mới |
| Deep subpixel mapping (IEEE TGRS)     | Fractional abundance → phân bổ lớp dưới pixel             | Không nhận occupancy/subpixel mapping là ý tưởng hoàn toàn mới          | Full-text comparison còn cần                              |
| STSPMNet (IEEE TGRS 2025)             | Deep spatiotemporal subpixel mapping/coarse-to-fine       | Nguồn ảnh/thời gian/decoder khác; phương trình cần đối chiếu            | Abstract/metadata; **chưa clearance full text**           |

**Công thức đối chiếu — để kiểm toán, KHÔNG tự suy ra code OccCond-g:** với `C=3`, `N=16`, occupancy `q_c=softmax(a)_c` và scores `s_{cj}`, một dạng tham chiếu là `P~_{cj}=softmax_c(s_{cj}/tau + log(q_c))`; nó chỉ đảm bảo `sum_c P~=1`, **không** đảm bảo `mean_j P~=q`. Kiểm tra mã `src/nets/mcsd.py` trước khi claim OccCond-g dùng đúng dạng này.

**MCSD constrained form (mô hình toán học):**

```text
min_{P>=0} -sum_{c,j} P[c,j]*s[c,j]
            + tau*sum_{c,j} P[c,j]*(log(P[c,j])-1)
subject to sum_j P[c,j] = N*q[c]
           sum_c P[c,j] = 1, for each j.
```

Finite 5 Sinkhorn iterations chỉ _approximately_ thỏa row marginal; không viết `exact mass conservation`.

**Claim an toàn hiện tại:** “We investigate occupancy-conditioned sub-cell decoding and quantify a constraint–sensitivity trade-off for multispectral small-weed segmentation using five-seed WeedsGalore validation.”

**Chưa được claim:** `first`, `SOTA`, `proven generalization`, `Sinkhorn universally improves segmentation`, `held-out improvements for OccCond-g`, `external-domain validation`.

## 5. Thí nghiệm/code cần chạy tiếp — chốt theo thứ tự và GO/NO-GO

> **Quy tắc chung:** tất cả thí nghiệm mới chỉ trên **synthetic, train hoặc validation** đã được phép dùng; dùng exploratory label và ghi vào một protocol/version mới. Không dùng kết quả để thay đổi checkpoint/manifest Exp08; không mở test. Trước khi train, đăng ký method, seed, metrics, criterion, budget và stopping rule; không chọn kết quả tốt nhất trong nhiều thử nghiệm rồi gọi là confirmatory.

| ID    | Ưu tiên            | Cần chạy?                                   | Thiết kế / code                                                                                                                                                                                                                                                      | Cần xuất                                                                             | Gate kết luận                                                                                                                     |
| ----- | ------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Q2-A0 | **P0**             | **BẮT BUỘC, không GPU**                     | Static inspect `mcsd.py`, `train.py`, `evaluate_ccmc.py`: xác nhận OccCond-g thực tế dùng q thế nào, `spectral_guide` đưa vào đâu, Sinkhorn nhận score nào, loss q_GT chỉ lấy train; trích mã/line/hash                                                              | `q2_decoder_implementation_audit.json`, source hash, sơ đồ toán tử                   | PASS khi source khớp checkpoint config và mọi claim phương trình; không khớp → sửa mô tả, không sửa checkpoint                    |
| Q2-A1 | **P0**             | **BẮT BUỘC, synthetic**                     | Unit tests shapes `[B,3,H,W]`, `q>=0`, sum q≈1, mỗi subcell class probs≈1, gradient finite, extremely small q, Sinkhorn residual (1/5/20) và không NaN                                                                                                               | `q2_decoder_unit_tests.json`, logs, test code                                        | 100% tests pass, mọi đối chứng tương thích output; test fail → STOP train                                                         |
| Q2-A2 | **P0**             | **BẮT BUỘC, reuse results**                 | Recompute/đối chiếu từ artifact E04–07 các cặp L0, PS, MCSD, MCSD-g, OccCond-g, D½-none/LiteSent; cùng 5 seed nếu sẵn, `10k` hierarchical bootstraps, size bins [45,147], BIoU 3px                                                                                   | CSV standardized + `q2_evidence_consistency.json`                                    | Hash provenance khớp; không trộn val/test; CIs dùng seed+image paired, không gán PASS từ 1 seed                                   |
| Q2-A3 | **P1**             | **NÊN, validation analysis; không retrain** | Phân tích q_pred vs q_GT theo Weed occupancy bins và ảnh/ngày; stratify instance nhỏ được/mất match giữa MCSD-g và OccCond-g. Dùng GT chỉ để **tính metric**, không feed vào model                                                                                   | `q2_occupancy_calibration_bins.csv`, paired IDs/CI, plots                            | Đủ số mẫu/bin; nếu q bias không giải thích trade-off → ghi `INCONCLUSIVE`                                                         |
| Q2-A4 | **P1**             | **NÊN nếu artifact sẵn; không retrain**     | Pareto plot và matched measurement: mIoU, recall, cov, BIoU, params, GFLOPs 600×600, CUDA-synchronized latency, peak VRAM. Giữ profiler/precision/batch/device thống nhất                                                                                            | `q2_compute_tradeoff.csv`, plot, profile logs                                        | Không so THOP 512 cũ với FlopCounter 600 mới như cùng thước đo                                                                    |
| Q2-A5 | **P1/CONDITIONAL** | **TỐI ĐA 1 new baseline**                   | Nếu còn GPU và approval: một decoder upsampling baseline **DySample** (ưu tiên) hoặc PointRend, không cả hai; same 5-band MiT-B0 + dataset, same seed 0–4, 100 ep, same loss/training selection/budget. Đăng ký trước adaptation và matched resolution; ghi overhead | source, five configs/checkpoints, 5 seed val summaries, bootstrap CIs, FLOPs/latency | Nếu không cải thiện hoặc CI bao 0: vẫn báo cáo; không chọn lại hyperparams sau nhìn val. Không phải confirmatory independent test |
| Q2-A6 | **P2/OPTIONAL**    | Chỉ nếu xác định thiếu đối chứng sạch       | Một factorial ablation occupancy/placement/spectral guide trên train+validation với đối chứng giữ gần nhất số tham số/output grid. **Không mở vô hạn nhiều cấu hình.**                                                                                               | protocol mới, 3–5 seed, CI, failure cases                                            | Attribution hợp lệ nếu thay đổi _một_ cơ chế, mọi yếu tố khác khớp; không dùng như bằng chứng test                                |
| Q2-T  | **BLOCKED**        | **KHÔNG CHẠY**                              | WeedsGalore held-out K8.3 đã abort sau marker                                                                                                                                                                                                                        | marker/abort audit/closeout giữ nguyên                                               | Chỉ reviewer-approved deviation mới có thể cho phép một release có nguồn gốc mới                                                  |
| Q2-X  | **DEFER**          | **KHÔNG FULL TRAIN**                        | WeedMap NOT_AUTHORIZED; Carrot-Weed RGB metadata feasibility only                                                                                                                                                                                                    | inventory/label/polarity/licence note                                                | Không suy luận external performance nếu chưa có protocol chuyển miền phù hợp                                                      |

### Giới hạn tài nguyên/tiêu chí dừng

- **Bắt buộc trước deadline:** A0 → A1 → A2. Có thể viết code/audit ngay trên Kaggle với synthetic hoặc phân tích output validation đã có.
- **Nếu còn 1–2 ngày:** A3 → A4 để tạo bằng chứng cơ chế và compute.
- **Chỉ nếu còn ngân sách rõ ràng và đã khóa protocol:** A5 (một baseline). Nếu không đủ 5 seed/CI thì ghi `PARTIAL`; không thay thế bằng 1 seed để claim thắng.
- **Không tự động làm A6**, vì phải chứng minh nó trả lời câu hỏi cơ chế mà A0–A5 còn chưa trả lời.
- Không sửa code gốc rồi nạp lại checkpoint gốc với semantics khác; tạo nhánh/commit riêng, giữ hash và thử compatibility trên synthetic.

## 6. Điều kiện chốt ba contribution (bắt buộc trả lời bằng evidence)

| Contribution                                     | Điều kiện để CHỐT                                                                                                                           | Nếu FAIL/INCONCLUSIVE                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **C1 — Resolution bottleneck**                   | E01/E04 artifact/CI/hash xác minh, và nêu giới hạn tương quan/độ phân giải                                                                  | Chuyển thành observation/diagnosis, không claim nhân quả tuyệt đối                          |
| **C2 — Occupancy-conditioned sub-cell decoder**  | A0 source/formula đúng, A1 unit tests pass, A2 5-seed baseline/PS controls nhất quán; novelty matrix full-text không phát hiện trùng đầy đủ | Thu claim về applied variant/comparative study, không dùng `first/new operator`             |
| **C3 — Marginal fidelity/sensitivity trade-off** | E07/A3 xác minh residual vs recall/coverage/BIoU/precision; mô tả quan hệ quan sát, không suy nguyên nhân tuyệt đối                         | Ghi một trade-off empirical, không gán lỗi recall cho “Sinkhorn” nếu chưa kiểm chứng causal |

**Research Qs có thể kiểm định:**

- RQ1: Prediction resolution chi phối Small-Weed Recall ở mức nào khi không đưa raw low-level details vào decoder?
- RQ2: OccCond-g cải thiện recall hơn L0/PS mà vẫn bảo vệ Crop/mIoU trên validation hay không?
- RQ3: Marginal-constrained allocation thay đổi fidelity và detection/boundary/precision như thế nào so với điều kiện hóa occupancy không Sinkhorn?
- RQ4 (nếu A5): Khi so với upsampling học được cùng backbone, liệu hiệu quả nhỏ-weed/compute của OccCond-g còn khác biệt?

## 7. Kế hoạch thực thi tối đa 7 ngày (dành cho người chỉ làm code/thí nghiệm)

| Ngày | Việc của bạn (code/experiment only)                                                                            | Output bàn giao                                       |
| ---- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| D1   | Snapshot commit/env; A0 audit model + frozen flags; rà CLI và no-GT inference                                  | JSON implementation audit + source hashes             |
| D2   | A1 synthetic unit/gradient/marginal tests + fix **test infrastructure only**                                   | Unit tests PASS và logs                               |
| D3   | A2 chuẩn hóa E04–07 result table, recompute paired seed/image comparisons                                      | Canonical validation CSV + comparison JSON            |
| D4   | A3 q calibration/size-bin audit và matched instance identities trên validation                                 | q-vs-GT bins + plots, CI                              |
| D5   | A4 compute profiling consistency, nếu có artifacts thì replot không chạy model                                 | FLOPs/latency Pareto table                            |
| D6   | Nếu đủ điều kiện thì A5 **một** DySample baseline bắt đầu theo protocol; nếu không thì viết mismatch/risk note | Protocol + run logs hoặc `NOT RUN` hợp lệ             |
| D7   | Verify SHA/artifacts, đóng gói evidence và bảng contribution PASS/INCONCLUSIVE                                 | Handoff `q2_evidence_pack/`, không có test result giả |

> Huấn luyện 5 seed baseline A5 có thể kéo dài quá 2 ngày nếu quota/phần cài đặt khác thực tế: **không hứa hoàn tất chỉ trong ngày D6**. Nếu không đủ thời gian, dừng tại A0–A4 là kết quả có giá trị hơn baseline nửa chừng.

## 8. Artifact pack và bảo toàn provenance

Dự kiến lưu các file mới dưới `results/q2_audit/` (không ghi đè `results/exp08/`):

```text
results/q2_audit/
  q2_decoder_implementation_audit.json
  q2_decoder_unit_tests.json
  q2_evidence_consistency.json
  q2_validation_main_table.csv
  q2_occupancy_calibration_bins.csv
  q2_compute_tradeoff.csv
  q2_contribution_gate.json
  logs/
```

**Bắt buộc metadata:** repo HEAD; source hash; checkpoint/config SHA-256; seed, split `675667edf3f35f16`; source metric; sample count (image/instance); sampler bootstrap; 95% CI; compute profiler/input shape/device; `evaluation_split=val`; `test_accessed=false` đối với các script mới (xác nhận theo thực tế, không tự gán nếu không xác minh).

**Source giấy tờ nội bộ:**

- `(NHÓM 6_UPDATE) CẢI THIỆN PHÂN ĐOẠN CÂY NGÔ VÀ CỎ DẠI TRÊN ẢNH UAV ĐA PHỔ(1).docx`
- `REVIEW_REPORT(5).md`
- `README_Q1_PAPER_ROADMAP.md`
- `08_external_validation_WG_ONLY_RUNBOOK(1).md`

**Nguồn học thuật để đọc/đối chiếu:**

- WeedsGalore, WACV 2025: https://openaccess.thecvf.com/WACV2025
- SegFormer, NeurIPS 2021: https://arxiv.org/abs/2105.15203
- PointRend, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html
- DySample, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html
- Sinkhorn Label Allocation, ICML 2021: https://proceedings.mlr.press/v139/tai21a.html
- STSPMNet, IEEE TGRS 2025: DOI https://doi.org/10.1109/TGRS.2025.3578351 (full text chưa được clearance trong audit hiện tại)

## 9. Quyết định cần người phụ trách nghiên cứu phê duyệt

- Dùng model nào làm **đề xuất chính** (OccCond-g ứng viên dựa trên validation; quyết định không được dựa trên test).
- Có bổ sung 1 baseline A5 không, và kinh phí/seed/protocol thế nào.
- Có được mở một **protocol deviation** cho K8.3 mới hay không; không mặc định có quyền chạy lại.
- Tuyên bố novelty cuối cùng sau full-text review và mọi kết quả A0–A4.

**Kết luận bàn giao:** Đã đủ bằng chứng để kiểm toán và kiểm tra cơ chế của contribution ứng viên C1–C3 trên validation. Cần **chạy A0–A2 tối thiểu**, nên chạy **A3–A4**; **A5 chỉ khi khả thi**. Không đổi checkpoint đã khóa, không mở lại test để hợp thức hóa kết quả, không chạy full Carrot-Weed trong tuần deadline.

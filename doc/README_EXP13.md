# Exp13 — Vegetation-Conditioned Factorised head (VCF) + B3, cổng validation > 86 rồi mới mở test

**Không phải bảo đảm đạt > 86 %.** Chưa chạy gì ở đây (không GPU, không truy cập repo); mã chưa được test trên Kaggle của bạn.

## 1. Đọc đúng kết quả Exp12 trước khi chọn hướng

| Cấu hình (val, 5 seed) | mIoU |
|---|---|
| B0 + LLR, không TTA | 83,73 |
| B2 + LLR, không TTA | 84,21 |
| B0 + LLR, D4 | 84,11 |
| **B2 + LLR, D4** | **84,72** |

- Còn thiếu ≈ **1,3 pp** để qua 86. B2, D4 mỗi thứ chỉ cho ≈ +0,4–0,6 pp; cộng dồn chưa đủ.
- Sub-cell (OCC2/MCSD2) thất bại vì đích sai: bg IoU đã 98,4 %, còn Crop ≈ 77 % và Weed ≈ 77 %, recall cỏ lớn @0,5 chỉ ≈ 50 %. Lỗi chính là **nhầm Crop ↔ Weed (ngữ nghĩa)**, không phải đặt biên ở mức sub-pixel.
- Kỳ vọng thực tế (ước đoán, chưa đo): B3 + VCF + D4 khoảng 85,0–85,6 val. **Khả năng qua 86 thấp.** Cổng bên dưới sẽ chặn mở test nếu không qua.

## 2. Cải tiến (VCF) — nhắm vào lỗi Crop↔Weed

Tách posterior: `P(bg)=1−v`, `P(crop)=v·(1−c)`, `P(weed)=v·c`.
`v` (thực vật) từ head nhẹ; `c = P(weed | vegetation)` từ **nhánh riêng** (depthwise dilated) chỉ nhận đặc trưng đã nhân cổng `sigmoid(v)`. Giữ nguyên SegFormer + raw LLR, loss CE+Dice như Exp10/12, không thêm loss phụ. Xuất log-prob nên dùng được với TTA/`predict_labels`.

**Về tính mới:** head phân cấp/factorised đã có trong văn học; tôi **không** khẳng định "chưa có paper nào". Phải làm tìm kiếm có hệ thống (ghi truy vấn + ngày): hierarchical/factorised softmax segmentation, vegetation-then-species / crop–weed two-stage head, WeedsGalore citations, MiT multispectral UAV. Chỉ viết "theo hiểu biết của chúng tôi…" sau bước đó. Nếu trùng, đóng góp thực sự nên là phân tích (vì sao sub-cell thất bại, lỗi chính là Crop↔Weed) chứ không phải operator.

## 3. Arms (đăng ký trước, `configs/exp13/exp13_arms.json`)

| Arm | Mục đích |
|---|---|
| `P_B2_LLR` (Exp12, đã có seed 0–4) | đối chứng |
| `P13_B2_VCF` | tác dụng riêng của VCF (cùng backbone) |
| `P13_B3_LLR` | tác dụng riêng của B3 |
| `P13_B3_VCF` | cấu hình PERF dự kiến |

Common config giữ nguyên `exp12_common_config.json`.

## 4. Cổng test (quy tắc)

1. Dev: seed 0–4 cho từng arm; chọn arm theo CI95 ghép cặp (`compare_runs.py`) như Exp12. Chọn **một** arm + **một** TTA, ghi lại **trước** khi train seed 5–9.
2. Xác nhận: train seed 5–9 của arm đã chọn, đánh giá **validation** với TTA đã chọn.
3. `exp13_gate.py`: trung bình val mIoU seed 5–9 **> 86,0** mới PASS. FAIL → thoát mã 2, **không mở test**, báo số nguyên bản (không chỉnh rồi gate lại).
4. PASS + `--run_test`: tạo marker độc quyền (`open(...,'x')`, chạy lần hai sẽ lỗi) rồi test đúng một lần/seed bằng `final_ema.pth`, weed_bias = 0.
5. Lưu ý trung thực: val không còn độc lập (26 ảnh dùng cho mọi lựa chọn từ Exp01), nên val > 86 là điều kiện cần chứ không bảo đảm test > 86. Báo cả CI95.

## 5. File

| File | Đặt vào repo | Việc |
|---|---|---|
| `exp13.py` | `src/nets/exp13.py` | head VCF (mới) |
| `exp13_apply_patch.py` | `src/scripts/` | sửa `segformer.py` (thêm b3), `ccmc_common.py`, `train.py` (đăng ký head `vcf`, backbone b3); mỗi thay thế có assert |
| `exp13_make_arms.py` | `src/scripts/` | sinh `configs/exp13/exp13_arms.json` |
| `exp13_gate.py` | `src/scripts/` | cổng val > 86 → test một lần |
| `test_exp13.py` | `tests/` | test chuẩn hoá log-prob, gradient |

`evaluate_ccmc.py` không cần sửa (đã dùng `define_model_flags` và `predict_labels`).

## 6. Quy trình Kaggle

```python
# 1) branch mới, copy file theo bảng, rồi:
!python src/scripts/exp13_apply_patch.py && git diff --stat
!python src/scripts/exp13_make_arms.py
!python -m pytest -q tests/test_exp13.py
# 2) profile B3 (exp12_profile.py cần thêm 'b3' vào choices) bs8 trước khi train; B3 ước ~70 phút/run trên T4
# 3) train: dùng lại train_one() của exp123.ipynb nhưng đặt arms_path=configs/exp13/exp13_arms.json
#    (common_path vẫn là exp12_common_config.json); dev: P13_B2_VCF, P13_B3_LLR, P13_B3_VCF seed 0-4
# 4) đánh giá val none + d4 (out_prefix .../val_none, .../val_d4), so sánh bằng compare_runs.py
# 5) chọn 1 arm + TTA, ghi vào README, train seed 5-9, đánh giá val_d4, rồi:
!python src/scripts/exp13_gate.py --arm P13_B3_VCF --tta d4 \
    --dataset_path=$DS --bins_json=doc/exp02_results/results/exp02/instance_bins.json
# chỉ khi PASS: thêm --run_test
```

Ngân sách (ước đoán): dev ≈ 3 arm × 5 seed; B3 chiếm phần lớn. Nếu thiếu giờ, chạy seed 0–2 cho `P13_B2_VCF` trước: nếu VCF không hơn `P_B2_LLR` thì dừng, không train B3+VCF.

## 7. Điều kiện dừng

- `P13_B2_VCF` không hơn `P_B2_LLR` (CI95 dưới ≤ 0) → bỏ VCF, báo là kết quả âm.
- Val seed 5–9 ≤ 86 → báo nguyên số, không mở test.

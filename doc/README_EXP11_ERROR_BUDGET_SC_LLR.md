# Exp11 — Error budget trước, Spectral-Contrast Adaptive LLR (SC-LLR) sau

> **Trạng thái: specification + runbook, CHƯA CHẠY.** Các cell phụ thuộc vào source chưa tồn tại (xem mục "Source contract"). Kaggle phải chạy đúng commit đã chứa source, không vá source bằng cell.
> Phạm vi: **validation only**. Không mở test trong Exp11.

## 1. Mục tiêu và vì sao thiết kế lại

Mục tiêu đã nêu: một phiên bản SegFormer + LLR cải tiến thật, mIoU > 86 %, đủ evidence để viết bài Scopus Q2. Trạng thái evidence hiện tại (validation, 5 seed):

| Thí nghiệm                         |      mIoU val | Kết luận đã ghi                               |
| ---------------------------------- | ------------: | --------------------------------------------- |
| E9-B0 / E10-B0 (LLR gốc)           | 84,41 / 84,42 | Baseline                                      |
| E9-M1 (OccCond trên LLR)           |         84,52 | INCONCLUSIVE (+0,11 [−0,08; +0,32])           |
| E10-B1 (residual 2×2)              |         84,51 | INCONCLUSIVE (+0,08 [−0,12; +0,25] so với B0) |
| E10-M1 (occupancy-guided residual) |         84,48 | INCONCLUSIVE (+0,06 [−0,29; +0,27] so với B0) |
| Mục tiêu                           |       > 86,00 | `TARGET_86_REACHED = FAIL`                    |

Khoảng cách tới mục tiêu khoảng **+1,58 pp** so với E10-B0 (script sẽ tính lại). Với IoU nền > 98 %, mức này tương đương tăng khoảng **+2,3 pp** cho mỗi lớp Crop và Weed. Mọi can thiệp decoder đã đo trên nền LLR đều ≤ +0,11 pp mIoU, nên **không có bằng chứng nào cho thấy 86 % đạt được bằng một thay đổi cùng loại**.

Vì vậy Exp11 đi theo thứ tự: **lỗi thực tế → trần của từng loại lỗi → cơ chế → đối chứng → mIoU**. Chỉ khi bước chẩn đoán cho thấy trần đủ lớn mới được huấn luyện.

### 1.1 Ba việc phải kiểm tra trước khi tin kết quả Exp10 hoặc đề xuất SC-LLR

1. **Nhánh residual của Exp10 có thực sự hoạt động không?** Exp10 khởi tạo gate `sigmoid(a) = 0,02` (tức `a ≈ −3,9`). Với Adam ở lr 3e-4 và khoảng 1 300 bước, `a` chỉ dịch tối đa khoảng 0,4, nên gate khó vượt khoảng 0,03. Đây là đúng kiểu lỗi đã gặp ở Exp02 (λ kẹt ở 0,05). Trọng số placement vẫn có thể lớn dần để bù, nên đây là **giả thuyết cần loại trừ**, không phải kết luận. Kiểm tra bằng activity audit (E11-D0): đặt nhánh residual về 0 lúc suy luận và đo ΔmIoU. Nếu |ΔmIoU| < 0,05 pp thì kết quả null của Exp10 không có thông tin.
2. **Bao nhiêu mIoU còn nằm ở từng loại lỗi?** Ceiling của biên, cỏ nhỏ, đảo nhãn nguyên cây Crop↔Weed, dương tính giả nền. Không có ceiling thì không biết 86 % có khả thi hay không.
3. **Validation không còn là bằng chứng độc lập.** Mọi lựa chọn từ Exp01 đến Exp10 đều dùng 26 ảnh validation, còn lần mở test K8.3 đã bị `ABORTED` và **không được phép chạy lại** nếu không có protocol deviation được duyệt (xem REVIEW mục 11). Do đó `TARGET_86_REACHED` trong Exp11 chỉ là `VALIDATION_ONLY`, không phải claim vượt baseline test `85,49 ± 0,30 %`.

## 2. Nhận xét về SC-LLR (hướng được đề xuất)

SC-LLR thay phép tổng hợp cục bộ tại H/2 bằng trọng số thích nghi dựa trên tương đồng phổ (5 băng) và tương đồng ngữ nghĩa:

$$
w_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}\!\left(-\frac{\lVert g_i-g_j\rVert^2}{2\sigma^2}+\frac{\langle u_i,u_j\rangle}{\tau\,\lVert u_i\rVert\lVert u_j\rVert}\right),\quad
\tilde F_i=\sum_{j\in\mathcal N(i)}w_{ij}F_j,\quad
F'_i=F_i+\alpha\,(\tilde F_i-F_i)
$$

với `g = Φ(x)` là phép chiếu 1×1 của ảnh 5 băng đã chuẩn hóa ở H/2, `u = Ψ(F)` là phép chiếu 1×1 của feature LLR, và `α = sigmoid(a)` là gate.

| Điểm                    | Nhận định                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tính mới                | Operator này thuộc họ **pixel-adaptive / guided filtering**: PAC (Su et al., CVPR 2019, arXiv 1904.05373), PPAC (Wannenwetsch & Roth, CVPR 2020, arXiv 2003.14407), semantic lattice (arXiv 1909.03677) và các upsampler dựa trên tương đồng (SAPA, ReSFU arXiv 2407.02283, có đề cập SegFormer). Phần có thể mới là **dẫn hướng bằng tương đồng phổ 5 băng trong LLR của SegFormer cho cỏ dại**, và phải được chứng minh bằng đối chứng M1 so với B1 và M2. Không viết "first" hay "novel operator". |
| Phạm vi cơ chế          | Cửa sổ `N(i)` chỉ 3×3 ở H/2 (khoảng 6×6 px ảnh gốc). Nó chỉ có thể sửa lỗi **ở biên và nhiễu cục bộ**. Lỗi đảo nhãn nguyên cây, hoặc cỏ nhỏ bị bỏ sót, nằm ngoài tầm với của nó.                                                                                                                                                                                                                                                                                                                      |
| Phụ thuộc vào chẩn đoán | Nếu ceiling của lỗi biên nhỏ, SC-LLR không thể đưa mIoU lên 86 %, bất kể thiết kế. Đó là lý do E11-D1 là cổng bắt buộc.                                                                                                                                                                                                                                                                                                                                                                               |
| Khởi tạo gate           | Phải tránh lặp lỗi Exp02: `α` init 0,1 và có lr riêng 0,01, ghi quỹ đạo `α` mỗi epoch và đo activity lúc đánh giá.                                                                                                                                                                                                                                                                                                                                                                                    |

## 3. Quy trình (pipeline) và quy tắc quyết định

```
E11-00 session + khóa tiền đăng ký
   ↓
E11-P0a source/checkpoint contract (chẩn đoán)  →  E11-P1a unit test chẩn đoán
   ↓
E11-D0 error budget + activity audit trên 15 checkpoint Exp10 (không train)
   ↓
E11-D1 quyết định route (script, đọc ngưỡng đã khóa)
   ├─ route A_LOCAL hợp lệ ──► E11-P0b/P1b (source SC-LLR) ──► E11-P2 train 4 arm × 5 seed ──► E11-P3 eval ──► E11-P4 GO/NO-GO
   ├─ route B (nhầm nhãn nguyên cây) hoặc C (cỏ nhỏ) hợp lệ ──► dừng Exp11, viết README Exp12 cho cơ chế tương ứng
   └─ không route nào hợp lệ ──► STOP_86_UNREACHABLE_BY_DECODER: dừng hướng decoder, viết paper theo evidence hiện có
```

### 3.1 Ngưỡng tiền đăng ký (khóa ở E11-00, **trước khi xem D0**)

Các ngưỡng này là lựa chọn thiết kế của dự án, không phải chuẩn khoa học. Có thể đổi, nhưng phải đổi **trước** khi chạy D0 và ghi lại lý do.

| Tham số                    | Giá trị                                                          | Ý nghĩa                                                                                    |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `target_miou`              | 86,0                                                             | Mục tiêu, chỉ `VALIDATION_ONLY`                                                            |
| `rho_max`                  | 0,5                                                              | Phần ceiling tối đa được giả định phải lấy lại; route hợp lệ nếu `ceiling ≥ gap / rho_max` |
| `component_vote_min_delta` | +0,5 pp                                                          | Ngưỡng cho phép bỏ phiếu theo thành phần liên thông (probe không dùng GT)                  |
| `activity_min_abs_delta`   | 0,05 pp                                                          | Dưới ngưỡng này thì nhánh bị coi là không hoạt động                                        |
| `primary_boundary_radius`  | 2 px                                                             | Bán kính vùng biên dùng cho phân hoạch lỗi                                                 |
| `small_weed_area_px`       | 45                                                               | Lấy từ `instance_bins.json` (edges `[45, 147]`)                                            |
| Bootstrap                  | `n_boot=10000`, `cluster=image`, `seed_bootstrap=True`, `seed=0` | Giống Exp10                                                                                |
| Crop guard                 | CI95 dưới > −1,0 pp                                              | Giống Exp10                                                                                |

### 3.2 Định nghĩa phân hoạch lỗi (error budget)

Với mỗi checkpoint, mỗi pixel lỗi (`pred ≠ GT`) được gán **một** nhóm theo thứ tự ưu tiên cố định:

1. `boundary_r2`: lỗi trong vòng 2 px quanh biên giữa hai lớp GT.
2. `small_weed`: GT thuộc thành phần liên thông Weed có diện tích ≤ 45 px.
3. `component_flip`: GT thuộc thành phần liên thông Crop/Weed mà lớp dự đoán chiếm đa số là lớp thực vật kia.
4. `background_fp`: GT là nền nhưng dự đoán là Crop/Weed (ngoài vùng biên).
5. `other`: phần còn lại.

**Ceiling** của một nhóm = mức tăng mIoU khi chỉ sửa các pixel của nhóm đó về GT (oracle). Báo thêm độ nhạy `boundary_r1` và `boundary_r3` (sửa mọi lỗi trong r px, độc lập với phân hoạch) và `component_vote` (không oracle: bỏ phiếu đa số cho thành phần liên thông dự đoán). Thứ tự ưu tiên ảnh hưởng đến phân hoạch nên phải báo cùng độ nhạy.

mIoU phải được tính **bằng đúng bộ gộp của `ccmc_metrics`** (theo ảnh hay gộp toàn tập); script ghi `aggregator` vào JSON và cổng tái lập (E11-D0) kiểm tra mIoU nền khớp summary Exp10 trong sai số 1e-3 pp.

## 4. Source contract

Các file sau phải được commit **trước** khi chạy tương ứng. Tên là đề xuất; nếu đổi, phải sửa cell P0.

**Bắt buộc cho chẩn đoán (D0/D1):**

| File                                  | Nội dung                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/scripts/exp11_error_budget.py`   | CLI: `--dataset_path --split=val --ckpt --bins_json --run_name --out_prefix --radii=1,2,3 --primary_radius=2 --component_vote=True --activity_audit=True` cộng các cờ model giống `evaluate_ccmc.py`. Từ chối `--split=test`. Xuất `{prefix}_budget.json` (schema dưới) và `{prefix}_budget_per_image.npz` (ma trận nhầm 3×3 theo ảnh cho nền và từng phép sửa). Có hàm `aggregate_miou`. |
| `src/scripts/exp11_route_decision.py` | CLI: `--budget_dir --prereg --out`. Gộp 5 seed, bootstrap phân tầng seed + ảnh cho ceiling, áp quy tắc mục 3.1, ghi `e11_route_decision.json`.                                                                                                                                                                                                                                            |
| `tests/test_exp11_error_budget.py`    | Test tổng hợp (mục 5.1).                                                                                                                                                                                                                                                                                                                                                                  |

Mô hình có nhánh bổ sung (E10 heads, SC-LLR) phải cho phép ghi đè cường độ nhánh bằng một hàm (`set_branch_scale(0.0)`) để activity audit chạy được.

Schema `{prefix}_budget.json` (tối thiểu): `run_name`, `split`, `n_images`, `aggregator`, `baseline{mIoU, IoU_bg, IoU_crop, IoU_weed}`, `categories{boundary_r1, boundary_r2, boundary_r3, small_weed, component_flip, background_fp, other}` (mỗi nhóm có `error_pixel_share`, `ceiling_delta_mIoU`), `partition_order`, `component_vote{delta_mIoU}`, `activity{delta_mIoU_when_branch_zero, branch_scale_supported}`. mIoU cùng đơn vị với summary của `evaluate_ccmc.py`.

**Chỉ cần nếu route A hợp lệ (SC-LLR):**

| File                                                                        | Nội dung                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/nets/sc_llr.py`                                                        | `SCLLRSegFormer` và `SC_LLR_HEADS = ('sc_llr_static', 'sc_llr_sem', 'sc_llr_spec')`                                                                                                                                                              |
| `tests/test_sc_llr.py`                                                      | Test mục 5.2                                                                                                                                                                                                                                     |
| Tích hợp trong `src/train.py`, `src/ccmc_common.py`, `src/evaluate_ccmc.py` | Cờ `--head`, `--sc_radius=1`, `--sc_guide_dim=8`, `--sc_embed_dim=16`, `--sc_alpha_init=0.1`, `--sc_alpha_lr=0.01`, `--sc_alpha_override` (chỉ dùng lúc đánh giá); ghi `sc_alpha` và entropy trung bình của `w_ij` vào `history.jsonl` mỗi epoch |

### 4.1 Đặc tả SC-LLR (cho route A)

- Vị trí: trên feature LLR ở H/2, **sau** khối refine cuối và **trước** classifier + bilinear gốc.
- Hợp nhất qua cộng dịch (shift-and-add) trên `K² = 9` offset để không phải mở (unfold) toàn bộ feature; bộ nhớ và FLOPs phải được profile, không ước lượng.
- `g = Φ(x)`: conv 1×1 từ 5 băng đã chuẩn hóa (cùng tensor mô hình nhận) xuống `sc_guide_dim`; `u = Ψ(F)`: conv 1×1 từ feature LLR xuống `sc_embed_dim`. `σ`, `τ` dương (softplus), khởi tạo cố định; **không quét `σ`, `τ`, bán kính trên validation**.
- Không thêm chỉ số thực vật (NDVI/NDRE): bài đối chiếu trên WeedsGalore cho thấy thêm nhiều chỉ số không chắc có lợi.
- Không có aux loss (`occ_weight = 0`) ở mọi arm, để chỉ khác biệt ở operator.
- Inference không đọc GT.

### 4.2 Ma trận arm (khi route A hợp lệ)

| ID     | `--head`        | Trọng số `w_ij`                                              | Câu hỏi                                                                 |
| ------ | --------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| E11-B0 | `bilinear`      | — (LLR gốc)                                                  | Mốc đối chứng; huấn luyện lại trong Exp11, không dùng lại checkpoint cũ |
| E11-B1 | `sc_llr_static` | Softmax của 9 tham số học được, **không phụ thuộc nội dung** | Tổng hợp lân cận đơn thuần có ích không                                 |
| E11-M2 | `sc_llr_sem`    | Chỉ tương đồng ngữ nghĩa                                     | Trọng số thích nghi không cần phổ                                       |
| E11-M1 | `sc_llr_spec`   | Tương đồng phổ **và** ngữ nghĩa                              | Phổ có thêm giá trị không                                               |

Số tham số của M1/M2/B1 phải chênh nhau dưới 1 % tổng tham số; nếu không, dừng và sửa trước khi train.

## 5. Unit test (cổng bắt buộc)

### 5.1 Chẩn đoán (`tests/test_exp11_error_budget.py`)

1. Dự đoán hoàn hảo cho mọi ngân sách bằng 0; sửa mọi pixel lỗi cho mIoU = 100.
2. Phân hoạch rời nhau và tổng `error_pixel_share` = 1.
3. Vùng biên r px đúng trên mask đồ chơi có đáp án tính tay.
4. `component_flip` chỉ đếm thành phần có lớp đa số dự đoán sai thuộc lớp thực vật kia.
5. `aggregate_miou` khớp với bộ gộp tham chiếu của `ccmc_metrics` trên dữ liệu ngẫu nhiên.
6. `--split=test` bị từ chối.

### 5.2 SC-LLR (`tests/test_sc_llr.py`, chỉ cho route A)

1. **Identity:** `α = 0` cho output trùng B0 trong sai số số học.
2. `w_ij ≥ 0` và tổng theo `j` bằng 1.
3. **Locality:** đổi một pixel ngoài bán kính không làm đổi `F̃_i`.
4. **Phụ thuộc phổ:** M1 đổi `w_ij` khi đổi ảnh 5 băng với `F` giữ nguyên; M2 và B1 thì không.
5. **Phụ thuộc ngữ nghĩa:** M1 và M2 đổi `w_ij` khi đổi `F`; B1 thì không.
6. Gradient riêng tới `Φ`, `Ψ`, `α`; không có GT trong forward; output hữu hạn.
7. Số tham số B1/M1/M2 chênh dưới 1 %.

## 6. Kaggle cells

### E11-00 — Session, commit sạch, khóa tiền đăng ký

```python
from datetime import datetime, timezone
from pathlib import Path
import glob, hashlib, json, os, subprocess, sys

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), PROJECT
os.chdir(PROJECT)

dirty = subprocess.check_output(
    ['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
assert not dirty, f'Có thay đổi tracked chưa commit:\n{dirty}'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = PROJECT / 'results' / 'exp11' / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
for sub in ('logs', 'runs', 'validation', 'diagnostics'):
    (OUT / sub).mkdir()
(PROJECT / 'results' / 'exp11' / 'CURRENT_RUN.txt').write_text(RUN_ID, encoding='utf-8')

if 'DS' not in globals():
    hits = sorted(glob.glob('/kaggle/input/**/2023-05-25/semantics', recursive=True))
    assert hits, 'Hãy Add Input WeedsGalore.'
    DS = str(Path(hits[0]).parent.parent)
assert Path(DS).is_dir(), DS
BINS = PROJECT / 'results' / 'instance_bins.json'
assert BINS.is_file(), BINS
edges = json.loads(BINS.read_text(encoding='utf-8'))['edges']
assert edges == [45.0, 147.0], edges

PREREG = {
    'experiment': 'exp11',
    'git_commit': commit,
    'dataset_path': DS,
    'split_hash_expected': '675667edf3f35f16',
    'evaluation_split': 'val',
    'test_access': False,
    'seeds': [0, 1, 2, 3, 4],
    'target_miou': 86.0,
    'target_scope': 'VALIDATION_ONLY',
    'rho_max': 0.5,
    'component_vote_min_delta_pp': 0.5,
    'activity_min_abs_delta_pp': 0.05,
    'primary_boundary_radius_px': 2,
    'boundary_radii_px': [1, 2, 3],
    'small_weed_area_px': 45,
    'partition_order': ['boundary_r2', 'small_weed', 'component_flip',
                        'background_fp', 'other'],
    'bootstrap': {'n_boot': 10000, 'cluster': 'image', 'seed_bootstrap': True, 'seed': 0},
    'crop_guard_ci_lower_pp': -1.0,
    'sc_llr': {'radius': 1, 'guide_dim': 8, 'embed_dim': 16,
               'alpha_init': 0.1, 'alpha_lr': 0.01, 'occ_weight': 0.0},
}
prereg_path = OUT / 'e11_preregistration.json'
text = json.dumps(PREREG, indent=2, sort_keys=True)
prereg_path.write_text(text, encoding='utf-8')
digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
(OUT / 'e11_preregistration.sha256').write_text(digest, encoding='utf-8')
print('RUN_ID:', RUN_ID, '| commit:', commit)
print('Prereg sha256:', digest)
print('Ngưỡng đã khóa; không sửa prereg sau khi chạy E11-D0.')
```

### E11-H — Header khôi phục sau khi restart kernel (chạy đầu mọi phiên mới, không chạy E11-00 lại)

```python
from pathlib import Path
import glob, hashlib, json, os, subprocess, sys

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
os.chdir(PROJECT)
if 'OUT' not in globals():
    RUN_ID = (PROJECT / 'results' / 'exp11' / 'CURRENT_RUN.txt').read_text().strip()
    OUT = PROJECT / 'results' / 'exp11' / RUN_ID
    assert OUT.is_dir(), OUT
PREREG = json.loads((OUT / 'e11_preregistration.json').read_text(encoding='utf-8'))
recorded = (OUT / 'e11_preregistration.sha256').read_text().strip()
import hashlib
current = hashlib.sha256(json.dumps(PREREG, indent=2, sort_keys=True).encode()).hexdigest()
assert current == recorded, 'Prereg đã bị sửa sau khi khóa.'
DS = PREREG['dataset_path']
BINS = PROJECT / 'results' / 'instance_bins.json'
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert head == PREREG['git_commit'], (head, PREREG['git_commit'])
print('OUT:', OUT)
```

### E11-P0a — Source contract chẩn đoán và checkpoint Exp10

```python
import glob, hashlib, json
from pathlib import Path

required = [
    'src/scripts/exp11_error_budget.py',
    'src/scripts/exp11_route_decision.py',
    'tests/test_exp11_error_budget.py',
]
source_hashes = {}
for rel in required:
    path = PROJECT / rel
    assert path.is_file(), f'Thiếu file đã cam kết: {rel}'
    source_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
assert 'def aggregate_miou' in (PROJECT / required[0]).read_text(encoding='utf-8')
assert "'test'" in (PROJECT / required[0]).read_text(encoding='utf-8') or '"test"' in (
    PROJECT / required[0]).read_text(encoding='utf-8'), 'Script phải có chốt chặn split test'
assert 'def set_branch_scale' in (PROJECT / 'src' / 'nets' / 'llr_subcell.py').read_text(
    encoding='utf-8'), 'E10 residual model must expose set_branch_scale for activity audit'

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

EXP10_MODELS = ('E10B0', 'E10B1', 'E10M1')
ckpts = {}
for model in EXP10_MODELS:
    for seed in PREREG['seeds']:
        run = f'{model}_s{seed}'
        hits = []
        for pattern in (f'/kaggle/input/**/{run}/best_val.pth',
                        str(PROJECT / 'results' / 'exp10' / '**' / run / 'best_val.pth'),
                        str(PROJECT / 'runs' / run / 'best_val.pth')):
            hits += glob.glob(pattern, recursive=True)
        hits = sorted(set(hits))
        assert hits, f'Không thấy checkpoint {run}. Add Input kết quả Exp10.'
        hashes = {sha256_file(h) for h in hits}
        assert len(hashes) == 1, f'{run}: nhiều checkpoint khác nhau {hits}'
        cfg = Path(hits[0]).parent / 'run_config.json'
        if cfg.is_file():
            split_hash = json.loads(cfg.read_text(encoding='utf-8')).get('split_hash')
            assert split_hash == PREREG['split_hash_expected'], (run, split_hash)
        summary_hits = sorted(set(
            glob.glob(f'/kaggle/input/**/{run}_val_summary.json', recursive=True)
            + glob.glob(str(PROJECT / 'results' / 'exp10' / '**' /
                            f'{run}_val_summary.json'), recursive=True)
        ))
        ckpts[run] = {
            'path': hits[0], 'sha256': next(iter(hashes)),
            'validation_summary': summary_hits[0] if len(summary_hits) == 1 else None,
        }

report = {'status': 'PASS_SOURCE_AND_CKPT_CONTRACT', 'source_sha256': source_hashes,
          'checkpoints': ckpts, 'test_accessed': False}
(OUT / 'e11_p0a_contract.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('E11-P0a PASS:', len(ckpts), 'checkpoint')
```

### E11-P1a — Test chẩn đoán tổng hợp

```python
import subprocess, sys, json
log = OUT / 'logs' / 'e11_p1a_tests.log'
with log.open('w', encoding='utf-8') as f:
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_exp11_error_budget.py'],
                       cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT, text=True)
print(log.read_text(encoding='utf-8')[-3000:])
assert r.returncode == 0, 'Cổng test chẩn đoán thất bại; không chạy D0.'
(OUT / 'e11_p1a_test_summary.json').write_text(
    json.dumps({'status': 'PASS', 'test_data_accessed': False}), encoding='utf-8')
```

### E11-D0 — Error budget và activity audit trên 15 checkpoint Exp10 (không train, chỉ validation)

```python
import json, subprocess, sys
assert json.loads((OUT / 'e11_p1a_test_summary.json').read_text())['status'] == 'PASS'
contract = json.loads((OUT / 'e11_p0a_contract.json').read_text(encoding='utf-8'))

BASE = ['--ccmc=False', '--use_low_level_refine=True', '--llr_source=raw', '--spectral_guide=False']
ARCH = {
    'E10B0': ['--head=bilinear'],
    'E10B1': ['--head=llr_residual2', '--beta=0.5', '--residual_alpha_init=0.02'],
    'E10M1': ['--head=llr_residual_occ2', '--beta=0.5', '--residual_alpha_init=0.02'],
}
for run, info in contract['checkpoints'].items():
    model = run.split('_s')[0]
    prefix = OUT / 'diagnostics' / f'{run}'
    if Path(str(prefix) + '_budget.json').is_file():
        print('Đã có:', run); continue
    cmd = [sys.executable, 'src/scripts/exp11_error_budget.py',
           f'--dataset_path={DS}', '--split=val', f'--ckpt={info["path"]}', *BASE, *ARCH[model],
           f'--bins_json={BINS}', f'--run_name={run}', f'--out_prefix={prefix}',
           '--radii=1,2,3', '--primary_radius=2', '--component_vote=True',
           '--activity_audit=True']
    print('Budget:', run)
    subprocess.run(cmd, cwd=PROJECT, check=True)
print('E11-D0 hoàn tất, chạy cổng tái lập tiếp theo.')
```

### E11-D0b — Cổng tái lập: mIoU nền phải khớp summary Exp10

```python
import json
from pathlib import Path

rows, status = [], 'PASS'
for run, info in contract['checkpoints'].items():
    budget = json.loads(Path(OUT / 'diagnostics' / f'{run}_budget.json').read_text(encoding='utf-8'))
    summary_path = Path(info['validation_summary']) if info.get('validation_summary') else None
    if summary_path is None or not summary_path.is_file():
        rows.append({'run': run, 'status': 'NO_SUMMARY'}); status = 'PARTIAL'; continue
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    ref = summary.get('metrics', {}).get('mIoU',
                                         summary.get('mIoU', summary.get('miou')))
    assert ref is not None, f'Không thấy khóa mIoU trong {summary_path}; điều chỉnh cell cho đúng schema.'
    diff = abs(float(budget['baseline']['mIoU']) - float(ref))
    ok = diff <= 1e-3
    status = status if ok else 'FAIL'
    rows.append({'run': run, 'budget_mIoU': budget['baseline']['mIoU'], 'summary_mIoU': ref,
                 'abs_diff': diff, 'status': 'PASS' if ok else 'FAIL'})
(OUT / 'e11_d0b_reproduction.json').write_text(
    json.dumps({'status': status, 'rows': rows}, indent=2), encoding='utf-8')
print('Reproduction:', status)
assert status == 'PASS', 'Bộ gộp mIoU hoặc checkpoint không khớp Exp10; dừng trước D1.'
```

### E11-D1 — Quyết định route (script đọc ngưỡng đã khóa)

```python
import json, subprocess, sys
assert json.loads((OUT / 'e11_d0b_reproduction.json').read_text())['status'] == 'PASS'
cmd = [sys.executable, 'src/scripts/exp11_route_decision.py',
       f'--budget_dir={OUT / "diagnostics"}',
       f'--prereg={OUT / "e11_preregistration.json"}',
       f'--out={OUT / "e11_route_decision.json"}']
subprocess.run(cmd, cwd=PROJECT, check=True)
decision = json.loads((OUT / 'e11_route_decision.json').read_text(encoding='utf-8'))
for key in ('gap_pp', 'min_ceiling_required_pp', 'ceilings', 'component_vote',
            'activity_audit', 'eligible_routes', 'proceed_to_training', 'reason',
            'E10_INACTIVE_SUSPECTED'):
    print(key, '=', json.dumps(decision.get(key), indent=2))
```

Cách đọc kết quả:

| `eligible_routes`                        | Hành động                                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| chứa `A_LOCAL`                           | Tiếp tục P0b → P1b → P2                                                                                         |
| chỉ `B_COMPONENT` và/hoặc `C_SMALL_WEED` | Dừng Exp11. SC-LLR không nhắm đúng lỗi; thiết kế README Exp12 riêng                                             |
| rỗng (`STOP_86_UNREACHABLE_BY_DECODER`)  | Dừng hướng decoder. Viết paper theo evidence hiện có (mục 8)                                                    |
| `E10_INACTIVE_SUSPECTED = true`          | Kết quả null của Exp10 không có thông tin; cân nhắc Exp10b (gate hoạt động, tiền đăng ký riêng) **ngoài** Exp11 |

### E11-P0b — Source contract SC-LLR (chỉ chạy nếu `A_LOCAL` hợp lệ)

```python
import hashlib, json
decision = json.loads((OUT / 'e11_route_decision.json').read_text(encoding='utf-8'))
assert decision['proceed_to_training'] is True and 'A_LOCAL' in decision['eligible_routes'], \
    'Route A không hợp lệ theo ngưỡng đã khóa; không train.'

required = ['src/nets/sc_llr.py', 'tests/test_sc_llr.py', 'src/train.py',
            'src/ccmc_common.py', 'src/evaluate_ccmc.py']
hashes, text = {}, {}
for rel in required:
    path = PROJECT / rel
    assert path.is_file(), f'Thiếu {rel}'
    hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    text[rel] = path.read_text(encoding='utf-8')
assert 'class SCLLRSegFormer' in text['src/nets/sc_llr.py']
assert "SC_LLR_HEADS = ('sc_llr_static', 'sc_llr_sem', 'sc_llr_spec')" in text['src/nets/sc_llr.py']
for rel in ('src/train.py', 'src/ccmc_common.py'):
    assert 'SC_LLR_HEADS' in text[rel], f'Chưa đăng ký head SC-LLR trong {rel}'
for flag in ('sc_alpha_init', 'sc_alpha_lr', 'sc_radius'):
    assert flag in text['src/train.py'], f'Thiếu cờ {flag}'
assert 'sc_alpha_override' in text['src/evaluate_ccmc.py']
(OUT / 'e11_p0b_contract.json').write_text(
    json.dumps({'status': 'PASS_SC_LLR_CONTRACT', 'source_sha256': hashes}, indent=2),
    encoding='utf-8')
print('E11-P0b PASS')
```

### E11-P1b — Test SC-LLR (identity, locality, phụ thuộc phổ/ngữ nghĩa, gradient, tham số)

```python
import json, subprocess, sys
log = OUT / 'logs' / 'e11_p1b_tests.log'
with log.open('w', encoding='utf-8') as f:
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_sc_llr.py'],
                       cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT, text=True)
print(log.read_text(encoding='utf-8')[-3000:])
assert r.returncode == 0, 'Cổng test SC-LLR thất bại; không train, không đổi tên phương pháp để che lỗi.'
(OUT / 'e11_p1b_test_summary.json').write_text(
    json.dumps({'status': 'PASS', 'test_data_accessed': False}), encoding='utf-8')
```

### E11-P2 — Train 4 arm × 5 seed (cùng protocol Exp10)

Thứ tự theo seed để nếu hết quota thì vẫn có seed đầy đủ cho cả bốn arm. Ước lượng khoảng 25 phút mỗi run, 20 run ≈ 8,3 giờ GPU; chia nhiều phiên, cell bỏ qua run đã xong và resume run dở dang.

```python
import json, subprocess, sys
assert json.loads((OUT / 'e11_p0b_contract.json').read_text())['status'] == 'PASS_SC_LLR_CONTRACT'
assert json.loads((OUT / 'e11_p1b_test_summary.json').read_text())['status'] == 'PASS'

common = [
    f'--dataset_path={DS}', '--dataset_size_train=104', '--in_channels=5', '--num_classes=3',
    '--batch_size=8', '--num_workers=4', '--lr=0.0003',
    '--weight_bg=0.9', '--weight_crop=1.0', '--weight_weed=1.3',
    '--use_dice_loss=True', '--dice_weight=0.3', '--epochs=100',
    '--log_interval=50', '--ckpt_interval=1000000', '--pretrained_backbone=True',
    '--use_multiscale_fusion=True', '--spectral_guided=False', '--spectral_guide=False',
    '--measure_stats=False', '--resume=True', '--ccmc=False', '--segformer=True',
    '--use_low_level_refine=True', '--llr_source=raw', '--occ_weight=0.0',
]
sc = PREREG['sc_llr']
sc_flags = [f'--sc_radius={sc["radius"]}', f'--sc_guide_dim={sc["guide_dim"]}',
            f'--sc_embed_dim={sc["embed_dim"]}', f'--sc_alpha_init={sc["alpha_init"]}',
            f'--sc_alpha_lr={sc["alpha_lr"]}']
heads = {
    'E11B0': ['--head=bilinear'],
    'E11B1': ['--head=sc_llr_static', *sc_flags],
    'E11M2': ['--head=sc_llr_sem', *sc_flags],
    'E11M1': ['--head=sc_llr_spec', *sc_flags],
}
for seed in PREREG['seeds']:
    for model in ('E11B0', 'E11B1', 'E11M1', 'E11M2'):
        run = f'{model}_s{seed}'
        run_dir = OUT / 'runs' / run
        if (run_dir / 'run_summary.json').is_file():
            print('Đã xong:', run); continue
        log_path = OUT / 'logs' / f'train_{run}.log'
        cmd = [sys.executable, 'src/train.py', *common, *heads[model],
               f'--seed={seed}', f'--run_name={run}', f'--out_dir={run_dir}']
        print('Train:', run)
        with log_path.open('a', encoding='utf-8') as log:
            r = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            print(log_path.read_text(encoding='utf-8')[-4000:])
            raise RuntimeError(f'{run} lỗi; giữ log và dừng.')
        assert (run_dir / 'run_summary.json').is_file(), run
print('E11-P2 xong')
```

Kiểm tra activity trong khi train (chạy sau seed 0, không dùng để chọn cấu hình): đọc quỹ đạo `sc_alpha` từ `history.jsonl`.

```python
import json
for model in ('E11B1', 'E11M2', 'E11M1'):
    path = OUT / 'runs' / f'{model}_s0' / 'history.jsonl'
    if not path.is_file():
        continue
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    a = [r.get('sc_alpha') for r in rows if r.get('sc_alpha') is not None]
    print(model, 'alpha ep0/10/50/cuối:',
          [round(a[min(i, len(a) - 1)], 3) for i in (0, 10, 50, len(a) - 1)] if a else 'KHÔNG GHI')
```

Nếu `α` không rời khỏi khoảng giá trị khởi tạo ở seed 0 thì đó là lỗi triển khai (lặp lại Exp02), phải sửa và đăng ký lại **trước** khi chạy các seed còn lại.

### E11-P3 — Đánh giá validation, activity override và so sánh paired

```python
import json, subprocess, sys
from pathlib import Path

BASE = ['--ccmc=False', '--use_low_level_refine=True', '--llr_source=raw',
        '--spectral_guide=False', f'--bins_json={BINS}', '--boundary_px=3']
sc = PREREG['sc_llr']
sc_arch = [f'--sc_radius={sc["radius"]}', f'--sc_guide_dim={sc["guide_dim"]}',
           f'--sc_embed_dim={sc["embed_dim"]}']
archs = {
    'E11B0': ['--head=bilinear'],
    'E11B1': ['--head=sc_llr_static', *sc_arch],
    'E11M2': ['--head=sc_llr_sem', *sc_arch],
    'E11M1': ['--head=sc_llr_spec', *sc_arch],
}
runs = {}

def evaluate(model, seed, tag=None, extra=()):
    run = f'{model}_s{seed}'
    name = f'{tag or model}_s{seed}'
    ckpt = OUT / 'runs' / run / 'best_val.pth'
    assert ckpt.is_file(), ckpt
    prefix = OUT / 'validation' / f'{name}_val'
    cmd = [sys.executable, 'src/evaluate_ccmc.py', f'--dataset_path={DS}', '--split=val',
           f'--ckpt={ckpt}', *BASE, *archs[model], *extra,
           f'--run_name={name}', f'--out_prefix={prefix}']
    subprocess.run(cmd, cwd=PROJECT, check=True)
    assert Path(str(prefix) + '_per_image.csv').is_file()
    runs[name] = str(prefix)

for model in archs:
    for seed in PREREG['seeds']:
        evaluate(model, seed)
for seed in PREREG['seeds']:                       # activity audit: nhánh SC-LLR bị tắt lúc suy luận
    evaluate('E11M1', seed, tag='E11M1zero', extra=['--sc_alpha_override=0.0'])

(OUT / 'e11_p3_validation_manifest.json').write_text(json.dumps(
    {'evaluation_split': 'val', 'test_accessed': False, 'runs': runs}, indent=2), encoding='utf-8')

boot = PREREG['bootstrap']
contrasts = [('E11M1', 'E11B0'), ('E11M1', 'E11B1'), ('E11M1', 'E11M2'),
             ('E11B1', 'E11B0'), ('E11M2', 'E11B0'), ('E11M1', 'E11M1zero')]
for a, b in contrasts:
    pa = [f'{runs[f"{a}_s{s}"]}_per_image.csv' for s in PREREG['seeds']]
    pb = [f'{runs[f"{b}_s{s}"]}_per_image.csv' for s in PREREG['seeds']]
    cmd = [sys.executable, 'src/scripts/compare_runs.py',
           f'--a={",".join(pa)}', f'--b={",".join(pb)}', f'--name_a={a}', f'--name_b={b}',
           f'--cluster={boot["cluster"]}', f'--seed_bootstrap={boot["seed_bootstrap"]}',
           f'--n_boot={boot["n_boot"]}', f'--seed={boot["seed"]}', '--gate=none',
           '--metrics=SmallWeedRecall,cov_small,mIoU,IoU_weed,IoU_crop,BIoU_weed,PrecisionWeed',
           f'--out={OUT / f"cmp_{a}_vs_{b}.json"}']
    subprocess.run(cmd, cwd=PROJECT, check=True)
print('E11-P3 xong; đọc toàn bộ JSON trước khi kết luận.')
```

### E11-P4 — GO/NO-GO (ba kết luận tách riêng)

```python
import json

def load(name):
    return json.loads((OUT / f'cmp_{name}.json').read_text(encoding='utf-8'))

def mean_delta(m):
    return m.get('delta', m['A_mean'] - m['B_mean'])

def decision(rec):
    miou, crop = rec['metrics']['mIoU'], rec['metrics']['IoU_crop']
    if miou['ci95'][0] > 0 and crop['ci95'][0] > PREREG['crop_guard_ci_lower_pp']:
        return 'PASS'
    if miou['ci95'][1] <= 0 or crop['ci95'][1] <= PREREG['crop_guard_ci_lower_pp']:
        return 'FAIL'
    return 'INCONCLUSIVE'

m1_b0, m1_b1, m1_m2, m1_zero = (load('E11M1_vs_E11B0'), load('E11M1_vs_E11B1'),
                                load('E11M1_vs_E11M2'), load('E11M1_vs_E11M1zero'))
activity = m1_zero['metrics']['mIoU']
active = abs(mean_delta(activity)) >= PREREG['activity_min_abs_delta_pp']

alpha_final = {}
for model in ('E11B1', 'E11M2', 'E11M1'):
    vals = []
    for seed in PREREG['seeds']:
        p = OUT / 'runs' / f'{model}_s{seed}' / 'history.jsonl'
        rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        vals.append(rows[-1].get('sc_alpha'))
    alpha_final[model] = vals

spectral_value = 'PASS' if decision(m1_b1) == 'PASS' and decision(m1_m2) == 'PASS' else (
    'FAIL' if m1_b1['metrics']['mIoU']['ci95'][1] <= 0 else 'INCONCLUSIVE')
report = {
    'IMPROVED_VS_LLR': decision(m1_b0),
    'SPECTRAL_GUIDANCE_VALUE': spectral_value,
    'MECHANISM_ACTIVE': 'YES' if active else 'NO_BRANCH_EFFECT_AT_INFERENCE',
    'TARGET_86_REACHED': 'PASS' if m1_b0['metrics']['mIoU']['A_mean'] > PREREG['target_miou'] else 'FAIL',
    'target_scope': 'VALIDATION_ONLY',
    'primary': {'H1_M1_vs_B0': decision(m1_b0), 'H2_M1_vs_B1': decision(m1_b1),
                'H3_M1_vs_M2': decision(m1_m2)},
    'alpha_final_by_arm': alpha_final,
    'test_accessed': False,
    'interpretation': ('Chỉ H1 PASS kèm Crop guard mới hỗ trợ cải thiện đã kiểm chứng so với LLR. '
                       'SPECTRAL_GUIDANCE_VALUE cần cả H2 và H3 PASS. '
                       'Mean validation > 86 không phải claim test.'),
}
(OUT / 'e11_go_no_go.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
```

### E11-P5 — Đóng gói

```python
from zipfile import ZIP_DEFLATED, ZipFile
import glob

res_zip = Path('/kaggle/working') / f'exp11_{PREREG["git_commit"][:7]}_results.zip'
with ZipFile(res_zip, 'w', compression=ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.pth':
            z.write(p, arcname=str(p.relative_to(PROJECT)))
ckpt_zip = Path('/kaggle/working') / f'exp11_{PREREG["git_commit"][:7]}_ckpts.zip'
with ZipFile(ckpt_zip, 'w', compression=ZIP_DEFLATED) as z:
    for p in sorted(glob.glob(str(OUT / 'runs' / '*' / 'best_val.pth'))):
        z.write(p, arcname=str(Path(p).relative_to(PROJECT)))
print(res_zip, ckpt_zip)
```

## 7. Cách diễn giải và claim boundary

- **Không gọi** Exp11 là ensemble hay hợp nhất hai mô hình; vẫn là một mạng SegFormer + LLR với một operator ở H/2.
- **Không** viết "operator mới" hay "first". Viết đúng: _dẫn hướng bằng tương đồng phổ trong tổng hợp cục bộ của LLR_, đối chiếu với PAC/PPAC/SAPA, và chỉ khẳng định điều mà M1 vs B1 và M1 vs M2 chứng minh.
- **Không** báo M1 thắng nếu H1 không có CI95 dưới > 0; không dùng H2/H3 để thay H1.
- **Không** dùng số validation của Exp11 để tuyên bố vượt `85,49 ± 0,30 %`. Lần mở test K8.3 đã bị ABORTED; mọi lần mở test mới đòi hỏi một protocol deviation được duyệt, manifest bất biến mới và giữ nguyên hồ sơ lần hủy.
- **Không** quét `σ`, `τ`, bán kính, `α_init` hoặc checkpoint sau khi thấy kết quả. Một cấu hình đã khóa.
- Kết quả âm tính vẫn là evidence: ceiling nhỏ (không route nào hợp lệ) hoặc M1 không hơn B1/M2 đều là phát hiện công bố được, nếu báo cáo kèm CI.

## 8. Quy tắc dừng và hướng viết paper song song

1. Nếu D1 ra `STOP_86_UNREACHABLE_BY_DECODER`: dừng Exp11, không huấn luyện SC-LLR. Viết paper theo evidence đã có (xem `README_Q2_PAPER_UPDATE_GUIDE.md`).
2. Nếu M1 − B0 là `INCONCLUSIVE` hoặc `FAIL` sau 5 seed: dừng hướng decoder/LLR. Không chạy thêm seed hay đổi siêu tham số để "tìm" 86 %.
3. Mục tiêu 86 % không phải điều kiện của một bài báo. Evidence hiện có đã gồm: cải thiện Small-Weed Recall của OccCond-g (+10,55 pp so với L0, CI95 `[6,39; 14,84]`) và MCSD-g (+7,84 pp) với chi phí tính toán nhỏ, các đối chứng PS/OccCond, audit marginal, và hai kết quả âm tính được ghi trung thực (CCMC-SR, residual 2×2). Cần quyết định sớm biến thể nào là phương pháp chính, trước khi khóa cấu hình và quyết định có xin mở test hay không.

## 9. Việc cần xác minh trước khi viết Related Work cho SC-LLR

Đọc toán tử (không chỉ abstract) của: PAC, PPAC, semantic lattice, SAPA/ReSFU, các upsampler dựa trên guided filter, FG-UNet, VISA/BAWSeg, ASVLB-Net (NDVI-guided fusion). Ghi truy vấn tìm kiếm và kết quả vào nhật ký như mục 6.4 của REVIEW; có truy vấn đối chứng dương (một công trình chắc chắn tồn tại) để chứng minh cách tìm đủ nhạy.

## 10. Checklist review trước khi chạy

- [ ] Ba file chẩn đoán đã commit và test pass (P0a, P1a).
- [ ] Ngưỡng ở E11-00 đã được người phụ trách chấp nhận **trước** khi chạy D0; `e11_preregistration.sha256` đã lưu.
- [ ] Đã Add Input kết quả Exp10 chứa 15 checkpoint E10B0/B1/M1 (seed 0–4).
- [ ] Cổng tái lập D0b khớp mIoU Exp10 (sai số ≤ 1e-3).
- [ ] Chỉ khi D1 cho route A hợp lệ mới commit `sc_llr.py` và chạy P0b trở đi.
- [ ] Không có cell nào mở test hoặc chạm các artifact của Exp08 (`test_started.json`, manifest, closeout).

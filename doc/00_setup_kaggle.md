# Exp00 — Setup Kaggle (chạy đầu mỗi phiên)

**Trước khi chạy:** commit và push code hiện tại lên `https://github.com/ThaoTo2806/WeedsGalore_Ipr28` (nhánh `main`). Kaggle sẽ clone đúng commit này; `run_config.json` sẽ ghi lại `git_commit`.

Kaggle settings: **Accelerator = GPU T4** (dùng 1 GPU), **Internet = On** (cần cho `git clone` và tải `nvidia/mit-b0`). Nên thêm dataset WeedsGalore vào notebook (Add Input) nếu bạn đã có bản trên Kaggle.

---

### Cell S1 — Kiểm tra GPU

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch, sys, platform
print(platform.python_version(), torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

### Cell S2 — Clone repo

```python
%cd /kaggle/working
!rm -rf WeedsGalore_Ipr28
!git clone -q https://github.com/ThaoTo2806/WeedsGalore_Ipr28.git
%cd /kaggle/working/WeedsGalore_Ipr28
!git log --oneline -1
REPO = '/kaggle/working/WeedsGalore_Ipr28'
```

### Cell S3 — Cài thư viện (phiên bản được pin)

```python
!pip install -q "transformers==4.45.2" torchmetrics absl-py opencv-python-headless pytest scikit-learn thop
import transformers, torchmetrics, sklearn
print('transformers', transformers.__version__, '| torchmetrics', torchmetrics.__version__, '| sklearn', sklearn.__version__)
```

> Nếu pip báo xung đột phụ thuộc với gói khác của Kaggle thì có thể bỏ qua, miễn là dòng `print` in ra `transformers 4.45.2`.

### Cell S4 — Dataset

```python
import glob, os
# 1) Ưu tiên dataset đã Add Input (thư mục chứa các ngày 2023-05-25, ...)
hits = sorted(glob.glob('/kaggle/input/**/2023-05-25/semantics', recursive=True))
if hits:
    DS = os.path.dirname(os.path.dirname(hits[0]))
elif os.path.isdir('/kaggle/working/weedsgalore-dataset/2023-05-25'):
    DS = '/kaggle/working/weedsgalore-dataset'
else:
    # 2) Nếu chưa có: tải theo cách bạn đã dùng trước đây vào /kaggle/working/weedsgalore-dataset
    #    (trang dữ liệu: https://doidata.gfz.de/weedsgalore_e_celikkan_2024/)
    raise SystemExit('Chưa tìm thấy dataset: hãy Add Input hoặc tải về /kaggle/working/weedsgalore-dataset')
print('DS =', DS)
for d in sorted(os.listdir(DS)):
    p = os.path.join(DS, d)
    if os.path.isdir(p):
        print(d, sorted(os.listdir(p)))
```

Kết quả cần thấy: mỗi ngày có `images`, `semantics` và (nếu có) `instances`.

### Cell S5 — Kiểm tra API HuggingFace (kế hoạch mục 8.4)

```python
import inspect, os, sys
REPO = '/kaggle/working/WeedsGalore_Ipr28'
os.chdir(REPO)
from transformers.models.segformer.modeling_segformer import (
    SegformerEncoder, SegformerEfficientSelfAttention, SegformerOverlapPatchEmbeddings)
src_attn = inspect.getsource(SegformerEfficientSelfAttention.forward)
src_enc = inspect.getsource(SegformerEncoder.forward)
checks = {
    'Q computed before SR (query on un-reduced hidden_states)':
        src_attn.find('self.query(hidden_states)') < src_attn.find('self.sr('),
    'SR conv called on 4D tensor then layer_norm': 'self.sr(hidden_states)' in src_attn and 'self.layer_norm(hidden_states)' in src_attn,
    'encoder loop: embedding -> blocks -> norm -> reshape': all(s in src_enc for s in ['embedding_layer(hidden_states)', 'blk(hidden_states, height, width', 'norm_layer(hidden_states)']),
}
for k, v in checks.items():
    print('OK ' if v else 'FAIL', k)
os.makedirs('results/env', exist_ok=True)
with open('results/env/hf_source_check.txt', 'w') as f:
    f.write(src_attn + '\n\n' + src_enc)

import sys
if f'{REPO}/src' not in sys.path:
    sys.path.insert(0, f'{REPO}/src')
assert os.path.exists(f'{REPO}/src/nets/ccmc_sr.py'), 'Repo chưa có code mới: commit + push rồi chạy lại S2'
from nets.ccmc_sr import CCMCSegFormer
m = CCMCSegFormer(prior_mode='learned_weed')
for name, mod in m.named_modules():
    if name.endswith('.sr'):
        print(name, type(mod).__name__, tuple(mod.weight.shape), 'stride', mod.stride)
assert all(checks.values()), 'API HF khác giả định -> dừng lại, gửi file results/env/hf_source_check.txt để review'
```

### Cell S6 — Unit test (bắt buộc pass trước mọi thí nghiệm)

```python
%cd /kaggle/working/WeedsGalore_Ipr28
!mkdir -p results/env
!python -m pytest -q tests/test_ccmc_sr.py 2>&1 | tee results/env/unit_tests.log | tail -n 15
```

Kết quả cần thấy: `passed`, không có `failed`, không có `skipped` (nếu bị skip thì nghĩa là chưa tải được `nvidia/mit-b0`, cần bật Internet).

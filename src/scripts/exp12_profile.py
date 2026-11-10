"""E12.0 model resource probe: synthetic input, no train/val/test labels are touched.
Use a Kaggle T4, with transformers pretrained weights accessible.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nets import SegFormer5Band

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--backbone',choices=['b0','b1','b2'],required=True)
 ap.add_argument('--size',type=int,choices=[600,900],default=600)
 ap.add_argument('--batch_size',type=int,default=1)
 ap.add_argument('--steps',type=int,default=3)
 ap.add_argument('--out',default='')
 a=ap.parse_args()
 if not torch.cuda.is_available():raise SystemExit('GPU not found; E12.0 GPU profile requires CUDA')
 device='cuda'
 try:
  model=SegFormer5Band(num_classes=3,pretrained_backbone=True,in_channels=5,
    use_low_level_refine=True,use_multiscale_fusion=True,llr_source='raw',backbone=a.backbone).to(device)
 except Exception as e:
  raise SystemExit(f'Model load failed ({type(e).__name__}): {e}')
 params=sum(x.numel() for x in model.parameters())
 optim=torch.optim.Adam(model.parameters(),lr=3e-4)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 durations=[]
 try:
  for i in range(a.steps):
   torch.cuda.synchronize();start=time.perf_counter()
   model.train()
   x=torch.randn(a.batch_size,5,a.size,a.size,device=device)
   logits=model(x)
   assert logits.shape==(a.batch_size,3,a.size,a.size),logits.shape
   loss=logits.square().mean()
   loss.backward();optim.step();optim.zero_grad(set_to_none=True)
   torch.cuda.synchronize();durations.append(time.perf_counter()-start)
 except torch.cuda.OutOfMemoryError:
  print('OOM: profile failure. Do NOT alter batch/architecture without protocol registration.')
  raise
 report={'backbone':a.backbone,'size':a.size,'batch_size':a.batch_size,
   'steps':a.steps,'mean_step_s_excluding_first':sum(durations[1:])/max(len(durations)-1,1) if len(durations)>1 else durations[0],
   'max_vram_gb':torch.cuda.max_memory_allocated()/1e9,'params':params,
   'device':torch.cuda.get_device_name(0),'synthetic_only':True}
 print(json.dumps(report,indent=2))
 if a.out:
  o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
  o.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()

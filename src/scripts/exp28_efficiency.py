"""Measure one-checkpoint parameter count and GPU latency under fixed input size.
Model includes B2+LLR; D4 is 8 forward passes, not an ensemble.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from scripts.exp22_run import load_base,tta_pairs

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--ckpt',required=True)
 p.add_argument('--out_json',required=True)
 p.add_argument('--height',type=int,default=600)
 p.add_argument('--width',type=int,default=600)
 p.add_argument('--warmup',type=int,default=10)
 p.add_argument('--runs',type=int,default=30)
 a=p.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda':raise RuntimeError('Benchmark requires CUDA for comparable GPU latency')
 model=load_base(a.ckpt,device).eval()
 x=torch.zeros(1,5,a.height,a.width,device=device)
 def run(tta):
  with torch.inference_mode():
   acc=None;n=0
   for xt,undo in tta_pairs(x,tta):
    z=undo(model(xt).float())
    acc=z if acc is None else acc+z;n+=1
   return (acc/n).argmax(1)
 report={'checkpoint':a.ckpt,'params':sum(p.numel() for p in model.parameters()),
         'trainable_params_if_unfrozen':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'device':torch.cuda.get_device_name(0),'size':[a.height,a.width],'latency_ms':{}}
 for tta in ('none','d4'):
  for _ in range(a.warmup):run(tta)
  torch.cuda.synchronize();times=[]
  for _ in range(a.runs):
   torch.cuda.synchronize();start=time.perf_counter()
   run(tta);torch.cuda.synchronize()
   times.append((time.perf_counter()-start)*1000)
  report['latency_ms'][tta]={'mean':sum(times)/len(times),'median':sorted(times)[len(times)//2],
                             'p95':sorted(times)[int(.95*(len(times)-1))]}
 report['note']='Latency includes model, D4 transformations/aggregation and argmax. No checkpoint ensemble.'
 out=Path(a.out_json);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
 print('EXP28 EFFICIENCY PASS',json.dumps(report,indent=2))
if __name__=='__main__':main()

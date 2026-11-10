import json, pathlib, sys, numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from exp12_config import expected_flags, verify_flags, verify_eval, load_json
from exp12_sched import lr_factor, ema_decay_at
from scripts.exp12_preflight import inspect
C=load_json(ROOT/'configs/exp12/exp12_common_config.json')
A=load_json(ROOT/'configs/exp12/exp12_arms.json')
def test_preflight(): assert not inspect(C,A)
def test_science_contrast_is_explicit():
 assert A['arms']['S1_B0']['flags']['use_low_level_refine'] is False
 assert A['arms']['L12_B0_OCC2']['flags']['use_low_level_refine'] is True
 assert A['arms']['L12_B0_OCC2']['flags']['head']=='occ_conditioned2'
def test_common_flags_conflict():
 assert not (set(C['common']) & set(A['arms']['P_B0_LLR']['flags']))
def test_lr_schedule():
 assert lr_factor(0,100,10,'cosine',.01)>0
 assert lr_factor(100,100,10,'cosine',.01)==.01
 assert ema_decay_at(100,.99)<=.99
def test_eval_mismatched_backbone_rejected():
 assert verify_eval({'backbone':'b2','tta':'none'},A,'P_B0_LLR')
def test_eval_registered_flags_pass():
 flags=expected_flags(C,A,'P_B0_LLR');flags.update(tta='none')
 assert not verify_eval(flags,A,'P_B0_LLR')
def test_run_flags_validate():
 flags=expected_flags(C,A,'P_B0_LLR');flags['seed']=0
 assert not verify_flags(flags,C,A,'P_B0_LLR')
 flags['weight_weed']=2.0
 assert verify_flags(flags,C,A,'P_B0_LLR')

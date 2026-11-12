import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from nets.exp22_bandit import BanditPolicy,action_logits,region_features,reward_table,reinforce_loss

def test_shapes_and_noop():
    torch.manual_seed(0)
    x=torch.rand(1,5,48,48)
    logits=torch.randn(1,3,48,48)
    f=region_features(x,logits,tile=24)
    assert f.shape==(4,14)
    keep=action_logits(logits,torch.zeros(4,dtype=torch.long),tile=24)
    assert torch.equal(keep,logits)

def test_reward_train_only_and_actions():
    logits=torch.zeros(1,3,48,48); logits[:,1]=0.1
    label=torch.full((1,48,48),1,dtype=torch.long)
    reward=reward_table(logits,label,tile=24)
    assert reward.shape==(4,5)
    assert torch.allclose(reward[:,0],torch.zeros(4))
    policy=BanditPolicy()
    loss,r=reinforce_loss(policy(torch.rand(4,14)),reward)
    assert torch.isfinite(loss) and torch.isfinite(r)
    loss.backward()
    assert policy.net[-1].weight.grad is not None

def test_correction_and_harm():
    logits=torch.zeros(1,3,24,24)
    logits[:,1]=0.25; logits[:,2]=0.0
    label=torch.full((1,24,24),2,dtype=torch.long)
    r=reward_table(logits,label,tile=24)
    assert r[0,3]>0 and r[0,0]==0
    assert action_logits(logits,torch.tensor([3]),tile=24).argmax(1).eq(2).all()

def test_d4_inverse():
    from nets.exp22_bandit import tta_pairs
    x=torch.rand(1,5,48,48)
    for y,undo in tta_pairs(x,'d4'):
        assert torch.allclose(undo(y),x)

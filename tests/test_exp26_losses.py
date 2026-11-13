import torch
from exp26_losses import foreground_edge,boundary_ce,dice_loss

def test_edge_focus():
    y=torch.zeros(1,20,20,dtype=torch.long)
    y[:,5:15,5:15]=1
    edge=foreground_edge(y,2)
    assert edge.sum()>0
    assert edge[0,10,10]==0
    assert edge[0,5,5]==1

def test_gradient():
    z=torch.randn(1,3,20,20,requires_grad=True)
    y=torch.zeros(1,20,20,dtype=torch.long);y[:,5:15,5:15]=2
    l=boundary_ce(z,y,torch.tensor([.35,1.,1.5]))+dice_loss(z,y)
    l.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

def test_ignore_mask():
    z=torch.randn(1,3,20,20,requires_grad=True)
    y=torch.full((1,20,20),-1,dtype=torch.long)
    assert boundary_ce(z,y).item()==0

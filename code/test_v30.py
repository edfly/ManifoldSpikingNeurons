"""
Functional tests for v30 Manifold Spiking Neuron experiments (compact version).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch, numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

passed = 0; failed = 0

def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1; print(f"  [PASS] {msg}")
    else:
        failed += 1; print(f"  [FAIL] {msg}")

ROOT = "data/TU"

def t1_data():
    print("\n=== Test 1: Data Loading ===")
    for name in ["MUTAG", "PROTEINS", "DD", "ENZYMES"]:
        ds = TUDataset(root=ROOT, name=name)
        check(len(ds)>0, f"{name}: {len(ds)} graphs, {ds.num_classes} cls, {ds.num_features} feats")

def t2_manifold():
    print("\n=== Test 2: ProductManifold ===")
    from spikergnn.manifolds import ProductManifold
    m = ProductManifold(d_h=8, d_e=8, c=1.0)
    p = m.hyp.expmap0(torch.randn(4, 8)*0.1)
    check(p.norm(dim=-1).max()<1.0, f"Expmap0 inside ball: max={p.norm(dim=-1).max():.3f}")
    d = m.hyp.dist(p[:2], p[2:])
    check((d>=0).all(), f"Dist positive: {d}")

def t3_neuron():
    print("\n=== Test 3: ManifoldLIFNeuron ===")
    from spikergnn.neurons import ManifoldLIFNeuron
    from spikergnn.manifolds import ProductManifold
    m = ProductManifold(d_h=8, d_e=8, c=1.0)
    n = ManifoldLIFNeuron(manifold=m, surrogate_type="atan", ablation="full")
    N,T,D = 16,10,16
    x = torch.randn(N,T,D)*0.1
    p0=(m.hyp.expmap0(torch.randn(N,8)*0.05), torch.randn(N,8)*0.1)
    s,p = n(x,p0)
    check(s.shape==(N,T,D), f"Spike shape: {s.shape}")
    check(0<=n._spike_rate<=1, f"Spike rate: {n._spike_rate:.3f}")
    # no_spikes variant
    n2 = ManifoldLIFNeuron(manifold=m, ablation="no_spikes")
    s2,_=n2(x,p0)
    check(torch.allclose(s2,torch.ones_like(s2)), "no_spikes: all ones")
    # Fixed threshold
    n3 = ManifoldLIFNeuron(manifold=m, ablation="full")
    n3.u_th_learnable.requires_grad_(False)
    s3,_=n3(x,p0)
    check(not n3.u_th_learnable.requires_grad, "Fixed threshold: not learnable")

def t4_model():
    print("\n=== Test 4: Model Build ===")
    from experiment_v30 import ManifoldSpikingModel, NeuronVariant
    for v in [NeuronVariant.ON_MANIFOLD, NeuronVariant.NO_SPIKES]:
        m = ManifoldSpikingModel(in_dim=7, hidden_dim=32, h_dim=8, e_dim=8,
                                 num_classes=2, num_layers=2, T=5, neuron_variant=v)
        n = sum(p.numel() for p in m.parameters())
        check(n>0, f"{v.value}: {n:,} params")
        g1=Data(x=torch.randn(5,7),edge_index=torch.randint(0,5,(2,10)))
        g2=Data(x=torch.randn(3,7),edge_index=torch.randint(0,3,(2,6)))
        batch=Batch.from_data_list([g1,g2])
        logits,sr=m(batch)
        check(logits.shape==(2,2), f"{v.value}: logits {logits.shape}")

def t5_train():
    print("\n=== Test 5: Training Loop ===")
    from experiment_v30 import (ManifoldSpikingModel, NeuronVariant,
                                 train_epoch, evaluate)
    ds = TUDataset(root=ROOT, name="MUTAG")
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = ManifoldSpikingModel(in_dim=ds.num_features, hidden_dim=32,
        h_dim=8, e_dim=8, num_classes=ds.num_classes, num_layers=2, T=5,
        neuron_variant=NeuronVariant.ON_MANIFOLD)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.CrossEntropyLoss()
    loss, acc, sr = train_epoch(model, loader, opt, crit, lambda_sr=0.5, r_target=0.4)
    check(not np.isnan(loss), f"Loss: {loss:.4f}")
    check(0<=acc<=100, f"Acc: {acc:.1f}%")
    check(0<=sr<=1, f"SR: {sr:.3f}")
    print(f"  Loss={loss:.4f}, Acc={acc:.1f}%, SR={sr:.3f}")
    eloss, eacc, esr = evaluate(model, loader)
    print(f"  Eval: Loss={eloss:.4f}, Acc={eacc:.1f}%, SR={esr:.3f}")

def t6_cv():
    print("\n=== Test 6: CV Runner ===")
    from experiment_v30 import run_single, NeuronVariant
    r = run_single(dataset_name="MUTAG", neuron_variant=NeuronVariant.ON_MANIFOLD,
                   n_folds=3, epochs=5, batch_size=16, verbose=False)
    check(r["mean_acc"]>0, f"CV acc: {r['mean_acc']:.1f}%+/-{r['std_acc']:.1f}%")
    print(f"  3-fold Acc={r['mean_acc']:.1f}%+/-{r['std_acc']:.1f}%, SR={r['mean_sr']:.3f}")

if __name__=="__main__":
    print("="*60)
    print("V30 Functional Tests")
    print("="*60)
    t1_data(); t2_manifold(); t3_neuron(); t4_model(); t5_train(); t6_cv()
    print("\n"+"="*60)
    print(f"{passed} passed, {failed} failed")
    if failed>0: sys.exit(1)
    print("ALL TESTS PASSED!")

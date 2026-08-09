# -*- coding: utf-8 -*-
"""
PoC 2a — CONTROLE SINTETICO: a atencao consegue extrair interacao par-a-par
quando ela comprovadamente existe?

Sobre as escalacoes REAIS, geramos dois conjuntos de rotulos sinteticos:
  ADD: resultado depende so da media de overall (sem interacao)  -> sanidade
  INT: resultado depende de sinergias passe_i x finalizacao_j entre
       jogadores ADJACENTES na formacao (interacao pura, nao separavel
       em medias de time)
Modelos: LR em medias | Deep Sets (tokens, SEM atencao) | Transformer (COM atencao).
Expectativa: em ADD todos empatam; em INT so o Transformer recupera o sinal.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0)
np.random.seed(0)

d = np.load("dataset2.npz", allow_pickle=True)
X, coords, team, season = d["X"], d["coords"], d["team"], d["season"]
feat_names = list(d["feat_names"])
N = len(X)

io = feat_names.index("overall_rating")
ip = feat_names.index("short_passing")
if_ = feat_names.index("finishing")
z = lambda a: (a - a.mean()) / (a.std() + 1e-8)
ovr, pas, fin = z(X[:, :, io]), z(X[:, :, ip]), z(X[:, :, if_])

# adjacencia: distancia euclidiana na grade da formacao <= 3, dentro do time
adj = np.zeros((N, 22, 22), np.float32)
for t0 in (0, 11):
    c = coords[:, t0:t0 + 11]
    dist = np.linalg.norm(c[:, :, None] - c[:, None, :], axis=-1)
    a = (dist <= 3.0).astype(np.float32)
    a[:, range(11), range(11)] = 0
    adj[:, t0:t0 + 11, t0:t0 + 11] = a
print(f"pares adjacentes por time (media): {adj[:, :11, :11].sum((1, 2)).mean():.1f}")

def synergy(t0):
    a = adj[:, t0:t0 + 11, t0:t0 + 11]
    prod = pas[:, t0:t0 + 11, None] * fin[:, None, t0:t0 + 11]
    return (prod * a).sum((1, 2)) / (a.sum((1, 2)) + 1e-8)

A = z(ovr[:, :11].mean(1) - ovr[:, 11:].mean(1))
I = z(synergy(0) - synergy(11))
print(f"correlacao A x I: {np.corrcoef(A, I)[0, 1]:.3f}")

rng = np.random.default_rng(0)
noise = rng.normal(0, 0.5, N)
labels = {
    "ADD (sem interacao)": (A + noise > 0).astype(np.int64),
    "INT (interacao pura)": (I + noise > 0).astype(np.int64),
}

VAL, TEST = "2014/2015", "2015/2016"
tr = ~np.isin(season, [VAL, TEST]); va = season == VAL; te = season == TEST

# tokens minimos: [ovr, passe, fin, X/10, Y/11]
tok = np.stack([ovr, pas, fin, coords[..., 0] / 10.0, coords[..., 1] / 11.0], -1).astype(np.float32)
agg = np.concatenate([tok[:, :11, :3].mean(1), tok[:, 11:, :3].mean(1)], 1)  # medias p/ LR

class TokenEnc(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = nn.Linear(5, dim)
        self.team_emb = nn.Embedding(2, dim)
    def forward(self, x, t):
        return self.proj(x) + self.team_emb(t)

class DeepSets(nn.Module):  # sem atencao: cada token processado isoladamente
    def __init__(self, dim=32):
        super().__init__()
        self.enc = TokenEnc(dim)
        self.phi = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                 nn.Linear(dim, dim), nn.ReLU())
        self.head = nn.Linear(dim, 1)
    def forward(self, x, t):
        return self.head(self.phi(self.enc(x, t)).mean(1)).squeeze(-1)

class Former(nn.Module):    # com atencao entre os 22 tokens
    def __init__(self, dim=32):
        super().__init__()
        self.enc = TokenEnc(dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, 0.1,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, 1)
    def forward(self, x, t):
        return self.head(self.tf(self.enc(x, t)).mean(1)).squeeze(-1)

def train_nn(model, ytr_np, yva_np, yte_np):
    Xtr = torch.tensor(tok[tr]); Xva = torch.tensor(tok[va]); Xte = torch.tensor(tok[te])
    Ttr = torch.tensor(team[tr]); Tva = torch.tensor(team[va]); Tte = torch.tensor(team[te])
    ytr = torch.tensor(ytr_np, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    lf = nn.BCEWithLogitsLoss()
    best, state, pat = 0, None, 0
    for ep in range(30):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lf(model(Xtr[b], Ttr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = ((model(Xva, Tva) > 0).numpy() == yva_np).mean()
        if acc > best + 1e-4:
            best, state, pat = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 4:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return ((model(Xte, Tte) > 0).numpy() == yte_np).mean()

print(f"\n{'cenario':22s} {'LR medias':>10s} {'DeepSets':>10s} {'Transformer':>12s}")
for name, yy in labels.items():
    lr = LogisticRegression(max_iter=1000).fit(agg[tr], yy[tr])
    a_lr = (lr.predict(agg[te]) == yy[te]).mean()
    a_ds = train_nn(DeepSets(), yy[tr], yy[va], yy[te])
    a_tf = train_nn(Former(), yy[tr], yy[va], yy[te])
    print(f"{name:22s} {a_lr:10.4f} {a_ds:10.4f} {a_tf:12.4f}")

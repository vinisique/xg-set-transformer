# -*- coding: utf-8 -*-
"""
PoC 2b — DADOS REAIS com sinais de interacao enriquecidos.

Tokens: 38 atributos FIFA + posicao X,Y na formacao + entrosamento (coplay)
        + forma recente do time (+ embedding residual por ID de jogador).
Ablacao (2 seeds cada):
  DS   Deep Sets (mesmos tokens, SEM atencao)     -> sem interacao par-a-par
  TF-  Transformer SEM embedding de ID            -> interacao so via atributos
  TF+  Transformer COM embedding de ID            -> interacao + "quimica" latente
O delta TF- vs DS mede o valor da atencao; TF+ vs TF- mede o valor do ID.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss

d = np.load("dataset2.npz", allow_pickle=True)
X, coords, coplay, form = d["X"], d["coords"], d["coplay"], d["form"]
pidx, role, team, y, season, odds = (d["pidx"], d["role"], d["team"],
                                     d["y"], d["season"], d["odds"])
n_players = int(d["n_players"])
N, T, F = X.shape

VAL, TEST = "2014/2015", "2015/2016"
tr = ~np.isin(season, [VAL, TEST]); va = season == VAL; te = season == TEST
print(f"{N} partidas | treino {tr.sum()} val {va.sum()} teste {te.sum()}")

# monta tokens enriquecidos e normaliza com estatisticas do TREINO
extra = np.concatenate(
    [coords / [10.0, 11.0],
     np.log1p(coplay)[..., None],
     (form / 15.0)[..., None]], axis=-1).astype(np.float32)
tok = np.concatenate([X, extra], -1)
Fe = tok.shape[-1]
mu = tok[tr].reshape(-1, Fe).mean(0)
sd = tok[tr].reshape(-1, Fe).std(0) + 1e-8
tok = ((tok - mu) / sd).astype(np.float32)
print(f"tokens: {Fe} features")

def tens(a, dt=torch.float32):
    return torch.tensor(a, dtype=dt)

SPLITS = {}
for name, mask in (("tr", tr), ("va", va), ("te", te)):
    SPLITS[name] = dict(x=tens(tok[mask]), p=tens(pidx[mask], torch.long),
                        t=tens(team[mask], torch.long), r=tens(role[mask], torch.long),
                        y=tens(y[mask], torch.long))

class Tokens(nn.Module):
    def __init__(self, dim, use_id):
        super().__init__()
        self.proj = nn.Linear(Fe, dim)
        self.team_emb = nn.Embedding(2, dim)
        self.role_emb = nn.Embedding(4, dim)
        self.use_id = use_id
        if use_id:
            self.id_emb = nn.Embedding(n_players, dim)
            nn.init.normal_(self.id_emb.weight, 0, 0.02)
    def forward(self, x, p, t, r):
        h = self.proj(x) + self.team_emb(t) + self.role_emb(r)
        if self.use_id:
            h = h + self.id_emb(p)
        return h

class DeepSets(nn.Module):
    def __init__(self, dim=64, use_id=False, drop=0.25):
        super().__init__()
        self.tok = Tokens(dim, use_id)
        self.phi = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(dim * 2, dim), nn.ReLU())
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, 3))
    def forward(self, x, p, t, r):
        return self.head(self.phi(self.tok(x, p, t, r)).mean(1))

class Former(nn.Module):
    def __init__(self, dim=64, use_id=False, drop=0.25):
        super().__init__()
        self.tok = Tokens(dim, use_id)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, drop,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, 3))
    def forward(self, x, p, t, r):
        h = self.tok(x, p, t, r)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        return self.head(self.tf(h)[:, 0])

def run(model, seed):
    torch.manual_seed(seed)
    for m in model.modules():  # re-inicializa com a seed
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    if getattr(model.tok, "use_id", False):
        nn.init.normal_(model.tok.id_emb.weight, 0, 0.02)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
    lf = nn.CrossEntropyLoss()
    S = SPLITS
    best, state, pat = 9e9, None, 0
    for ep in range(40):
        model.train()
        perm = torch.randperm(len(S["tr"]["x"]))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lf(model(S["tr"]["x"][b], S["tr"]["p"][b],
                            S["tr"]["t"][b], S["tr"]["r"][b]), S["tr"]["y"][b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(S["va"]["x"], S["va"]["p"],
                                     S["va"]["t"], S["va"]["r"]), 1).numpy()
        ll = log_loss(y[va], pv, labels=[0, 1, 2])
        if ll < best - 1e-4:
            best, state, pat = ll, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pt = torch.softmax(model(S["te"]["x"], S["te"]["p"],
                                 S["te"]["t"], S["te"]["r"]), 1).numpy()
    return accuracy_score(y[te], pt.argmax(1)), log_loss(y[te], pt, labels=[0, 1, 2])

configs = [
    ("DS  DeepSets (sem atencao)", lambda: DeepSets(use_id=False)),
    ("TF- Transformer sem ID",     lambda: Former(use_id=False)),
    ("TF+ Transformer com ID",     lambda: Former(use_id=True)),
]
print(f"\n{'modelo':30s} {'acc (2 seeds)':>16s} {'logloss':>14s}")
for name, mk in configs:
    accs, lls = [], []
    for seed in (0, 1):
        a, l = run(mk(), seed)
        accs.append(a); lls.append(l)
    print(f"{name:30s} {np.mean(accs):8.4f} ±{np.std(accs):.4f} {np.mean(lls):9.4f} ±{np.std(lls):.4f}")

ok = ~np.isnan(odds[te]).any(1)
inv = 1.0 / odds[te][ok]
p_odds = inv / inv.sum(1, keepdims=True)
print(f"{'ODD Bet365 (teto)':30s} {accuracy_score(y[te][ok], p_odds.argmax(1)):8.4f}"
      f"          {log_loss(y[te][ok], p_odds, labels=[0,1,2]):9.4f}")

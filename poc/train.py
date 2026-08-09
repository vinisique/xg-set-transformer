# -*- coding: utf-8 -*-
"""
PoC — Transformer jogador-token vs. baselines na previsao de resultado (1X2).

Escada de comparacao:
  B0  mandante sempre vence
  B1  regressao logistica: diferenca da media de overall_rating
  B2  regressao logistica: medias de todas as features por time (diferenca)
  B3  MLP nas features agregadas por time (aula 02)
  T   Transformer com 22 tokens-jogador + CLS (sem positional encoding)
  ODD odds da Bet365 como referencia de teto informacional
Split temporal: treino <= 2013/2014, val 2014/2015, teste 2015/2016.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)
np.random.seed(0)

d = np.load("dataset.npz", allow_pickle=True)
X, role, team, y, season, odds = d["X"], d["role"], d["team"], d["y"], d["season"], d["odds"]
feat_names = list(d["feat_names"])
N, T, F = X.shape
print(f"{N} partidas, {T} tokens, {F} features | classes {np.bincount(y)}")

VAL, TEST = "2014/2015", "2015/2016"
tr = ~np.isin(season, [VAL, TEST])
va = season == VAL
te = season == TEST
print(f"treino {tr.sum()} | val {va.sum()} | teste {te.sum()}")

# normalizacao por feature ajustada SO no treino
mu = X[tr].reshape(-1, F).mean(0)
sd = X[tr].reshape(-1, F).std(0) + 1e-8
Xn = (X - mu) / sd

results = {}

def report(name, p_test):
    acc = accuracy_score(y[te], p_test.argmax(1))
    ll = log_loss(y[te], p_test, labels=[0, 1, 2])
    results[name] = (acc, ll)
    print(f"{name:28s}  acc={acc:.4f}  logloss={ll:.4f}")

# ---------- B0: mandante sempre vence ----------
p = np.zeros((te.sum(), 3)) + 1e-9
p[:, 0] = 1
report("B0 mandante-vence", p / p.sum(1, keepdims=True))

# ---------- B1: logistic na diferenca de overall_rating medio ----------
i_ovr = feat_names.index("overall_rating")
home_ovr = X[:, :11, i_ovr].mean(1)
away_ovr = X[:, 11:, i_ovr].mean(1)
z = (home_ovr - away_ovr).reshape(-1, 1)
lr = LogisticRegression(max_iter=1000).fit(z[tr], y[tr])
report("B1 logistic (diff rating)", lr.predict_proba(z[te]))

# ---------- B2: logistic nas medias de todas as features ----------
agg = np.concatenate([Xn[:, :11].mean(1), Xn[:, 11:].mean(1)], axis=1)
lr2 = LogisticRegression(max_iter=2000, C=0.1).fit(agg[tr], y[tr])
report("B2 logistic (agregado)", lr2.predict_proba(agg[te]))

# ---------- B3: MLP agregado (aula 02) ----------
sc = StandardScaler().fit(agg[tr])
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                    early_stopping=True, random_state=0).fit(sc.transform(agg[tr]), y[tr])
report("B3 MLP (agregado)", mlp.predict_proba(sc.transform(agg[te])))

# ---------- T: Transformer jogador-token ----------
class LineupTransformer(nn.Module):
    def __init__(self, n_feat, dim=64, heads=4, layers=2, drop=0.25):
        super().__init__()
        self.proj = nn.Linear(n_feat, dim)
        self.team_emb = nn.Embedding(2, dim)
        self.role_emb = nn.Embedding(4, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, 3))

    def forward(self, x, team, role):
        h = self.proj(x) + self.team_emb(team) + self.role_emb(role)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)  # sem positional encoding: escalacao e conjunto
        return self.head(self.encoder(h)[:, 0])

def tt(a, dt=torch.float32):
    return torch.tensor(a, dtype=dt)

Xtr, Xva, Xte = tt(Xn[tr]), tt(Xn[va]), tt(Xn[te])
Ttr, Tva, Tte = tt(team[tr], torch.long), tt(team[va], torch.long), tt(team[te], torch.long)
Rtr, Rva, Rte = tt(role[tr], torch.long), tt(role[va], torch.long), tt(role[te], torch.long)
ytr, yva = tt(y[tr], torch.long), tt(y[va], torch.long)

model = LineupTransformer(F)
n_par = sum(p.numel() for p in model.parameters())
print(f"Transformer: {n_par:,} parametros")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
lossf = nn.CrossEntropyLoss()

best_ll, best_state, patience = 9e9, None, 0
BS = 256
for epoch in range(40):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), BS):
        b = perm[i:i + BS]
        opt.zero_grad()
        loss = lossf(model(Xtr[b], Ttr[b], Rtr[b]), ytr[b])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(Xva, Tva, Rva), 1).numpy()
    ll = log_loss(y[va], pv, labels=[0, 1, 2])
    acc = accuracy_score(y[va], pv.argmax(1))
    print(f"  epoca {epoch+1:02d}  val_logloss={ll:.4f}  val_acc={acc:.4f}")
    if ll < best_ll - 1e-4:
        best_ll, best_state, patience = ll, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 5:
            print("  early stopping")
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pt = torch.softmax(model(Xte, Tte, Rte), 1).numpy()
report("T  Transformer jogador-token", pt)

# ---------- ODD: teto informacional (Bet365) ----------
ok = ~np.isnan(odds[te]).any(1)
inv = 1.0 / odds[te][ok]
p_odds = inv / inv.sum(1, keepdims=True)
acc_o = accuracy_score(y[te][ok], p_odds.argmax(1))
ll_o = log_loss(y[te][ok], p_odds, labels=[0, 1, 2])
print(f"{'ODD Bet365 (teto, n=%d)' % ok.sum():28s}  acc={acc_o:.4f}  logloss={ll_o:.4f}")

print("\n===== RESUMO (teste 2015/2016) =====")
for k, (a, l) in results.items():
    print(f"{k:28s}  acc={a:.4f}  logloss={l:.4f}")
print(f"{'ODD Bet365 (subset c/ odds)':28s}  acc={acc_o:.4f}  logloss={ll_o:.4f}")

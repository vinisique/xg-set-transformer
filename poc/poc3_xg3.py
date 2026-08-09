# -*- coding: utf-8 -*-
"""
PoC 3 — xG com freeze-frames: a atencao sobre jogadores melhora a previsao
de gol quando as posicoes REAIS no instante do chute estao disponiveis?

Escada:
  B1  LR classica (distancia, angulo, cabeceio)          -> xG "de livro"
  B2  LR + features manuais de interacao (GK, bloqueio)  -> interacao feita a mao
  DS  Deep Sets sobre tokens (SEM atencao)
  TF  Transformer sobre tokens (COM atencao)
Descoberta real = TF > DS e TF >= B2 sem features manuais.
Split por partida (70/15/15). Metricas: AUC e logloss. 2 seeds nos NNs.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

d = np.load("shots_all.npz", allow_pickle=True)
sx, sy, goal = d["sx"], d["sy"], d["goal"]
px, py, mate, gk, mask, header = d["px"], d["py"], d["mate"], d["gk"], d["mask"], d["header"]
match_id, comp = d["match_id"], d["comp"]
N, P = px.shape
print(f"{N} chutes | {goal.mean()*100:.1f}% gols | {len(np.unique(match_id))} partidas")

# ---------- geometria (gol em x=120, traves em y=36 e 44) ----------
dx, dy = 120.0 - sx, sy - 40.0
dist = np.hypot(dx, dy)
angle = np.abs(np.arctan2(8.0 * dx, dx**2 + dy**2 - 16.0))

# ---------- split por partida ----------
rng = np.random.default_rng(0)
uniq = np.unique(match_id)
rng.shuffle(uniq)
n1, n2 = int(0.7 * len(uniq)), int(0.85 * len(uniq))
tr = np.isin(match_id, uniq[:n1]); va = np.isin(match_id, uniq[n1:n2]); te = np.isin(match_id, uniq[n2:])
print(f"treino {tr.sum()} | val {va.sum()} | teste {te.sum()}")

results = {}
def report(name, p_te):
    auc = roc_auc_score(goal[te], p_te)
    ll = log_loss(goal[te], np.clip(p_te, 1e-6, 1 - 1e-6))
    results[name] = (auc, ll)
    print(f"{name:34s} AUC={auc:.4f}  logloss={ll:.4f}")

# ---------- B1: xG classico ----------
F1 = np.stack([dist, angle, header], 1)
lr1 = LogisticRegression(max_iter=1000).fit(F1[tr], goal[tr])
report("B1 LR (dist, angulo, cabeca)", lr1.predict_proba(F1[te])[:, 1])

# ---------- B2: + interacao manual ----------
opp = (mate < 0.5) & (mask > 0.5)
gkm = opp & (gk > 0.5)
gk_dist = np.where(gkm, np.hypot(120 - px, py - 40), np.nan)
gk_dist = np.nanmin(np.where(gkm, gk_dist, np.inf), 1)
gk_dist[np.isinf(gk_dist)] = 25.0
# desvio do GK em relacao a linha chutador->centro do gol
gkx = np.where(gkm, px, 0).sum(1) / np.maximum(gkm.sum(1), 1)
gky = np.where(gkm, py, 0).sum(1) / np.maximum(gkm.sum(1), 1)
t = np.clip(((gkx - sx) * dx + (gky - sy) * (40 - sy)) / (dist**2 + 1e-8), 0, 1)
gk_off = np.hypot(sx + t * dx - gkx, sy + t * (40 - sy) - gky)
# bloqueadores: adversarios dentro do triangulo chute->traves
def inside(pxx, pyy):
    d1 = (36 - sy[:, None]) * (pxx - sx[:, None]) - (120 - sx[:, None]) * (pyy - sy[:, None])
    d2 = (44 - sy[:, None]) * (pxx - sx[:, None]) - (120 - sx[:, None]) * (pyy - sy[:, None])
    return (d1 * d2 < 0) & (pxx > sx[:, None])
blockers = (inside(px, py) & opp & (gk < 0.5)).sum(1).astype(np.float32)
opp_d = np.where(opp, np.hypot(px - sx[:, None], py - sy[:, None]), np.inf)
near_opp = np.min(opp_d, 1); near_opp[np.isinf(near_opp)] = 30.0
F2 = np.stack([dist, angle, header, gk_dist, gk_off, blockers, near_opp], 1)
lr2 = LogisticRegression(max_iter=2000).fit(F2[tr], goal[tr])
report("B2 LR + interacao manual", lr2.predict_proba(F2[te])[:, 1])

# ---------- tokens v2: geometria RELATIVA ao chutador e a linha do chute ----------
# (informacao por jogador, nao par-a-par: mantem justa a ablacao DS vs TF)
pdist = np.hypot(120 - px, py - 40)
rdx = (px - sx[:, None]) / 40.0
rdy = (py - sy[:, None]) / 40.0
dshoot = np.hypot(px - sx[:, None], py - sy[:, None])
# projecao na linha chutador->centro do gol e distancia perpendicular a ela
tline = np.clip(((px - sx[:, None]) * dx[:, None] + (py - sy[:, None]) * (40 - sy)[:, None])
                / (dist[:, None] ** 2 + 1e-8), 0, 1)
perp = np.hypot(sx[:, None] + tline * dx[:, None] - px,
                sy[:, None] + tline * (40 - sy)[:, None] - py)
intri = (inside(px, py)).astype(np.float32)

Z = np.zeros(N); ZP = np.zeros((N, P))
shooter = np.stack([sx / 120, sy / 80, dist / 50, angle,
                    Z, Z, Z, Z, Z, Z,
                    np.ones(N), np.zeros(N), np.ones(N), header], 1)[:, None, :]
others = np.stack([px / 120, py / 80, pdist / 50, ZP,
                   rdx, rdy, dshoot / 40.0, tline, perp / 20.0, intri,
                   mate, gk, ZP, ZP], -1)
tok = np.concatenate([shooter, others], 1).astype(np.float32)   # [N, 22, 14]
pad = np.concatenate([np.zeros((N, 1)), 1 - mask], 1) > 0.5     # True = ignorar

class DeepSets(nn.Module):
    def __init__(self, dim=48, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(14, dim)
        self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(dim * 2, dim), nn.ReLU())
        self.head = nn.Linear(dim * 2, 1)   # pool mean+max: permite expressar extremos
    def forward(self, x, pad):
        h = self.phi(self.proj(x))
        w = (~pad).float().unsqueeze(-1)
        mean = (h * w).sum(1) / w.sum(1)
        mx = h.masked_fill(pad.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)

class Former(nn.Module):
    def __init__(self, dim=48, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(14, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, drop,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, 1)
    def forward(self, x, pad):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.proj(x)], 1)
        p = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool), pad], 1)
        return self.head(self.tf(h, src_key_padding_mask=p)[:, 0]).squeeze(-1)

X = torch.tensor(tok); PAD = torch.tensor(pad)
Y = torch.tensor(goal, dtype=torch.float32)
itr, iva, ite = map(np.flatnonzero, (tr, va, te))

def run(model, seed):
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    lf = nn.BCEWithLogitsLoss()
    best, state, patc = 0, None, 0
    for ep in range(60):
        model.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(itr)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lf(model(X[b], PAD[b]), Y[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(goal[va], model(X[iva], PAD[iva]).numpy())
        if auc > best + 1e-4:
            best, state, patc = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patc += 1
            if patc >= 8:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X[ite], PAD[ite])).numpy()

for name, mk in (("DS Deep Sets (sem atencao)", DeepSets),
                 ("TF Transformer (com atencao)", Former)):
    ps = [run(mk(), s) for s in (0, 1)]
    aucs = [roc_auc_score(goal[te], p) for p in ps]
    lls = [log_loss(goal[te], np.clip(p, 1e-6, 1 - 1e-6)) for p in ps]
    results[name] = (np.mean(aucs), np.mean(lls))
    print(f"{name:34s} AUC={np.mean(aucs):.4f} ±{np.std(aucs):.4f}  logloss={np.mean(lls):.4f} ±{np.std(lls):.4f}")

print("\n===== RESUMO (teste, split por partida) =====")
for k, (a, l) in results.items():
    print(f"{k:34s} AUC={a:.4f}  logloss={l:.4f}")

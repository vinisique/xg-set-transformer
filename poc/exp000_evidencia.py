# -*- coding: utf-8 -*-
"""
EXP-000 — reexecucao com EVIDENCIA COMPLETA.

Reproduz exatamente a escada de baselines de poc3_xg3.py (mesmas seeds, mesmo
split, mesma arquitetura) e, alem de imprimir as metricas, GUARDA as previsoes
de teste em disco. Sem as previsoes salvas nao existe curva de calibracao,
histograma de xG nem teste estatistico pareado — so os quatro numeros finais.

Alem de AUC e log loss, calcula Brier score e ECE. Calcular todas as metricas
nao antecipa o cartao 0001: aquele cartao decide qual metrica *decide*, e
qualquer que seja a resposta, todas precisam estar medidas.

Saidas:
  experiments/EXP-000/predicoes.npz    (fora do git — reconstrutivel)
  docs/experimentos/EXP-000-completo.json
  docs/experimentos/figuras/EXP-000-calibracao.png
  docs/experimentos/figuras/EXP-000-seeds.png

Uso:  python exp000_evidencia.py     (CPU: dezenas de minutos)
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import viz

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-000")
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
os.makedirs(DIR_EXP, exist_ok=True)

SEEDS = (0, 1)

# ------------------------------------------------------------- metricas ----

def brier(y, p):
    """Erro quadratico medio da probabilidade. Ao contrario da AUC, enxerga
    calibracao: prever 0,9 num nao-gol custa caro."""
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    """Expected Calibration Error: media ponderada de |observado - previsto|
    em bins de quantil. Bins de quantil (e nao de largura fixa) porque a maioria
    dos chutes tem xG baixo — bins fixos deixariam os ultimos quase vazios."""
    cortes = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    idx = np.digitize(p, cortes[1:-1])
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def curva_calibracao(y, p, n_bins=10):
    """Devolve (previsto_medio, observado, n) por bin de quantil."""
    cortes = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    idx = np.digitize(p, cortes[1:-1])
    prev, obs, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        prev.append(p[m].mean()); obs.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(prev), np.array(obs), np.array(ns)


def metricas(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }


# ----------------------------------------------------------------- dados ----
d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots_all.npz"),
            allow_pickle=True)
sx, sy, goal = d["sx"], d["sy"], d["goal"]
px, py, mate, gk, mask, header = d["px"], d["py"], d["mate"], d["gk"], d["mask"], d["header"]
match_id = d["match_id"]
N, P = px.shape
print(f"{N} chutes | {goal.mean()*100:.1f}% gols | {len(np.unique(match_id))} partidas")

# geometria: gol em x=120, traves em y=36 e 44
dx, dy = 120.0 - sx, sy - 40.0
dist = np.hypot(dx, dy)
angle = np.abs(np.arctan2(8.0 * dx, dx**2 + dy**2 - 16.0))

# split POR PARTIDA — chutes do mesmo jogo nunca caem em splits diferentes
rng = np.random.default_rng(0)
uniq = np.unique(match_id); rng.shuffle(uniq)
n1, n2 = int(0.7 * len(uniq)), int(0.85 * len(uniq))
tr = np.isin(match_id, uniq[:n1])
va = np.isin(match_id, uniq[n1:n2])
te = np.isin(match_id, uniq[n2:])
print(f"treino {tr.sum()} | val {va.sum()} | teste {te.sum()}")

y_te = goal[te]

# --------------------------------------------------- B1 e B2 (logisticas) ----
F1 = np.stack([dist, angle, header], 1)
lr1 = LogisticRegression(max_iter=1000).fit(F1[tr], goal[tr])
p_b1 = lr1.predict_proba(F1[te])[:, 1]

opp = (mate < 0.5) & (mask > 0.5)
gkm = opp & (gk > 0.5)
gk_dist = np.where(gkm, np.hypot(120 - px, py - 40), np.nan)
gk_dist = np.nanmin(np.where(gkm, gk_dist, np.inf), 1)
gk_dist[np.isinf(gk_dist)] = 25.0
gkx = np.where(gkm, px, 0).sum(1) / np.maximum(gkm.sum(1), 1)
gky = np.where(gkm, py, 0).sum(1) / np.maximum(gkm.sum(1), 1)
t = np.clip(((gkx - sx) * dx + (gky - sy) * (40 - sy)) / (dist**2 + 1e-8), 0, 1)
gk_off = np.hypot(sx + t * dx - gkx, sy + t * (40 - sy) - gky)


def inside(pxx, pyy):
    d1 = (36 - sy[:, None]) * (pxx - sx[:, None]) - (120 - sx[:, None]) * (pyy - sy[:, None])
    d2 = (44 - sy[:, None]) * (pxx - sx[:, None]) - (120 - sx[:, None]) * (pyy - sy[:, None])
    return (d1 * d2 < 0) & (pxx > sx[:, None])


blockers = (inside(px, py) & opp & (gk < 0.5)).sum(1).astype(np.float32)
opp_d = np.where(opp, np.hypot(px - sx[:, None], py - sy[:, None]), np.inf)
near_opp = np.min(opp_d, 1); near_opp[np.isinf(near_opp)] = 30.0
F2 = np.stack([dist, angle, header, gk_dist, gk_off, blockers, near_opp], 1)
lr2 = LogisticRegression(max_iter=2000).fit(F2[tr], goal[tr])
p_b2 = lr2.predict_proba(F2[te])[:, 1]

# ------------------------------------- tokens: geometria relativa ao chute ----
pdist = np.hypot(120 - px, py - 40)
rdx = (px - sx[:, None]) / 40.0
rdy = (py - sy[:, None]) / 40.0
dshoot = np.hypot(px - sx[:, None], py - sy[:, None])
tline = np.clip(((px - sx[:, None]) * dx[:, None] + (py - sy[:, None]) * (40 - sy)[:, None])
                / (dist[:, None] ** 2 + 1e-8), 0, 1)
perp = np.hypot(sx[:, None] + tline * dx[:, None] - px,
                sy[:, None] + tline * (40 - sy)[:, None] - py)
intri = inside(px, py).astype(np.float32)

Z = np.zeros(N); ZP = np.zeros((N, P))
shooter = np.stack([sx / 120, sy / 80, dist / 50, angle, Z, Z, Z, Z, Z, Z,
                    np.ones(N), np.zeros(N), np.ones(N), header], 1)[:, None, :]
others = np.stack([px / 120, py / 80, pdist / 50, ZP,
                   rdx, rdy, dshoot / 40.0, tline, perp / 20.0, intri,
                   mate, gk, ZP, ZP], -1)
tok = np.concatenate([shooter, others], 1).astype(np.float32)     # [N, 22, 14]
pad = np.concatenate([np.zeros((N, 1)), 1 - mask], 1) > 0.5       # True = ignorar


class DeepSets(nn.Module):
    """Processa cada jogador ISOLADAMENTE e agrega. Nunca compara dois jogadores
    entre si — e por isso que ele e o baseline que isola o valor da atencao."""
    def __init__(self, dim=48, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(14, dim)
        self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(dim * 2, dim), nn.ReLU())
        self.head = nn.Linear(dim * 2, 1)

    def forward(self, x, pad):
        h = self.phi(self.proj(x))
        w = (~pad).float().unsqueeze(-1)
        mean = (h * w).sum(1) / w.sum(1)
        mx = h.masked_fill(pad.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


class Former(nn.Module):
    """Set Transformer: [CLS] + encoder SEM positional encoding (a cena e um
    conjunto, nao uma sequencia) + mascara de padding para cenas incompletas."""
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


def treina(model, seed):
    """Treino identico ao de poc3_xg3.py: selecao por AUC de validacao."""
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


# ------------------------------------------------------------- execucao ----
previsoes = {"B1": p_b1, "B2": p_b2}
por_seed = {}

for chave, classe in (("DS", DeepSets), ("TF", Former)):
    ps = []
    for s in SEEDS:
        t0 = time.time()
        p = treina(classe(), s)
        ps.append(p)
        print(f"  {chave} seed {s}: AUC={roc_auc_score(y_te, p):.4f} "
              f"Brier={brier(y_te, p):.5f}  ({time.time()-t0:.0f}s)")
    por_seed[chave] = ps
    previsoes[chave] = np.mean(ps, axis=0)   # ensemble das seeds

np.savez_compressed(os.path.join(DIR_EXP, "predicoes.npz"),
                    y=y_te, **{k: v for k, v in previsoes.items()},
                    **{f"{k}_seed{s}": p for k, ps in por_seed.items()
                       for s, p in zip(SEEDS, ps)})
print(f"previsoes salvas em {DIR_EXP}")

resumo = {"id": "EXP-000", "n_teste": int(len(y_te)),
          "taxa_gol_teste": float(y_te.mean()), "seeds": list(SEEDS), "modelos": {}}
for k in ("B1", "B2", "DS", "TF"):
    resumo["modelos"][k] = {"media_seeds": metricas(y_te, previsoes[k])}
    if k in por_seed:
        resumo["modelos"][k]["por_seed"] = [metricas(y_te, p) for p in por_seed[k]]

with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-000-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)

print("\n===== EXP-000 (teste, split por partida) =====")
print(f"{'modelo':8s} {'AUC':>8s} {'logloss':>9s} {'Brier':>9s} {'ECE':>8s}")
for k in ("B1", "B2", "DS", "TF"):
    m = resumo["modelos"][k]["media_seeds"]
    print(f"{k:8s} {m['auc']:8.4f} {m['logloss']:9.4f} {m['brier']:9.5f} {m['ece']:8.4f}")

# -------------------------------------------------------------- figuras ----
ORDEM = ["B1", "B2", "DS", "TF"]
ROTULO = {"B1": "B1 · logística", "B2": "B2 · + interação manual",
          "DS": "DS · Deep Sets", "TF": "TF · Transformer"}

# Figura 1 — curva de calibracao (o pedido nº 2 do professor)
fig, axs = viz.figura(largura=12.0, altura=5.0, colunas=2)
ax = axs[0]
ax.plot([0, 0.62], [0, 0.62], color=viz.COR["eixo"], lw=1.2, zorder=1)
ax.annotate("calibração perfeita", (0.42, 0.42), rotation=38, fontsize=8,
            color=viz.COR["suave"], ha="center", va="bottom")
for k in ORDEM:
    prev, obs, _ = curva_calibracao(y_te, previsoes[k])
    ax.plot(prev, obs, "-o", color=viz.COR_MODELO[k], lw=2, ms=6,
            mec=viz.COR["superficie"], mew=1.5, label=ROTULO[k], zorder=3)
ax.set_title("Curva de calibração", color=viz.COR["tinta"], pad=28)
ax.annotate("entre os chutes a que o modelo deu 20%, quantos viraram gol?",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("xG previsto (média do decil)")
ax.set_ylabel("frequência observada de gol")
ax.grid(True); ax.set_axisbelow(True)
ax.legend(loc="upper left")

# Figura 1b — Brier por modelo
ax = axs[1]
ys = list(range(len(ORDEM)))[::-1]
bs = [resumo["modelos"][k]["media_seeds"]["brier"] for k in ORDEM]
for k, y_, b in zip(ORDEM, ys, bs):
    ax.plot([min(bs) * 0.995, b], [y_, y_], color=viz.COR_MODELO[k], lw=2.5,
            solid_capstyle="round", zorder=2)
    ax.plot([b], [y_], "o", ms=9, color=viz.COR_MODELO[k],
            mec=viz.COR["superficie"], mew=2, zorder=3)
    ax.annotate(f"{b:.5f}".replace(".", ","), (b, y_), textcoords="offset points",
                xytext=(12, 0), va="center", fontsize=9, fontweight="bold",
                color=viz.COR["tinta"])
ax.set_yticks(ys); ax.set_yticklabels([ROTULO[k] for k in ORDEM], fontsize=8.5)
ax.set_title("Brier score", color=viz.COR["tinta"], pad=28)
ax.annotate("erro quadrático da probabilidade — menor é melhor",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
ax.set_xlim(min(bs) * 0.995, max(bs) * 1.006)

fig.suptitle("EXP-000 · calibração — as métricas que a AUC não enxerga",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-000 · teste = {len(y_te):,} chutes · decis de xG previsto · "
                "modelos neurais: média das seeds 0 e 1. "
                "Fonte: docs/experimentos/EXP-000-completo.json".replace(",", "."))
fig.tight_layout(rect=[0, 0, 1, 0.93])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-000-calibracao.png"))

# Figura 2 — o ganho da atencao vs. o ruido entre seeds
fig, ax = viz.figura(largura=8.4, altura=3.6)
for i, k in enumerate(["DS", "TF"]):
    aucs = [m["auc"] for m in resumo["modelos"][k]["por_seed"]]
    y_ = 1 - i
    ax.plot(aucs, [y_] * len(aucs), "o", ms=11, color=viz.COR_MODELO[k],
            mec=viz.COR["superficie"], mew=2, zorder=3)
    for s, a in zip(SEEDS, aucs):
        ax.annotate(f"seed {s}", (a, y_), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=7.5,
                    color=viz.COR["suave"])
    ax.annotate(f"{np.mean(aucs):.4f}".replace(".", ","),
                (np.mean(aucs), y_), textcoords="offset points", xytext=(0, -22),
                ha="center", fontsize=9, fontweight="bold", color=viz.COR["tinta"])
ax.set_yticks([1, 0]); ax.set_yticklabels(["DS · Deep Sets", "TF · Transformer"])
ax.set_ylim(-0.6, 1.6)
ax.set_title("O ganho da atenção é da ordem do ruído entre seeds",
             color=viz.COR["tinta"], pad=28)
ax.annotate("cada ponto é uma semente de inicialização; se as nuvens se tocam, "
            "a diferença não está estabelecida",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("AUC no teste")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
viz.rodape(fig, "EXP-000 · apenas 2 seeds — insuficiente para conclusão. "
                "O teste pareado com 5+ seeds é o EXP-004.")
fig.tight_layout()
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-000-seeds.png"))

print("\nEvidencia completa gerada.")

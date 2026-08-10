# -*- coding: utf-8 -*-
"""
EXP-009 — ablacao causal: o ganho vem MESMO da atencao nos bloqueadores?

O EXP-005 mostrou onde a atencao se concentra. Isso e CORRELACIONAL: nao prova
que o desempenho dependa dessa concentracao. Aqui a atencao do [CLS] sobre
grupos especificos e SUPRIMIDA (zerada e renormalizada) e mede-se quanto o
Brier piora.

O controle e a parte que da sentido ao teste: suprimir N jogadores sempre muda
alguma coisa. A pergunta e se suprimir os BLOQUEADORES machuca mais do que
suprimir a MESMA QUANTIDADE de jogadores sorteados. Sem esse controle, qualquer
degradacao seria confundida com efeito do grupo escolhido.

Limitacao declarada: usa a semente 0, unica cujos pesos foram salvos (o EXP-004
guardou previsoes, nao modelos).

Uso:  python exp009_ablacao.py
"""
import json
import os

import numpy as np
import torch

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
PESOS = os.path.join(RAIZ, "experiments", "EXP-005", "former_seed0.pt")
LOTE = 1024

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
bruto = np.load(os.path.join(AQUI, "shots_all.npz"), allow_pickle=True)
te = D["te"]
y = D["goal"][te]

modelo = xb.Former()
modelo.load_state_dict(torch.load(PESOS, weights_only=True))
modelo.eval()

X = torch.tensor(D["tok"][te]); PAD = torch.tensor(D["pad"][te])
n = len(X)

# ------------------------------------------------------------- papeis -------
mask = bruto["mask"][te] > 0.5
mate = bruto["mate"][te] > 0.5
gk = bruto["gk"][te] > 0.5
px, py = bruto["px"][te], bruto["py"][te]
sx, sy = bruto["sx"][te], bruto["sy"][te]
d1 = (36 - sy[:, None]) * (px - sx[:, None]) - (120 - sx[:, None]) * (py - sy[:, None])
d2 = (44 - sy[:, None]) * (px - sx[:, None]) - (120 - sx[:, None]) * (py - sy[:, None])
no_tri = (d1 * d2 < 0) & (px > sx[:, None])

bloq = mask & ~mate & ~gk & no_tri
goleiro = mask & ~mate & gk
comp = mask & mate

# controle: mesma QUANTIDADE de bloqueadores, mas jogadores sorteados
rng = np.random.default_rng(0)
sorteado = np.zeros_like(bloq)
for i in range(n):
    vis = np.flatnonzero(mask[i])
    k = int(bloq[i].sum())
    if k and len(vis):
        sorteado[i, rng.choice(vis, size=min(k, len(vis)), replace=False)] = True


def para23(m):
    """[n,21] dos jogadores -> [n,23] com CLS e chutador em False."""
    out = np.zeros((n, 23), dtype=bool)
    out[:, 2:] = m
    return torch.tensor(out)


CONDICOES = {
    "nenhuma (referência)": None,
    "bloqueadores": para23(bloq),
    "goleiro": para23(goleiro),
    "companheiros": para23(comp),
    "sorteados (mesmo nº de bloqueadores)": para23(sorteado),
    "todos os jogadores": para23(mask),
}


def prever(sup):
    saida = []
    for i in range(0, n, LOTE):
        s = None if sup is None else sup[i:i + LOTE]
        lo = xb.forward_com_intervencao(modelo, X[i:i + LOTE], PAD[i:i + LOTE],
                                        suprimir=s)
        saida.append(torch.sigmoid(lo))
    return torch.cat(saida).numpy()


print(f"teste: {n} chutes · média de {bloq.sum(1).mean():.2f} bloqueadores por cena")
print(f"cenas com ao menos um bloqueador: {(bloq.sum(1) > 0).sum()} "
      f"({(bloq.sum(1) > 0).mean()*100:.0f}%)\n")

res = {}
base = None
print(f"{'condição suprimida':40s} {'Brier':>9s} {'Δ vs ref':>10s} {'AUC':>8s}")
for nome, sup in CONDICOES.items():
    p = prever(sup)
    m = xb.metricas(y, p)
    if base is None:
        base = m["brier"]
    res[nome] = {"brier": m["brier"], "auc": m["auc"], "ece": m["ece"],
                 "delta_brier": m["brier"] - base}
    print(f"{nome:40s} {m['brier']:9.5f} {m['brier']-base:+10.5f} {m['auc']:8.4f}")

with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-009-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump({"id": "EXP-009", "semente": 0, "n_teste": int(n),
               "media_bloqueadores": float(bloq.sum(1).mean()),
               "condicoes": res}, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figura ----
ORDEM = ["bloqueadores", "sorteados (mesmo nº de bloqueadores)", "goleiro",
         "companheiros", "todos os jogadores"]
CORES = [viz.SERIE[3], viz.COR["suave"], viz.SERIE[1], viz.SERIE[2], viz.SERIE[0]]

fig, ax = viz.figura(largura=9.6, altura=4.4)
ys = list(range(len(ORDEM)))[::-1]
vals = [res[k]["delta_brier"] for k in ORDEM]
ax.axvline(0, color=viz.COR["tinta"], lw=1.6, zorder=4)
for k, y_, v, c in zip(ORDEM, ys, vals, CORES):
    ax.plot([0, v], [y_, y_], color=c, lw=3, solid_capstyle="round", zorder=2)
    ax.plot([v], [y_], "o", ms=10, color=c, mec=viz.COR["superficie"], mew=2, zorder=3)
    ax.annotate(f"{v:+.5f}".replace(".", ","), (v, y_), textcoords="offset points",
                xytext=(12, 0), va="center", fontsize=9, fontweight="bold",
                color=viz.COR["tinta"])
ax.set_yticks(ys); ax.set_yticklabels(ORDEM, fontsize=9)
ax.set_ylim(-0.6, len(ORDEM) - 0.4)
ax.set_xlim(-max(vals) * 0.08, max(vals) * 1.3)
ax.set_title("Quanto o Brier piora ao proibir o [CLS] de olhar para cada grupo",
             color=viz.COR["tinta"], pad=28)
ax.annotate("à direita = pior. A linha cinza é o controle: mesmo número de "
            "jogadores, sorteados",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("aumento do Brier em relação ao modelo intacto")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
rodape = (f"EXP-009 · Transformer semente 0 · {n} chutes de teste · atenção do "
          f"[CLS] zerada e renormalizada · Brier de referência "
          + f"{base:.5f}".replace(".", ","))
viz.rodape(fig, rodape)
fig.tight_layout()
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-009-ablacao.png"))
print("\nEXP-009 concluído.")

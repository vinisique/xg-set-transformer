# -*- coding: utf-8 -*-
"""
Refaz a figura do EXP-012 comparando PERICIA, nao Brier bruto.

Por que: o Brier depende da taxa de gol do conjunto de teste. A Premier League
2015/16 tem 9,31% de gols contra 10,52% do teste habitual, entao QUALQUER modelo
tem Brier mais baixo la. Sobrepor os dois valores sugeriria que o modelo melhora
numa competicao que nunca viu — o contrario do que acontece.

A medida comparavel e a pericia sobre a referencia trivial (prever sempre a taxa
media de gol):   pericia = 1 - Brier / [p(1-p)].

Uso:  python fig_exp012.py
"""
import json
import os

import numpy as np

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DOCS = os.path.join(RAIZ, "docs", "experimentos")

e12 = json.load(open(os.path.join(DOCS, "EXP-012-completo.json"), encoding="utf-8"))
e10 = json.load(open(os.path.join(DOCS, "EXP-010-completo.json"), encoding="utf-8"))

# taxas de gol de cada conjunto de teste
p_fora = e12["gols"] / e12["n_teste"]
y_hab = np.load(os.path.join(RAIZ, "experiments", "EXP-004", "predicoes.npz"))["y"]
p_dentro = float(y_hab.mean())
ref_fora, ref_dentro = p_fora * (1 - p_fora), p_dentro * (1 - p_dentro)

DENTRO = {"B1": "B1 · logística", "B2": "B2 · logística + interação",
          "DS": "DS · Deep Sets (tokens)", "TF": "TF · Transformer (tokens)"}
ORDEM = ["B1", "B2", "DS", "TF"]

pericia = {}
for k in ORDEM:
    pericia[k] = {
        "dentro": (1 - e10["modelos"][DENTRO[k]]["brier"] / ref_dentro) * 100,
        "fora": (1 - e12["modelos"][k]["brier"] / ref_fora) * 100,
        "auc_dentro": e10["modelos"][DENTRO[k]]["auc"],
        "auc_fora": e12["modelos"][k]["auc"],
    }

print(f"taxa de gol: dentro {p_dentro*100:.2f}%  ·  fora {p_fora*100:.2f}%")
print(f"{'modelo':6s} {'perícia dentro':>15s} {'perícia fora':>14s} "
      f"{'AUC dentro':>11s} {'AUC fora':>10s}")
for k in ORDEM:
    v = pericia[k]
    print(f"{k:6s} {v['dentro']:14.2f}% {v['fora']:13.2f}% "
          f"{v['auc_dentro']:11.4f} {v['auc_fora']:10.4f}")

fig, axs = viz.figura(largura=12.0, altura=4.0, colunas=2)
for ax, chaves, titulo, sufixo in (
        (axs[0], ("dentro", "fora"), "Perícia sobre a taxa média de gol", "%"),
        (axs[1], ("auc_dentro", "auc_fora"), "AUC", "")):
    ys = list(range(len(ORDEM)))[::-1]
    for k, y_ in zip(ORDEM, ys):
        d, f = pericia[k][chaves[0]], pericia[k][chaves[1]]
        ax.plot([d, f], [y_, y_], color=viz.COR["grade"], lw=1.6, zorder=2)
        ax.plot([d], [y_], "o", ms=8, color=viz.COR["suave"],
                mec=viz.COR["superficie"], mew=1.5, zorder=3)
        ax.plot([f], [y_], "o", ms=10, color=viz.COR_MODELO[k],
                mec=viz.COR["superficie"], mew=2, zorder=4)
        casas = 1 if sufixo else 4
        ax.annotate(f"{f:.{casas}f}{sufixo}".replace(".", ","), (f, y_),
                    textcoords="offset points", xytext=(0, -19), ha="center",
                    fontsize=8.5, fontweight="bold", color=viz.COR["tinta"])
    ax.set_yticks(ys); ax.set_yticklabels(ORDEM if sufixo else [""] * 4)
    ax.set_ylim(-0.75, len(ORDEM) - 0.4)
    ax.set_title(titulo + "  ·  maior é melhor", color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)

axs[0].annotate("cinza = partidas sorteadas · colorido = competição nunca vista",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
axs[1].annotate("a AUC é comparável entre conjuntos; o Brier bruto não seria",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle("EXP-012 · quanto o modelo perde numa competição que nunca viu",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-012 · fora do treino: {e12['competicao_fora']} · "
                f"{e12['n_teste']} chutes, {e12['gols']} gols · "
                f"taxa de gol {p_fora*100:.2f}% contra {p_dentro*100:.2f}% do teste "
                "habitual — por isso a comparação usa perícia, não Brier bruto.")
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DOCS, "figuras", "EXP-012-holdout.png"))

# -*- coding: utf-8 -*-
"""
Regenera as figuras do EXP-000 a partir das PREVISOES SALVAS.

Nao retreina nada: le experiments/EXP-000/predicoes.npz. E exatamente por isso
que guardar as previsoes valeu a pena — corrigir uma figura custa segundos em
vez de 40 minutos de CPU.

Uso:  python fig_exp000.py
"""
import os

import numpy as np

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")

d = np.load(os.path.join(RAIZ, "experiments", "EXP-000", "predicoes.npz"))
y = d["y"]
SEEDS = (0, 1)
ORDEM = ["B1", "B2", "DS", "TF"]
ROTULO = {"B1": "B1 · logística", "B2": "B2 · + interação manual",
          "DS": "DS · Deep Sets", "TF": "TF · Transformer"}

M = {k: xb.metricas(y, d[k]) for k in ORDEM}
POR_SEED = {k: [xb.metricas(y, d[f"{k}_seed{s}"]) for s in SEEDS] for k in ("DS", "TF")}

# ============================================ figura 1 — calibracao =========
fig, axs = viz.figura(largura=12.0, altura=5.0, colunas=2)

ax = axs[0]
ax.plot([0, 0.62], [0, 0.62], color=viz.COR["eixo"], lw=1.2, zorder=1)
ax.annotate("calibração perfeita", (0.45, 0.45), rotation=38, fontsize=8,
            color=viz.COR["suave"], ha="center", va="bottom")
for k in ORDEM:
    prev, obs, _ = xb.curva_calibracao(y, d[k])
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

# Brier e ECE lado a lado. Grafico de PONTOS puro, sem hastes: haste a partir
# de uma origem arbitraria seria um grafico de barras truncado — o comprimento
# sugeriria proporcao que o eixo nao tem.
ax = axs[1]
ys = list(range(len(ORDEM)))[::-1]
for k, y_ in zip(ORDEM, ys):
    b = M[k]["brier"]
    ax.plot([b], [y_], "o", ms=11, color=viz.COR_MODELO[k],
            mec=viz.COR["superficie"], mew=2, zorder=3)
    ax.annotate(f"{b:.5f}".replace(".", ","), (b, y_), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=9, fontweight="bold",
                color=viz.COR["tinta"])
    ax.annotate(f"ECE {M[k]['ece']:.4f}".replace(".", ","), (b, y_),
                textcoords="offset points", xytext=(0, -20), ha="center",
                fontsize=8, color=viz.COR["suave"])
ax.set_yticks(ys); ax.set_yticklabels([ROTULO[k] for k in ORDEM], fontsize=8.5)
ax.set_ylim(-0.7, len(ORDEM) - 0.3)
ax.set_title("Brier score  ·  menor é melhor", color=viz.COR["tinta"], pad=28)
ax.annotate("abaixo de cada ponto, o erro de calibração (ECE)",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
bs = [M[k]["brier"] for k in ORDEM]
folga = (max(bs) - min(bs)) * 0.18
ax.set_xlim(min(bs) - folga, max(bs) + folga)

fig.suptitle("EXP-000 · calibração — o que a AUC não enxerga",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-000 · teste = {len(y):,} chutes · decis de xG previsto · "
                "modelos neurais: média das sementes 0 e 1. "
                "Fonte: experiments/EXP-000/predicoes.npz".replace(",", "."))
fig.tight_layout(rect=[0, 0, 1, 0.93])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-000-calibracao.png"))

# ============================================ figura 2 — por semente ========
fig, axs = viz.figura(largura=12.0, altura=3.8, colunas=2)
for ax, met, titulo, menor in ((axs[0], "auc", "AUC por semente", False),
                               (axs[1], "brier", "Brier por semente", True)):
    for i, k in enumerate(["DS", "TF"]):
        vals = [m[met] for m in POR_SEED[k]]
        y_ = 1 - i
        ax.plot(vals, [y_] * len(vals), "o", ms=11, color=viz.COR_MODELO[k],
                mec=viz.COR["superficie"], mew=2, zorder=3)
        # rotulo unico da faixa — rotular semente a semente colide quando os
        # valores praticamente coincidem, como acontece no Deep Sets
        # compara os valores JA ARREDONDADOS: se imprimem igual, um rotulo so
        lo, hi = f"{min(vals):.5f}", f"{max(vals):.5f}"
        txt = (lo if lo == hi else f"{lo} a {hi}").replace(".", ",")
        ax.annotate(txt, (np.mean(vals), y_), textcoords="offset points",
                    xytext=(0, -22), ha="center", fontsize=8.5,
                    fontweight="bold", color=viz.COR["tinta"])
    ax.set_yticks([1, 0]); ax.set_yticklabels(["DS · Deep Sets", "TF · Transformer"])
    ax.set_ylim(-0.75, 1.6)
    ax.set_title(titulo + ("  ·  menor é melhor" if menor else "  ·  maior é melhor"),
                 color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    todos = [m[met] for k in ("DS", "TF") for m in POR_SEED[k]]
    folga = (max(todos) - min(todos)) * 0.22
    ax.set_xlim(min(todos) - folga, max(todos) + folga)

axs[0].annotate("cada ponto é uma semente de inicialização",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
axs[1].annotate("as duas sementes do Deep Sets coincidem em Brier",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])

fig.suptitle("EXP-000 · nas duas sementes, o Transformer ficou acima do Deep Sets",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, "EXP-000 · as faixas NÃO se sobrepõem — mas com apenas 2 sementes "
                "isso é indício, não conclusão. O teste pareado com 5 sementes é o EXP-004.")
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-000-seeds.png"))

print("\n=== EXP-000 (previsoes salvas) ===")
print(f"{'modelo':10s} {'AUC':>8s} {'logloss':>9s} {'Brier':>9s} {'ECE':>8s}")
for k in ORDEM:
    m = M[k]
    print(f"{k:10s} {m['auc']:8.4f} {m['logloss']:9.4f} {m['brier']:9.5f} {m['ece']:8.4f}")

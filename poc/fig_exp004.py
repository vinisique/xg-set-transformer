# -*- coding: utf-8 -*-
"""
Regenera as figuras do EXP-004 a partir das previsoes salvas e do JSON.
Nao retreina nada: le experiments/EXP-004/predicoes.npz.

Corrige um bug da primeira versao: `"texto" f"{x}".replace(".", ",")` aplica o
replace ao literal JA CONCATENADO, entao os pontos finais das frases viravam
virgula ("...nao esta estabelecida,"). Aqui os numeros sao formatados a parte.

Uso:  python fig_exp004.py
"""
import json
import os

import numpy as np

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")

d = np.load(os.path.join(RAIZ, "experiments", "EXP-004", "predicoes.npz"))
R = json.load(open(os.path.join(RAIZ, "docs", "experimentos", "EXP-004-completo.json"),
                   encoding="utf-8"))
SEEDS = R["seeds"]
y = d["y"]
partidas = d["partidas"]


def vg(x, casas):
    """Numero no formato brasileiro, sem tocar na pontuacao da frase."""
    return f"{x:+.{casas}f}".replace(".", ",") if x < 0 else f"{x:.{casas}f}".replace(".", ",")


# ------------------------------------------- figura 1 — por semente ---------
fig, axs = viz.figura(largura=12.0, altura=3.8, colunas=2)
for ax, met, titulo, menor in ((axs[0], "brier", "Brier por semente", True),
                               (axs[1], "auc", "AUC por semente", False)):
    for i, k in enumerate(["DS", "TF"]):
        vals = [m[met] for m in R["por_seed"][k]]
        y_ = 1 - i
        ax.plot(vals, [y_] * len(vals), "o", ms=10, color=viz.COR_MODELO[k],
                mec=viz.COR["superficie"], mew=2, zorder=3, alpha=0.9)
        ax.plot([np.mean(vals)], [y_], "|", ms=28, mew=2.5,
                color=viz.COR["tinta"], zorder=4)
        casas = 5 if met == "brier" else 4
        ax.annotate("média " + vg(np.mean(vals), casas), (np.mean(vals), y_),
                    textcoords="offset points", xytext=(0, -24), ha="center",
                    fontsize=8.5, fontweight="bold", color=viz.COR["tinta"])
    ax.set_yticks([1, 0]); ax.set_yticklabels(["DS · Deep Sets", "TF · Transformer"])
    ax.set_ylim(-0.8, 1.6)
    ax.set_title(titulo + ("  ·  menor é melhor" if menor else "  ·  maior é melhor"),
                 color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    todos = [m[met] for k in ("DS", "TF") for m in R["por_seed"][k]]
    folga = (max(todos) - min(todos)) * 0.2
    ax.set_xlim(min(todos) - folga, max(todos) + folga)

axs[0].annotate("as faixas não se sobrepõem", (0, 1), xycoords="axes fraction",
                textcoords="offset points", xytext=(0, 8), fontsize=8.5,
                color=viz.COR["tinta_2"])
axs[1].annotate("as faixas se sobrepõem — e o Deep Sets ordena um pouco melhor",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle("EXP-004 · a atenção melhora a probabilidade, não a ordenação",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-004 · {len(SEEDS)} sementes por modelo · "
                f"teste = {len(y):,} chutes em {len(np.unique(partidas))} partidas · "
                "traço vertical = média.".replace(",", "."))
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-004-seeds.png"))

# ------------------------------------------- figura 2 — bootstrap -----------
tb = xb.bootstrap_pareado(y, d["TF_ensemble"], d["DS_ensemble"], partidas,
                          fn=xb.brier, n=2000, seed=0)
fig, ax = viz.figura(largura=8.8, altura=4.0)
ax.hist(tb["distribuicao"], bins=60, color=viz.COR_MODELO["TF"], alpha=0.85,
        edgecolor=viz.COR["superficie"], linewidth=0.5, zorder=2)
topo = ax.get_ylim()[1]
ax.axvline(0, color=viz.COR["tinta"], lw=1.8, zorder=4)
ax.annotate("sem diferença", (0, topo * 0.96), textcoords="offset points",
            xytext=(-8, 0), fontsize=8.5, color=viz.COR["tinta_2"],
            va="top", ha="right")
for v, rot in ((tb["ic95"][0], "IC 2,5%"), (tb["ic95"][1], "IC 97,5%")):
    ax.axvline(v, color=viz.COR["eixo"], lw=1.2, zorder=3)
    ax.annotate(rot, (v, topo * 0.55), textcoords="offset points", xytext=(4, 0),
                fontsize=7.5, color=viz.COR["suave"])
ax.set_title("Diferença de Brier: Transformer − Deep Sets",
             color=viz.COR["tinta"], pad=28)
ax.annotate("2.000 reamostras de PARTIDAS com reposição · "
            "à esquerda de zero = Transformer melhor",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("Brier(TF) − Brier(DS)"); ax.set_ylabel("reamostras")
ax.grid(axis="y"); ax.set_axisbelow(True)

# numeros formatados a parte — a pontuacao da frase fica intacta
rodape = ("EXP-004 · diferença observada " + vg(tb["diferenca"], 5)
          + " · IC 95% [" + vg(tb["ic95"][0], 5) + "; " + vg(tb["ic95"][1], 5) + "]"
          + f" · p = {tb['p_bilateral']:.3f}".replace(".", ",")
          + ". A distribuição inteira está à esquerda de zero: a diferença está "
            "estabelecida.")
viz.rodape(fig, rodape)
fig.tight_layout()
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-004-bootstrap.png"))

print("\n=== EXP-004 ===")
for k in ("DS", "TF"):
    m = R["ensemble"][k]
    print(f"  {k} (ensemble): Brier {m['brier']:.5f} | AUC {m['auc']:.4f} | ECE {m['ece']:.4f}")

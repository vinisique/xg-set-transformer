# -*- coding: utf-8 -*-
"""
Refaz a figura do EXP-009 a partir do JSON, sem recalcular nada.

A primeira versao punha "todos os jogadores" (+0,02582) na mesma escala dos
grupos especificos (+0,00005 a +0,00088). O resultado: a comparacao que da
sentido ao experimento — bloqueadores contra o controle sorteado — virava um
amontoado ilegivel junto de zero. Aqui a supressao total sai do grafico e vira
referencia no rodape.

Uso:  python fig_exp009.py
"""
import json
import os

import viz

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(RAIZ, "docs", "experimentos",
                                "EXP-009-completo.json"), encoding="utf-8"))
C = R["condicoes"]

ORDEM = ["bloqueadores", "sorteados (mesmo nº de bloqueadores)",
         "goleiro", "companheiros"]
ROT = {"bloqueadores": "Bloqueadores\n(adversário na linha do chute)",
       "sorteados (mesmo nº de bloqueadores)":
           "CONTROLE: sorteados\n(mesma quantidade)",
       "goleiro": "Goleiro",
       "companheiros": "Companheiros"}
CORES = [viz.SERIE[3], viz.COR["suave"], viz.SERIE[1], viz.SERIE[2]]

fig, ax = viz.figura(largura=9.8, altura=4.4)
ys = list(range(len(ORDEM)))[::-1]
vals = [C[k]["delta_brier"] for k in ORDEM]

ax.axvline(0, color=viz.COR["tinta"], lw=1.6, zorder=4)
for k, y_, v, c in zip(ORDEM, ys, vals, CORES):
    ax.plot([0, v], [y_, y_], color=c, lw=3.5, solid_capstyle="round", zorder=2)
    ax.plot([v], [y_], "o", ms=11, color=c, mec=viz.COR["superficie"], mew=2, zorder=3)
    ax.annotate(f"+{v:.5f}".replace(".", ","), (v, y_), textcoords="offset points",
                xytext=(13, 0), va="center", fontsize=9.5, fontweight="bold",
                color=viz.COR["tinta"])

# a comparacao que importa, marcada explicitamente
razao = C["bloqueadores"]["delta_brier"] / C["sorteados (mesmo nº de bloqueadores)"]["delta_brier"]
ax.annotate(f"{razao:.0f}× o controle", (vals[0], 3), textcoords="offset points",
            xytext=(13, 16), fontsize=9, fontweight="bold", color=viz.SERIE[3])

ax.set_yticks(ys); ax.set_yticklabels([ROT[k] for k in ORDEM], fontsize=9)
ax.set_ylim(-0.6, len(ORDEM) - 0.25)
ax.set_xlim(-max(vals) * 0.06, max(vals) * 1.38)
ax.set_title("Quanto o Brier piora ao proibir o [CLS] de olhar para cada grupo",
             color=viz.COR["tinta"], pad=30)
ax.annotate("mais à direita = mais dano. O controle isola o efeito de simplesmente "
            "remover jogadores",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("aumento do Brier em relação ao modelo intacto")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)

ref = C["nenhuma (referência)"]["brier"]
todos = C["todos os jogadores"]["delta_brier"]
viz.rodape(fig, "EXP-009 · Transformer semente 0 · "
                f"{R['n_teste']} chutes · Brier de referência "
                + f"{ref:.5f}".replace(".", ",")
                + " · fora da escala: suprimir TODOS os jogadores custa "
                + f"+{todos:.5f}".replace(".", ",") + ".")
fig.tight_layout()
viz.salvar(fig, os.path.join(RAIZ, "docs", "experimentos", "figuras",
                             "EXP-009-ablacao.png"))
print("razao bloqueadores/controle: %.1fx" % razao)

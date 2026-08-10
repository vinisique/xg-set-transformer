# -*- coding: utf-8 -*-
"""
Diagrama da arquitetura: cena -> tokens -> encoder -> xG.

Desenhado a mao em matplotlib, no mesmo estilo das demais figuras, para o
relatorio nao ter uma unica imagem fora da linguagem visual do resto.

Uso:  python fig_arquitetura.py
"""
import os

import matplotlib.patches as mp
import numpy as np

import viz

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(RAIZ, "docs", "experimentos", "figuras", "arquitetura.png")

fig, ax = viz.figura(largura=12.2, altura=3.9)
ax.set_xlim(0, 122); ax.set_ylim(0, 40); ax.axis("off")


def seta(x0, x1, y=20):
    ax.annotate("", (x1, y), (x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                color=viz.COR["tinta_2"], shrinkA=0, shrinkB=0))


def rotulo(x, y, txt, tam=8.5, peso="normal", cor=None):
    ax.text(x, y, txt, ha="center", va="center", fontsize=tam,
            fontweight=peso, color=cor or viz.COR["tinta"])


# ---------------------------------------------------------- 1. a cena -------
ax.add_patch(mp.Rectangle((2, 8), 22, 24, fill=False, ec=viz.COR["eixo"], lw=1.2))
ax.add_patch(mp.Rectangle((17, 13), 7, 14, fill=False, ec=viz.COR["eixo"], lw=1))
ax.plot([24, 24], [17.5, 22.5], color=viz.COR["tinta"], lw=3, solid_capstyle="butt")
rng = np.random.default_rng(3)
for cor, n in ((viz.SERIE[0], 5), (viz.SERIE[2], 4)):
    ax.plot(rng.uniform(5, 21, n), rng.uniform(10, 30, n), "o", ms=5,
            color=cor, alpha=0.75, mec=viz.COR["superficie"], mew=1)
ax.plot([21.5], [20], "s", ms=7, color=viz.SERIE[1], mec=viz.COR["superficie"], mew=1)
ax.plot([11], [20], "*", ms=15, color=viz.COR["tinta"],
        mec=viz.COR["superficie"], mew=1.2)
rotulo(13, 34.5, "cena da finalização", 9.5, "bold")
rotulo(13, 5, "freeze-frame:\naté 22 jogadores", 8, cor=viz.COR["tinta_2"])

seta(26, 33)

# ------------------------------------------------------- 2. os tokens -------
for i, (rot, cor) in enumerate([("[CLS]", viz.COR["tinta"]),
                                ("chutador", viz.COR["tinta"]),
                                ("jogador 1", viz.SERIE[0]),
                                ("jogador 2", viz.SERIE[2]),
                                (". . .", None),
                                ("jogador 21", viz.SERIE[0])]):
    y = 31 - i * 4.6
    if rot == ". . .":
        # reticencias verticais desenhadas como tres pontos: a fonte do sistema
        # nao tem o glifo U+22EE e ele sairia como quadradinho
        for dy in (-1.2, 0.0, 1.2):
            ax.plot([43], [y + dy], "o", ms=2, color=viz.COR["suave"])
        continue
    ax.add_patch(mp.FancyBboxPatch((35, y - 1.7), 16, 3.4, boxstyle="round,pad=0.25",
                                   fc=viz.COR["superficie"], ec=cor, lw=1.4))
    rotulo(43, y, rot, 7.5, "bold" if rot == "[CLS]" else "normal")
rotulo(43, 36.5, "tokens", 9.5, "bold")
rotulo(43, 2.5, "14 atributos por jogador\n(geometria + papel)", 8,
       cor=viz.COR["tinta_2"])

seta(53, 60)

# ------------------------------------------------------- 3. o encoder -------
ax.add_patch(mp.FancyBboxPatch((61, 9), 26, 22, boxstyle="round,pad=0.4",
                               fc=viz.COR["superficie"], ec=viz.SERIE[3], lw=2))
rotulo(74, 27.5, "Transformer encoder", 9.5, "bold")
rotulo(74, 23, "2 camadas · 4 cabeças · dim 48", 8, cor=viz.COR["tinta_2"])
ax.plot([64, 84], [20.5, 20.5], color=viz.COR["grade"], lw=1)
rotulo(74, 17.5, "self-attention", 8.5, "bold", viz.SERIE[3])
rotulo(74, 13.5, "SEM positional encoding\nmáscara nas posições vazias", 7.5,
       cor=viz.COR["tinta_2"])
rotulo(74, 5.5, "a cena é um conjunto:\nembaralhar os jogadores não muda a saída", 8,
       cor=viz.COR["tinta_2"])

seta(89, 96)

# ---------------------------------------------------------- 4. a saida ------
ax.add_patch(mp.FancyBboxPatch((97, 16), 22, 9, boxstyle="round,pad=0.3",
                               fc=viz.COR["superficie"], ec=viz.COR["tinta"], lw=1.6))
rotulo(108, 22, "sigmoide sobre o [CLS]", 8)
rotulo(108, 18.5, "xG entre 0 e 1", 10.5, "bold")   # a fonte não tem o glifo de pertence
rotulo(108, 30, "probabilidade de gol", 9.5, "bold")

fig.suptitle("Arquitetura: o jogador como token",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
fig.tight_layout(rect=[0, 0, 1, 0.94])
viz.salvar(fig, SAIDA)

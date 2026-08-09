# -*- coding: utf-8 -*-
"""
Figura da escada de baselines do EXP-000.

Le docs/experimentos/EXP-000-metricas.json (fonte da verdade, versionada) e
desenha o ganho de cada degrau. A forma escolhida e um grafico de pontos com
segmento de ganho — e nao barras a partir do zero — porque as AUCs vivem entre
0,76 e 0,82: barras desde a origem tornariam as diferencas invisiveis, e barras
truncadas mentiriam sobre a proporcao. O segmento desenha literalmente o
incremento de cada degrau, que e a historia do experimento.

Uso:  python fig_escada.py
"""
import json
import os

import viz

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(RAIZ, "docs", "experimentos", "EXP-000-metricas.json")
SAIDA = os.path.join(RAIZ, "docs", "experimentos", "figuras", "EXP-000-escada.png")

with open(JSON, encoding="utf-8") as f:
    m = json.load(f)

ordem = m["ordem_escada"]
modelos = m["modelos"]


def painel(ax, metrica, titulo, subtitulo, casas=4, menor_melhor=False):
    vals = [modelos[k][metrica] for k in ordem]
    ys = list(range(len(ordem)))[::-1]   # B1 no topo, TF embaixo

    for i, (k, y, v) in enumerate(zip(ordem, ys, vals)):
        cor = viz.COR_MODELO[k]
        if i > 0:
            # segmento do valor anterior ate o atual: o ganho do degrau
            ax.plot([vals[i - 1], v], [y, y], color=cor, lw=2.5,
                    solid_capstyle="round", zorder=2)
            d = v - vals[i - 1]
            meio = (vals[i - 1] + v) / 2
            ax.annotate(f"{d:+.4f}".replace(".", ","), (meio, y),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=viz.COR["tinta_2"],
                        fontweight="bold")
        # marcador com anel da superficie, para nao encostar no segmento
        ax.plot([v], [y], "o", ms=9, color=cor, mec=viz.COR["superficie"],
                mew=2, zorder=3)
        # rotulo direto do valor (alivio para o WARN de contraste da paleta)
        ax.annotate(f"{v:.4f}".replace(".", ","), (v, y),
                    textcoords="offset points", xytext=(12, 0),
                    va="center", fontsize=9, color=viz.COR["tinta"],
                    fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([viz.NOME_MODELO[k] for k in ordem], fontsize=8.5)
    # titulo bem acima; subtitulo logo abaixo dele — sem sobreposicao
    ax.set_title(titulo, color=viz.COR["tinta"], pad=28)
    ax.annotate(subtitulo, (0, 1), xycoords="axes fraction",
                textcoords="offset points", xytext=(0, 8),
                fontsize=8.5, color=viz.COR["tinta_2"])
    ax.set_xlabel(("menor é melhor" if menor_melhor else "maior é melhor"),
                  fontsize=8, color=viz.COR["suave"])
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)

    lo, hi = min(vals), max(vals)
    folga = (hi - lo) * 0.35
    ax.set_xlim(lo - folga, hi + folga * 1.6)
    ax.set_ylim(-0.7, len(ordem) - 0.3)


fig, axs = viz.figura(largura=12.4, altura=4.4, colunas=2)

painel(axs[0], "auc", "AUC — capacidade de ordenar os chutes",
       "cada segmento é o ganho daquele degrau sobre o anterior")
painel(axs[1], "logloss", "Log loss — qualidade da probabilidade",
       "os mesmos degraus, na métrica que enxerga o valor previsto",
       menor_melhor=True)
axs[1].set_yticklabels([])

fig.suptitle(
    "EXP-000 · a representação por token vale muito; a atenção, quase nada",
    x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])

viz.rodape(fig, (
    f"EXP-000 · {m['n_chutes']:,} finalizações · {m['n_partidas']:,} partidas · "
    f"split por partida · teste = {m['split']['teste']:,} chutes · "
    "modelos neurais: média de 2 seeds. "
    "Fonte: docs/experimentos/EXP-000-metricas.json"
).replace(",", "."))

fig.tight_layout(rect=[0, 0, 1, 0.93])
viz.salvar(fig, SAIDA)

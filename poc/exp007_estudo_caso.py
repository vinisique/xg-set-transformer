# -*- coding: utf-8 -*-
"""
EXP-007 — estudo de caso qualitativo: Eurocopa 2024.

Fecha o pedido nº 7 do professor. A proposta previa a Copa do Mundo de 2026,
que a StatsBomb nao publicou ate a data de entrega (cartao 0003); o torneio de
selecoes mais recente disponivel e a Eurocopa 2024.

Usa APENAS lances do conjunto de TESTE — 166 finalizacoes em 6 partidas que o
modelo nunca viu. Usar lances de treino tornaria a analise bonita e vazia.

E analise, nao avaliacao: o professor pediu explicitamente que isso ficasse
claro. Os numeros agregados aparecem para dar contexto, nao para provar nada.

Uso:  python exp007_estudo_caso.py
"""
import json
import os

import matplotlib.patches as mpatches
import numpy as np
import torch

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
PESOS = os.path.join(RAIZ, "experiments", "EXP-005", "former_seed0.pt")
TORNEIO = "UEFA Euro 2024"

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
bruto = np.load(os.path.join(AQUI, "shots_all.npz"), allow_pickle=True)

sel = D["te"] & (D["comp"] == TORNEIO)
idx = np.flatnonzero(sel)
print(f"{TORNEIO}: {len(idx)} finalizações no teste, "
      f"{int(D['goal'][idx].sum())} gols em "
      f"{len(np.unique(D['match_id'][idx]))} partidas")

modelo, _ = xb.treina_e_guarda(xb.Former(), D, 0, PESOS, criterio="brier")
X = torch.tensor(D["tok"][idx]); PAD = torch.tensor(D["pad"][idx])
logit, attn = xb.atencao_do_cls(modelo, X, PAD)
xg = torch.sigmoid(logit).numpy()
a = attn.mean(axis=(1, 2)).numpy()          # [n, 23] media de camadas e cabecas
a_jog = a[:, 2:]                            # 21 jogadores do freeze-frame
a_jog = a_jog / np.maximum(a_jog.sum(1, keepdims=True), 1e-9)

gol = D["goal"][idx]
sx, sy = bruto["sx"][idx], bruto["sy"][idx]
px, py = bruto["px"][idx], bruto["py"][idx]
mask = bruto["mask"][idx] > 0.5
mate = bruto["mate"][idx] > 0.5
gk = bruto["gk"][idx] > 0.5

d1 = (36 - sy[:, None]) * (px - sx[:, None]) - (120 - sx[:, None]) * (py - sy[:, None])
d2 = (44 - sy[:, None]) * (px - sx[:, None]) - (120 - sx[:, None]) * (py - sy[:, None])
no_tri = (d1 * d2 < 0) & (px > sx[:, None])

# ------------------------------------------------------- contexto agregado ---
agregado = {"torneio": TORNEIO, "n_chutes": int(len(idx)),
            "gols": int(gol.sum()), "xg_total": float(xg.sum()),
            "xg_medio": float(xg.mean())}
print(f"xG somado {xg.sum():.1f} contra {int(gol.sum())} gols "
      f"({(xg.sum()/gol.sum()-1)*100:+.1f}%)")

# --------------------------------------------------- escolha dos tres casos --
gk_att = np.where(mask & ~mate & gk, a_jog, 0).sum(1)
blo_att = np.where(mask & ~mate & ~gk & no_tri, a_jog, 0).sum(1)

casos = [
    ("Atenção concentrada no goleiro", int(np.argmax(gk_att))),
    ("Atenção concentrada nos bloqueadores", int(np.argmax(blo_att))),
    ("Maior xG previsto do torneio", int(np.argmax(xg))),
]
vistos, finais = set(), []
for titulo, i in casos:                      # evita repetir o mesmo lance
    if i in vistos:
        ordem = np.argsort(-xg if "xG" in titulo else -blo_att)
        i = int(next(j for j in ordem if j not in vistos))
    vistos.add(i); finais.append((titulo, i))

# --------------------------------------------------------------- desenho -----
def campo(ax):
    """Terco de ataque em coordenadas StatsBomb (gol em x=120, traves 36 e 44)."""
    c = viz.COR["eixo"]
    ax.add_patch(mpatches.Rectangle((60, 0), 60, 80, fill=False, ec=c, lw=1.2))
    ax.add_patch(mpatches.Rectangle((102, 18), 18, 44, fill=False, ec=c, lw=1.2))
    ax.add_patch(mpatches.Rectangle((114, 30), 6, 20, fill=False, ec=c, lw=1.2))
    ax.plot([120, 120], [36, 44], color=viz.COR["tinta"], lw=3.5,
            solid_capstyle="butt", zorder=5)
    ax.plot([108], [40], "o", ms=3, color=c)
    ax.set_xlim(59, 124); ax.set_ylim(-2, 82)
    ax.set_aspect("equal"); ax.axis("off")


fig, axs = viz.figura(largura=13.4, altura=4.8, colunas=3)
for ax, (titulo, i) in zip(axs, finais):
    campo(ax)
    # triangulo do chute ate os postes
    ax.fill([sx[i], 120, 120], [sy[i], 36, 44], color=viz.SERIE[0], alpha=0.07, zorder=1)
    ax.plot([sx[i], 120], [sy[i], 40], color=viz.COR["eixo"], lw=0.9, ls="-", zorder=2)

    vis = np.flatnonzero(mask[i])
    peso = a_jog[i, vis]
    escala = peso / max(peso.max(), 1e-9)
    for j, e in zip(vis, escala):
        if gk[i, j] and not mate[i, j]:
            cor, marca = viz.SERIE[1], "s"      # goleiro
        elif not mate[i, j]:
            cor, marca = (viz.SERIE[3] if no_tri[i, j] else viz.SERIE[0]), "o"
        else:
            cor, marca = viz.SERIE[2], "o"      # companheiro
        ax.plot([px[i, j]], [py[i, j]], marca, ms=5 + 16 * e, color=cor,
                alpha=0.25 + 0.7 * e, mec=viz.COR["superficie"], mew=1.2, zorder=4)
    ax.plot([sx[i]], [sy[i]], "*", ms=19, color=viz.COR["tinta"],
            mec=viz.COR["superficie"], mew=1.4, zorder=6)

    desfecho = "GOL" if gol[i] else "sem gol"
    ax.set_title(titulo, color=viz.COR["tinta"], fontsize=10, pad=6)
    ax.annotate(f"xG previsto {xg[i]:.2f}".replace(".", ",") + f"  ·  {desfecho}",
                (0.5, -0.04), xycoords="axes fraction", ha="center",
                fontsize=9, fontweight="bold", color=viz.COR["tinta"])

leg = [mpatches.Patch(color=viz.SERIE[1], label="goleiro"),
       mpatches.Patch(color=viz.SERIE[3], label="adversário na linha do chute"),
       mpatches.Patch(color=viz.SERIE[0], label="outro adversário"),
       mpatches.Patch(color=viz.SERIE[2], label="companheiro")]
axs[1].legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=4, fontsize=8.5)

fig.suptitle(f"EXP-007 · estudo de caso: {TORNEIO}",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
# a fonte do sistema nao tem o glifo da estrela — descrever em palavras
fig.text(0.0, 0.905, "a estrela marca o chutador · o tamanho de cada jogador é a "
                     "atenção que o token [CLS] dirige a ele",
         fontsize=8.5, color=viz.COR["tinta_2"])
viz.rodape(fig, f"EXP-007 · {len(idx)} finalizações do conjunto de TESTE "
                f"({len(np.unique(D['match_id'][idx]))} partidas) · Transformer semente 0. "
                "Análise qualitativa — não é avaliação quantitativa do modelo.")
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-007-eurocopa.png"))

agregado["casos"] = [{"titulo": t, "xg": float(xg[i]), "gol": int(gol[i]),
                      "atencao_goleiro": float(gk_att[i]),
                      "atencao_bloqueadores": float(blo_att[i]),
                      "jogadores_visiveis": int(mask[i].sum())} for t, i in finais]
with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-007-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump(agregado, f, indent=2, ensure_ascii=False)

print("\ncasos escolhidos:")
for c in agregado["casos"]:
    print(f"  {c['titulo']:38s} xG {c['xg']:.3f} | {'gol' if c['gol'] else 'sem gol':7s}"
          f" | atenção GK {c['atencao_goleiro']:.2f} | bloq {c['atencao_bloqueadores']:.2f}")

# -*- coding: utf-8 -*-
"""
EXP-005 — o que a atencao esta olhando?

O EXP-004b mostrou QUE a atencao melhora a probabilidade. Este experimento
pergunta O QUE ela usa para isso — o pedido nº 6 do feedback do professor.

A pergunta e falsificavel e por isso vale a pena:

    O modelo concentra atencao no GOLEIRO e nos BLOQUEADORES sem nunca ter
    recebido "este e o goleiro adversario na frente do chute" como atributo?

O token de cada jogador traz geometria e um indicador de papel, mas NUNCA a
informacao "este jogador esta atrapalhando este chute". Se a atencao do [CLS]
se concentrar justamente neles, o modelo redescobriu sozinho o que a literatura
de xG codifica a mao.

Metrica: razao de atencao = atencao recebida / atencao uniforme (1/n_visiveis).
Valor 1,0 significa "olhado como um jogador qualquer da cena"; 2,0 significa
"olhado o dobro do que mereceria por acaso".

Uso:  python exp005_atencao.py     (treina 1 semente se ainda nao houver pesos)
"""
import json
import os

import numpy as np
import torch

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-005")
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
PESOS = os.path.join(DIR_EXP, "former_seed0.pt")
SEED = 0

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
bruto = np.load(os.path.join(AQUI, "shots_all.npz"), allow_pickle=True)

te = D["te"]
print(f"teste: {te.sum()} chutes")

modelo, do_cache = xb.treina_e_guarda(xb.Former(), D, SEED, PESOS, criterio="brier")
print("pesos " + ("carregados do disco" if do_cache else "treinados e salvos"))

# ------------------------------------------------ atencao no conjunto de teste
X = torch.tensor(D["tok"][te]); PAD = torch.tensor(D["pad"][te])
logits, attn = [], []
for i in range(0, len(X), 512):
    lo, at = xb.atencao_do_cls(modelo, X[i:i + 512], PAD[i:i + 512])
    logits.append(lo); attn.append(at)
logit = torch.cat(logits).numpy()
attn = torch.cat(attn).numpy()          # [N, camadas, cabecas, 23]
print(f"atencao extraida: {attn.shape}  (forward replicado confere com o modelo)")

# media entre camadas e cabecas; descarta a coluna do proprio [CLS] e do chutador
a = attn.mean(axis=(1, 2))              # [N, 23]
a_jog = a[:, 2:]                        # 21 jogadores do freeze-frame
a_jog = a_jog / np.maximum(a_jog.sum(1, keepdims=True), 1e-9)   # renormaliza

# ------------------------------------------------------------- papeis --------
mask = bruto["mask"][te] > 0.5
mate = bruto["mate"][te] > 0.5
gk = bruto["gk"][te] > 0.5
px, py = bruto["px"][te], bruto["py"][te]
sx, sy = bruto["sx"][te], bruto["sy"][te]

d1 = (36 - sy[:, None]) * (px - sx[:, None]) - (120 - sx[:, None]) * (py - sy[:, None])
d2 = (44 - sy[:, None]) * (px - sx[:, None]) - (120 - sx[:, None]) * (py - sy[:, None])
no_triangulo = (d1 * d2 < 0) & (px > sx[:, None])

GRUPOS = {
    "Goleiro adversário": mask & ~mate & gk,
    "Bloqueador\n(adversário na linha do chute)": mask & ~mate & ~gk & no_triangulo,
    "Outro adversário": mask & ~mate & ~gk & ~no_triangulo,
    "Companheiro": mask & mate,
}

n_vis = mask.sum(1)
uniforme = 1.0 / np.maximum(n_vis, 1)

resultados = {}
print("\n=== razão de atenção (1,0 = olhado como um jogador qualquer) ===")
for nome, sel in GRUPOS.items():
    tem = sel.any(1)
    razoes = np.array([a_jog[i, sel[i]].mean() / uniforme[i]
                       for i in np.flatnonzero(tem)])
    resultados[nome.replace("\n", " ")] = {
        "razao_media": float(razoes.mean()),
        "razao_mediana": float(np.median(razoes)),
        "n_cenas": int(tem.sum()),
    }
    print(f"  {nome.replace(chr(10),' '):46s} {razoes.mean():5.2f}x   "
          f"(mediana {np.median(razoes):.2f}x, {tem.sum()} cenas)")

with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-005-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump({"id": "EXP-005", "semente": SEED, "n_teste": int(te.sum()),
               "grupos": resultados}, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figura -----
fig, ax = viz.figura(largura=9.2, altura=4.2)
nomes = list(GRUPOS.keys())
vals = [resultados[n.replace("\n", " ")]["razao_media"] for n in nomes]
ys = list(range(len(nomes)))[::-1]
cores = [viz.SERIE[1], viz.SERIE[3], viz.SERIE[0], viz.SERIE[2]]

ax.axvline(1.0, color=viz.COR["tinta"], lw=1.6, zorder=4)
# anotacao entre o 3o e o 4o item, longe do subtitulo
ax.annotate("atenção que o jogador\nreceberia por acaso", (1.0, 0.55),
            textcoords="offset points", xytext=(10, 0), fontsize=8,
            color=viz.COR["tinta_2"], va="center")
for nome, y_, v, c in zip(nomes, ys, vals, cores):
    ax.plot([1.0, v], [y_, y_], color=c, lw=3, solid_capstyle="round", zorder=2)
    ax.plot([v], [y_], "o", ms=11, color=c, mec=viz.COR["superficie"], mew=2, zorder=3)
    ax.annotate(f"{v:.2f}x".replace(".", ","), (v, y_), textcoords="offset points",
                xytext=(14, 0), va="center", fontsize=10, fontweight="bold",
                color=viz.COR["tinta"])
ax.set_yticks(ys); ax.set_yticklabels(nomes, fontsize=9)
ax.set_ylim(-0.6, len(nomes) - 0.4)
ax.set_xlim(0, max(vals) * 1.25)
ax.set_title("Para quem o token [CLS] olha?", color=viz.COR["tinta"], pad=28)
ax.annotate("razão entre a atenção recebida e a que o jogador receberia por acaso",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("razão de atenção")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
viz.rodape(fig, "EXP-005 · Transformer semente 0 · média das 2 camadas e 4 cabeças · "
                f"{te.sum():,} chutes de teste. O modelo nunca recebeu "
                "\"este jogador atrapalha este chute\" como atributo."
           .replace(",", "."))
fig.tight_layout()
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-005-atencao-papeis.png"))

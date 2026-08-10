# -*- coding: utf-8 -*-
"""
EXP-010 — separar "representacao por token" de "nao-linearidade".

O degrau B2 -> Deep Sets, o maior da escada, acrescenta DUAS coisas ao mesmo
tempo: a cena completa E a nao-linearidade, porque o B2 e uma regressao
logistica linear. Comparar um modelo linear com redes neurais e comparar
tambem classes de modelo, nao so representacoes.

Aqui um gradient boosting e treinado sobre EXATAMENTE os mesmos 7 atributos
manuais do B2 — mesmo espaco de informacao, classe de modelo diferente. Ele e
o modelo padrao da literatura aplicada de xG, nao um adversario exotico.

Uso:  python exp010_baseline_nao_linear.py
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
tr, va, te = D["tr"], D["va"], D["te"]
y_tr, y_va, y_te = D["goal"][tr], D["goal"][va], D["goal"][te]
F2 = D["F2"]

print(f"treino {tr.sum()} | validação {va.sum()} | teste {te.sum()}")
print(f"atributos: {F2.shape[1]} (os mesmos do B2)\n")

# O numero de arvores e escolhido pela NOSSA validacao (split por partida).
# Deixar o sklearn separar sua propria fracao de validacao embaralharia chutes
# da mesma partida entre treino e validacao — o vazamento que o split existe
# para evitar. Por isso a busca e explicita, e o treino usa so o conjunto de
# treino, como fizeram os modelos neurais.
melhor = (None, np.inf, None)
for n_arvores in (50, 100, 200, 300, 500, 800):
    g = HistGradientBoostingClassifier(
        max_iter=n_arvores, learning_rate=0.06, max_leaf_nodes=31,
        early_stopping=False, random_state=0).fit(F2[tr], y_tr)
    b_val = xb.brier(y_va, g.predict_proba(F2[va])[:, 1])
    print(f"  {n_arvores:4d} árvores: Brier de validação {b_val:.5f}")
    if b_val < melhor[1]:
        melhor = (n_arvores, b_val, g)
n_arvores, b_val, gbm = melhor
print(f"escolhido pela validação: {n_arvores} árvores (Brier {b_val:.5f})")
p_gbm = gbm.predict_proba(F2[te])[:, 1]

# B1 e B2 lineares, para a escada ficar completa
lineares = xb.logisticas(D)

# previsoes neurais ja salvas (EXP-004, 5 sementes)
e4 = np.load(os.path.join(RAIZ, "experiments", "EXP-004", "predicoes.npz"))
assert np.array_equal(e4["y"], y_te), "teste divergente entre experimentos"

MODELOS = {
    "B1 · logística": lineares["B1"],
    "B2 · logística + interação": lineares["B2"],
    "B2-GBM · mesmos 7 atributos, não-linear": p_gbm,
    "DS · Deep Sets (tokens)": e4["DS_ensemble"],
    "TF · Transformer (tokens)": e4["TF_ensemble"],
}

res = {}
print(f"\n{'modelo':42s} {'AUC':>8s} {'Brier':>9s} {'ECE':>8s}")
for k, p in MODELOS.items():
    m = xb.metricas(y_te, p)
    res[k] = m
    print(f"{k:42s} {m['auc']:8.4f} {m['brier']:9.5f} {m['ece']:8.4f}")

# quanto do salto B2 -> DS e apenas nao-linearidade?
a_b2, a_gbm, a_ds = (res["B2 · logística + interação"]["auc"],
                     res["B2-GBM · mesmos 7 atributos, não-linear"]["auc"],
                     res["DS · Deep Sets (tokens)"]["auc"])
b_b2, b_gbm, b_ds = (res["B2 · logística + interação"]["brier"],
                     res["B2-GBM · mesmos 7 atributos, não-linear"]["brier"],
                     res["DS · Deep Sets (tokens)"]["brier"])
frac_auc = (a_gbm - a_b2) / (a_ds - a_b2)
frac_brier = (b_b2 - b_gbm) / (b_b2 - b_ds)
print(f"\ndo salto B2 -> Deep Sets, a não-linearidade sozinha explica:")
print(f"   AUC   {frac_auc*100:5.1f}%")
print(f"   Brier {frac_brier*100:5.1f}%")

with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-010-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump({"id": "EXP-010", "n_teste": int(te.sum()),
               "n_arvores_escolhido": int(n_arvores), "brier_validacao": float(b_val),
               "modelos": res,
               "fracao_do_salto_explicada_por_nao_linearidade":
                   {"auc": float(frac_auc), "brier": float(frac_brier)}},
              f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figura ----
ORDEM = list(MODELOS)
CORES = [viz.SERIE[0], viz.SERIE[1], viz.COR["suave"], viz.SERIE[2], viz.SERIE[3]]
fig, axs = viz.figura(largura=12.6, altura=4.6, colunas=2)
for ax, met, titulo, menor in ((axs[0], "auc", "AUC", False),
                               (axs[1], "brier", "Brier", True)):
    vals = [res[k][met] for k in ORDEM]
    ys = list(range(len(ORDEM)))[::-1]
    for k, y_, v, c in zip(ORDEM, ys, vals, CORES):
        ax.plot([v], [y_], "o", ms=10, color=c, mec=viz.COR["superficie"],
                mew=2, zorder=3)
        casas = 4 if met == "auc" else 5
        ax.annotate(f"{v:.{casas}f}".replace(".", ","), (v, y_),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=9, fontweight="bold", color=viz.COR["tinta"])
    ax.set_yticks(ys)
    ax.set_yticklabels(ORDEM if met == "auc" else [""] * len(ORDEM), fontsize=8.5)
    ax.set_ylim(-0.7, len(ORDEM) - 0.3)
    ax.set_title(titulo + ("  ·  menor é melhor" if menor else "  ·  maior é melhor"),
                 color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    folga = (max(vals) - min(vals)) * 0.16
    ax.set_xlim(min(vals) - folga, max(vals) + folga)
axs[0].annotate("o GBM usa os MESMOS atributos do B2 — muda a classe de modelo, "
                "não a informação",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle("EXP-010 · quanto do salto é representação e quanto é não-linearidade?",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-010 · teste = {int(te.sum())} chutes · modelos neurais: "
                "combinação de 5 sementes (EXP-004) · a não-linearidade sozinha "
                f"explica {frac_auc*100:.0f}% do salto em AUC e "
                f"{frac_brier*100:.0f}% em Brier.")
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-010-nao-linear.png"))

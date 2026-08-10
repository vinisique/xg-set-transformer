# -*- coding: utf-8 -*-
"""
EXP-011 — robustez da conclusao: por semente e por subgrupo.

Duas fragilidades apontadas na revisao critica, ambas respondiveis com as
previsoes ja salvas — sem retreinar nada.

(A) INFERENCIA POR SEMENTE. O teste do EXP-004 compara uma combinacao de 5
    sementes contra outra, e reamostra partidas. Ele propaga a variabilidade
    dos DADOS, mas nao a de INICIALIZACAO. Aqui o mesmo teste e refeito
    semente a semente: se a diferenca so aparece na combinacao, isso precisa
    estar no relatorio.

(B) SUBGRUPOS. O relatorio justifica misturar futebol masculino e feminino
    dizendo que as taxas de gol sao parecidas. Isso e invalido: taxa marginal
    igual nao implica relacao condicional igual. O que sustenta a decisao e
    verificar se a conclusao vale nos dois recortes — e e isso que falta.

Uso:  python exp011_robustez.py
"""
import json
import os

import numpy as np

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 2000

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
te = D["te"]
y = D["goal"][te]
partidas = D["match_id"][te]
comp = D["comp"][te]

e4 = np.load(os.path.join(RAIZ, "experiments", "EXP-004", "predicoes.npz"))
assert np.array_equal(e4["y"], y)

# ======================================================== (A) por semente ====
print("=== (A) teste pareado SEMENTE A SEMENTE ===")
print(f"{'semente':9s} {'Brier DS':>10s} {'Brier TF':>10s} {'diferença':>11s} "
      f"{'IC 95%':>26s} {'p':>7s}")
por_semente = []
for s in SEEDS:
    ds, tf = e4[f"DS_seed{s}"], e4[f"TF_seed{s}"]
    t = xb.bootstrap_pareado(y, tf, ds, partidas, fn=xb.brier, n=N_BOOT, seed=s)
    cruza = t["ic95"][0] <= 0 <= t["ic95"][1]
    por_semente.append({"semente": s, "brier_ds": xb.brier(y, ds),
                        "brier_tf": xb.brier(y, tf),
                        "diferenca": t["diferenca"], "ic95": t["ic95"],
                        "p": t["p_bilateral"], "cruza_zero": bool(cruza)})
    print(f"{s:^9d} {xb.brier(y,ds):10.5f} {xb.brier(y,tf):10.5f} "
          f"{t['diferenca']:+11.5f} "
          f"[{t['ic95'][0]:+.5f}; {t['ic95'][1]:+.5f}] {t['p_bilateral']:7.3f}"
          f"  {'cruza zero' if cruza else 'não cruza'}")

n_cruza = sum(x["cruza_zero"] for x in por_semente)
n_negat = sum(x["diferenca"] < 0 for x in por_semente)
print(f"\n  sinal favorece o Transformer em {n_negat}/5 sementes")
print(f"  intervalo cruza zero em {n_cruza}/5 sementes")

# ========================================================== (B) subgrupos ====
print("\n=== (B) conclusão por subgrupo ===")
fem = np.array(["Women" in c or "Liga F" in c or "NWSL" in c or "FA Women" in c
                for c in comp])
coorte1516 = np.array(["2015/2016" in c for c in comp])

GRUPOS = {
    "futebol feminino": fem,
    "futebol masculino": ~fem,
    "temporadas 2015/16": coorte1516,
    "demais competições": ~coorte1516,
}
subgrupos = {}
print(f"{'recorte':22s} {'chutes':>8s} {'gols':>6s} {'dif. Brier':>12s} "
      f"{'IC 95%':>26s}")
for nome, m in GRUPOS.items():
    t = xb.bootstrap_pareado(y[m], e4["TF_ensemble"][m], e4["DS_ensemble"][m],
                             partidas[m], fn=xb.brier, n=N_BOOT, seed=0)
    cruza = t["ic95"][0] <= 0 <= t["ic95"][1]
    subgrupos[nome] = {"n": int(m.sum()), "gols": int(y[m].sum()),
                       "diferenca": t["diferenca"], "ic95": t["ic95"],
                       "p": t["p_bilateral"], "cruza_zero": bool(cruza)}
    print(f"{nome:22s} {m.sum():8d} {int(y[m].sum()):6d} {t['diferenca']:+12.5f} "
          f"[{t['ic95'][0]:+.5f}; {t['ic95'][1]:+.5f}]"
          f"  {'cruza zero' if cruza else 'não cruza'}")

with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-011-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump({"id": "EXP-011", "n_bootstrap": N_BOOT,
               "por_semente": por_semente, "subgrupos": subgrupos},
              f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figura ----
fig, axs = viz.figura(largura=12.4, altura=4.4, colunas=2)

ax = axs[0]
for i, x in enumerate(por_semente):
    y_ = len(por_semente) - 1 - i
    lo, hi = x["ic95"]
    cor = viz.COR["suave"] if x["cruza_zero"] else viz.SERIE[3]
    ax.plot([lo, hi], [y_, y_], color=cor, lw=2.5, solid_capstyle="round", zorder=2)
    ax.plot([x["diferenca"]], [y_], "o", ms=9, color=cor,
            mec=viz.COR["superficie"], mew=2, zorder=3)
ax.axvline(0, color=viz.COR["tinta"], lw=1.6, zorder=4)
ax.set_yticks(range(len(por_semente))[::-1])
ax.set_yticklabels([f"semente {x['semente']}" for x in por_semente])
ax.set_ylim(-0.6, len(por_semente) - 0.4)
ax.set_title("Cada semente, isoladamente", color=viz.COR["tinta"], pad=28)
ax.annotate(f"laranja = intervalo não cruza zero ({5-n_cruza} de 5)",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("Brier(TF) − Brier(DS)")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)

ax = axs[1]
nomes = list(GRUPOS)
for i, nome in enumerate(nomes):
    x = subgrupos[nome]
    y_ = len(nomes) - 1 - i
    lo, hi = x["ic95"]
    cor = viz.COR["suave"] if x["cruza_zero"] else viz.SERIE[0]
    ax.plot([lo, hi], [y_, y_], color=cor, lw=2.5, solid_capstyle="round", zorder=2)
    ax.plot([x["diferenca"]], [y_], "o", ms=9, color=cor,
            mec=viz.COR["superficie"], mew=2, zorder=3)
ax.axvline(0, color=viz.COR["tinta"], lw=1.6, zorder=4)
ax.set_yticks(range(len(nomes))[::-1])
ax.set_yticklabels([f"{n}\n{subgrupos[n]['n']} chutes" for n in nomes], fontsize=8.5)
ax.set_ylim(-0.6, len(nomes) - 0.4)
ax.set_title("Por subgrupo (combinação das 5 sementes)",
             color=viz.COR["tinta"], pad=28)
ax.annotate("à esquerda de zero = Transformer melhor",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("Brier(TF) − Brier(DS)")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)

fig.suptitle("EXP-011 · a conclusão sobrevive a recortes mais exigentes?",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-011 · bootstrap pareado agrupado por partida, "
                f"{N_BOOT} reamostras · barras = IC 95%.")
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-011-robustez.png"))

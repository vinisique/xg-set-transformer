# -*- coding: utf-8 -*-
"""
EXP-008 — controle de capacidade: a vantagem do Transformer e atencao ou tamanho?

A revisao critica apontou a lacuna mais seria do protocolo: o Transformer tem
38.737 parametros contra 10.273 do Deep Sets original — 3,8 vezes mais. A
diferenca entre os dois nao isola a atencao, isola "atencao mais capacidade".

Este experimento treina um Deep Sets com dim=96, que da 38.977 parametros
(100,6% do Transformer). Se a vantagem do Transformer sobreviver, ela e da
atencao. Se desaparecer, era capacidade — e a conclusao do trabalho muda.

As previsoes do Transformer sao REAPROVEITADAS do EXP-004 (mesmas 5 sementes,
mesmo protocolo, mesmo split), entao so o Deep Sets largo precisa treinar.

Uso:  python exp008_capacidade.py
"""
import json
import os
import time

import numpy as np

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-008")
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
os.makedirs(DIR_EXP, exist_ok=True)
CACHE = os.path.join(DIR_EXP, "predicoes.npz")

SEEDS = (0, 1, 2, 3, 4)
DIM = 96

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
y_te = D["goal"][D["te"]]
partidas_te = D["match_id"][D["te"]]

n_ds = sum(p.numel() for p in xb.DeepSets(dim=DIM).parameters())
n_tf = sum(p.numel() for p in xb.Former().parameters())
print(f"Deep Sets largo (dim={DIM}): {n_ds:,} parâmetros")
print(f"Transformer:                {n_tf:,} parâmetros  "
      f"({n_ds/n_tf*100:.1f}% de equiparação)")
print(f"teste: {len(y_te)} chutes em {len(np.unique(partidas_te))} partidas\n")

# ------------------------------------------------------------- treinos ------
guardado = {}
if os.path.exists(CACHE):
    d = np.load(CACHE)
    guardado = {k: d[k] for k in d.files}
    print(f"cache: {len([k for k in guardado if 'seed' in k])} sementes já treinadas")

preds = []
for s in SEEDS:
    nome = f"DSL_seed{s}"
    if nome in guardado:
        p = guardado[nome]
        print(f"  DS-largo seed {s}: (do cache)", flush=True)
    else:
        t0 = time.time()
        p = xb.treina(xb.DeepSets(dim=DIM), D, s, criterio="brier")
        m = xb.metricas(y_te, p)
        print(f"  DS-largo seed {s}: Brier={m['brier']:.5f}  AUC={m['auc']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        guardado[nome] = p
        guardado["y"] = y_te
        np.savez_compressed(CACHE, **guardado)   # salva a cada semente
    preds.append(p)

ens_dsl = np.mean(preds, axis=0)

# --------------------------------- previsoes do TF e do DS original (EXP-004) -
e4 = np.load(os.path.join(RAIZ, "experiments", "EXP-004", "predicoes.npz"))
assert np.array_equal(e4["y"], y_te), "conjunto de teste divergente entre experimentos"
tf_seeds = [e4[f"TF_seed{s}"] for s in SEEDS]
ds_seeds = [e4[f"DS_seed{s}"] for s in SEEDS]
ens_tf, ens_ds = e4["TF_ensemble"], e4["DS_ensemble"]

# ------------------------------------------------------------- resultados ----
import statistics as st

por_seed = {
    "DS (dim 48)": [xb.metricas(y_te, p) for p in ds_seeds],
    "DS-largo (dim 96)": [xb.metricas(y_te, p) for p in preds],
    "TF (dim 48)": [xb.metricas(y_te, p) for p in tf_seeds],
}
ensembles = {"DS (dim 48)": ens_ds, "DS-largo (dim 96)": ens_dsl, "TF (dim 48)": ens_tf}

print("\n=== por semente (média ± dp das 5) ===")
print(f"{'modelo':20s} {'params':>8s} {'Brier':>18s} {'AUC':>16s}")
PARAMS = {"DS (dim 48)": 10273, "DS-largo (dim 96)": n_ds, "TF (dim 48)": n_tf}
for k, ms in por_seed.items():
    b = [m["brier"] for m in ms]; a = [m["auc"] for m in ms]
    print(f"{k:20s} {PARAMS[k]:8,} {st.mean(b):9.5f} ±{st.stdev(b):.5f} "
          f"{st.mean(a):8.4f} ±{st.stdev(a):.4f}")

print("\n=== combinação das 5 sementes ===")
for k, p in ensembles.items():
    m = xb.metricas(y_te, p)
    print(f"{k:20s} Brier {m['brier']:.5f} | AUC {m['auc']:.4f} | ECE {m['ece']:.4f}")

# ---------------------------------------------------- testes pareados --------
print("\n=== bootstrap pareado agrupado por partida (2.000 reamostras) ===")
testes = {}
for rot, a, b in (("TF menos DS-largo", ens_tf, ens_dsl),
                  ("TF menos DS original", ens_tf, ens_ds),
                  ("DS-largo menos DS original", ens_dsl, ens_ds)):
    t = xb.bootstrap_pareado(y_te, a, b, partidas_te, fn=xb.brier, n=2000, seed=0)
    testes[rot] = {k: v for k, v in t.items() if k != "distribuicao"}
    cruza = t["ic95"][0] <= 0 <= t["ic95"][1]
    print(f"  Brier · {rot:28s} {t['diferenca']:+.5f}  "
          f"IC95% [{t['ic95'][0]:+.5f}; {t['ic95'][1]:+.5f}]  p={t['p_bilateral']:.3f}"
          f"   -> {'NAO estabelecida' if cruza else 'estabelecida'}")

resumo = {"id": "EXP-008", "dim_ds_largo": DIM, "parametros": PARAMS,
          "seeds": list(SEEDS),
          "por_seed": {k: v for k, v in por_seed.items()},
          "ensemble": {k: xb.metricas(y_te, p) for k, p in ensembles.items()},
          "testes_pareados": testes}
with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-008-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figura -----
fig, axs = viz.figura(largura=12.0, altura=4.2, colunas=2)
ORDEM = ["DS (dim 48)", "DS-largo (dim 96)", "TF (dim 48)"]
CORES = {"DS (dim 48)": viz.SERIE[2], "DS-largo (dim 96)": viz.SERIE[0],
         "TF (dim 48)": viz.SERIE[3]}
for ax, met, titulo, menor in ((axs[0], "brier", "Brier por semente", True),
                               (axs[1], "auc", "AUC por semente", False)):
    for i, k in enumerate(ORDEM):
        vals = [m[met] for m in por_seed[k]]
        y_ = len(ORDEM) - 1 - i
        ax.plot(vals, [y_] * len(vals), "o", ms=9, color=CORES[k],
                mec=viz.COR["superficie"], mew=2, zorder=3, alpha=0.9)
        ax.plot([st.mean(vals)], [y_], "|", ms=26, mew=2.5,
                color=viz.COR["tinta"], zorder=4)
        casas = 5 if met == "brier" else 4
        ax.annotate(f"{st.mean(vals):.{casas}f}".replace(".", ","),
                    (st.mean(vals), y_), textcoords="offset points",
                    xytext=(0, -22), ha="center", fontsize=8.5,
                    fontweight="bold", color=viz.COR["tinta"])
    ax.set_yticks(range(len(ORDEM))[::-1])
    ax.set_yticklabels([f"{k}\n{PARAMS[k]:,} par.".replace(",", ".") for k in ORDEM],
                       fontsize=8)
    ax.set_ylim(-0.8, len(ORDEM) - 0.3)
    ax.set_title(titulo + ("  ·  menor é melhor" if menor else "  ·  maior é melhor"),
                 color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    todos = [m[met] for k in ORDEM for m in por_seed[k]]
    folga = (max(todos) - min(todos)) * 0.18
    ax.set_xlim(min(todos) - folga, max(todos) + folga)

axs[0].annotate("o Deep Sets largo tem a MESMA capacidade do Transformer",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle("EXP-008 · a vantagem do Transformer é atenção ou capacidade?",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-008 · 5 sementes por modelo · teste = {len(y_te):,} chutes · "
                "traço vertical = média. Deep Sets largo: dim 96 "
                f"({n_ds:,} parâmetros contra {n_tf:,} do Transformer)."
           .replace(",", "."))
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-008-capacidade.png"))
print("\nEXP-008 concluído.")

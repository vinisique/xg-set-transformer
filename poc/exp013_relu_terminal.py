# -*- coding: utf-8 -*-
"""
EXP-013 — o Deep Sets esta handicapado pela arquitetura, e nao pela falta de
atencao?

Suspeita levantada na revisao: o `phi` do Deep Sets TERMINA em ReLU, de modo que
toda representacao por jogador e nao-negativa. A media e o maximo agregam so
valores >= 0, e a cabeca linear recebe um vetor no ortante positivo.

O Transformer nao tem essa restricao: a saida do [CLS] vem do fluxo residual,
com sinal livre.

Se for isso, a comparacao TF vs DS mede "atencao + liberdade de sinal", nao so
atencao — e o EXP-008 nao capturaria, porque dobrar a dimensao nao remove a
restricao.

Este experimento remove a ReLU terminal do Deep Sets e mais nada. Se o modelo
alcancar o Transformer, a conclusao do trabalho muda. Se nao alcancar, a
suspeita e afastada e a comparacao fica mais solida do que estava.

Uso:  python exp013_relu_terminal.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-013")
os.makedirs(DIR_EXP, exist_ok=True)
CACHE = os.path.join(DIR_EXP, "predicoes.npz")
SEEDS = (0, 1, 2, 3, 4)


class DeepSetsSemReLU(nn.Module):
    """Identico ao xb.DeepSets, EXCETO pela ReLU final do phi.

    Unica diferenca em relacao ao original — proposital: se mudassemos duas
    coisas, o resultado nao atribuiria causa a nenhuma delas.
    """

    def __init__(self, dim=48, drop=0.1, n_feat=14):
        super().__init__()
        self.proj = nn.Linear(n_feat, dim)
        self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(dim * 2, dim))  # sem ReLU
        self.head = nn.Linear(dim * 2, 1)

    def forward(self, x, pad):
        h = self.phi(self.proj(x))
        w = (~pad).float().unsqueeze(-1)
        mean = (h * w).sum(1) / w.sum(1)
        mx = h.masked_fill(pad.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
y_te = D["goal"][D["te"]]
partidas_te = D["match_id"][D["te"]]

n_orig = sum(p.numel() for p in xb.DeepSets().parameters())
n_novo = sum(p.numel() for p in DeepSetsSemReLU().parameters())
print(f"parâmetros: original {n_orig:,} · sem ReLU terminal {n_novo:,} "
      f"({'idênticos' if n_orig == n_novo else 'DIFERENTES'})")

guardado = {}
if os.path.exists(CACHE):
    d = np.load(CACHE)
    guardado = {k: d[k] for k in d.files}

preds = []
for s in SEEDS:
    nome = f"DSsr_seed{s}"
    if nome in guardado:
        p = guardado[nome]
        print(f"  seed {s}: (do cache)", flush=True)
    else:
        t0 = time.time()
        p = xb.treina(DeepSetsSemReLU(), D, s, criterio="brier")
        m = xb.metricas(y_te, p)
        print(f"  seed {s}: Brier={m['brier']:.5f} AUC={m['auc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        guardado[nome] = p
        guardado["y"] = y_te
        np.savez_compressed(CACHE, **guardado)
    preds.append(p)

ens = np.mean(preds, axis=0)
e4 = np.load(os.path.join(RAIZ, "experiments", "EXP-004", "predicoes.npz"))
assert np.array_equal(e4["y"], y_te)

import statistics as st
por_seed = {
    "DS original": [xb.metricas(y_te, e4[f"DS_seed{s}"]) for s in SEEDS],
    "DS sem ReLU terminal": [xb.metricas(y_te, p) for p in preds],
    "TF": [xb.metricas(y_te, e4[f"TF_seed{s}"]) for s in SEEDS],
}
ENS = {"DS original": e4["DS_ensemble"], "DS sem ReLU terminal": ens,
       "TF": e4["TF_ensemble"]}

print(f"\n{'modelo':24s} {'Brier (média±dp)':>22s} {'AUC (média±dp)':>20s}")
for k, ms in por_seed.items():
    b = [m["brier"] for m in ms]; a = [m["auc"] for m in ms]
    print(f"{k:24s} {st.mean(b):11.5f} ±{st.stdev(b):.5f} "
          f"{st.mean(a):11.4f} ±{st.stdev(a):.4f}")

print("\n=== bootstrap pareado agrupado por partida ===")
testes = {}
for rot, a, b in (("TF menos DS sem ReLU", ENS["TF"], ENS["DS sem ReLU terminal"]),
                  ("DS sem ReLU menos DS original",
                   ENS["DS sem ReLU terminal"], ENS["DS original"])):
    t = xb.bootstrap_pareado(y_te, a, b, partidas_te, fn=xb.brier, n=2000, seed=0)
    testes[rot] = {k: v for k, v in t.items() if k != "distribuicao"}
    cruza = t["ic95"][0] <= 0 <= t["ic95"][1]
    print(f"  {rot:32s} {t['diferenca']:+.5f} "
          f"IC95% [{t['ic95'][0]:+.5f}; {t['ic95'][1]:+.5f}] p={t['p_bilateral']:.3f}"
          f"  -> {'NAO estabelecida' if cruza else 'estabelecida'}")

json.dump({"id": "EXP-013", "seeds": list(SEEDS),
           "parametros": {"DS original": n_orig, "DS sem ReLU terminal": n_novo},
           "por_seed": por_seed,
           "ensemble": {k: xb.metricas(y_te, v) for k, v in ENS.items()},
           "testes_pareados": testes},
          open(os.path.join(RAIZ, "docs", "experimentos", "EXP-013-completo.json"),
               "w", encoding="utf-8"), indent=2, ensure_ascii=False)

fig, axs = viz.figura(largura=12.0, altura=3.8, colunas=2)
ORDEM = ["DS original", "DS sem ReLU terminal", "TF"]
CORES = {"DS original": viz.SERIE[2], "DS sem ReLU terminal": viz.SERIE[0],
         "TF": viz.SERIE[3]}
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
    ax.set_yticklabels(ORDEM if met == "brier" else [""] * len(ORDEM), fontsize=8.5)
    ax.set_ylim(-0.8, len(ORDEM) - 0.3)
    ax.set_title(titulo + ("  ·  menor é melhor" if menor else "  ·  maior é melhor"),
                 color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    todos = [m[met] for k in ORDEM for m in por_seed[k]]
    folga = (max(todos) - min(todos)) * 0.18
    ax.set_xlim(min(todos) - folga, max(todos) + folga)
axs[0].annotate("única diferença entre as duas versões do Deep Sets: a ReLU final",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle("EXP-013 · o Deep Sets estava limitado pela ReLU terminal?",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-013 · 5 sementes por modelo · teste = {len(y_te)} chutes · "
                f"as duas versões do Deep Sets têm os mesmos {n_orig:,} parâmetros."
           .replace(",", "."))
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(RAIZ, "docs", "experimentos", "figuras",
                             "EXP-013-relu.png"))

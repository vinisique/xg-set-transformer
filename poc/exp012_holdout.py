# -*- coding: utf-8 -*-
"""
EXP-012 — hold-out por competicao: o modelo generaliza para o que nunca viu?

Todos os testes anteriores usam partidas SORTEADAS do mesmo conjunto de 80
competicoes-temporada. Isso mede robustez interna: garante que o modelo nao
decorou partidas. Nao garante que ele nao decorou estilo de liga e de epoca.

Aqui uma competicao inteira e removida do treino e da validacao, e usada como
teste. E o unico teste de generalizacao de verdade do trabalho.

Escolha da competicao: Premier League 2015/16, com 9.817 finalizacoes. E grande
o bastante para o teste ter poder — a Eurocopa 2024, com 1.304, daria intervalo
quase 4x mais largo, e o efeito medido (~0,0005 de Brier) sumiria dentro dele
por construcao, nao por ausencia.

Uso:  python exp012_holdout.py
"""
import json
import os
import time

import numpy as np

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-012")
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
os.makedirs(DIR_EXP, exist_ok=True)
CACHE = os.path.join(DIR_EXP, "predicoes.npz")

FORA = "Premier League 2015/2016"
# Sementes efetivamente treinadas. O Transformer ficou com 2 em vez de 3: a
# maquina suspendia durante a noite e o tempo de parede inviabilizou a terceira.
# A limitacao e declarada no relatorio em vez de silenciada.
SEEDS_POR_MODELO = {"DS": (0, 1, 2), "TF": (0, 1)}

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))

# --------------------------------------- redefine os splits: competicao fora --
alvo = D["comp"] == FORA
resto = ~alvo
partidas_resto = np.unique(D["match_id"][resto])
rng = np.random.default_rng(0)
rng.shuffle(partidas_resto)
corte = int(0.85 * len(partidas_resto))
D["tr"] = np.isin(D["match_id"], partidas_resto[:corte])
D["va"] = np.isin(D["match_id"], partidas_resto[corte:])
D["te"] = alvo

y_te = D["goal"][D["te"]]
partidas_te = D["match_id"][D["te"]]
print(f"fora do treino: {FORA}")
print(f"  treino {D['tr'].sum()} | validação {D['va'].sum()} | "
      f"teste {D['te'].sum()} ({len(np.unique(partidas_te))} partidas, "
      f"{int(y_te.sum())} gols)")
assert not (D["tr"] & D["te"]).any() and not (D["va"] & D["te"]).any()
print("  verificado: nenhuma finalização da competição vazou para treino/validação\n")

# ------------------------------------------------------------ baselines -----
lin = xb.logisticas(D)

# ------------------------------------------------------------- neurais ------
guardado = {}
if os.path.exists(CACHE):
    d = np.load(CACHE)
    guardado = {k: d[k] for k in d.files}

preds = {"DS": [], "TF": []}
for chave, classe in (("DS", xb.DeepSets), ("TF", xb.Former)):
    for s in SEEDS_POR_MODELO[chave]:
        nome = f"{chave}_seed{s}"
        if nome in guardado:
            p = guardado[nome]
            print(f"  {chave} seed {s}: (do cache)", flush=True)
        else:
            t0 = time.time()
            p = xb.treina(classe(), D, s, criterio="brier")
            m = xb.metricas(y_te, p)
            print(f"  {chave} seed {s}: Brier={m['brier']:.5f} AUC={m['auc']:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            guardado[nome] = p
            guardado["y"] = y_te
            np.savez_compressed(CACHE, **guardado)
        preds[chave].append(p)

ens = {k: np.mean(v, axis=0) for k, v in preds.items()}

# ------------------------------------------------------------ resultados ----
MODELOS = {"B1": lin["B1"], "B2": lin["B2"],
           "DS": ens["DS"], "TF": ens["TF"]}
res = {k: xb.metricas(y_te, p) for k, p in MODELOS.items()}

print(f"\n=== desempenho em {FORA} (competição nunca vista) ===")
print(f"{'modelo':8s} {'AUC':>8s} {'Brier':>9s} {'ECE':>8s}")
for k, m in res.items():
    print(f"{k:8s} {m['auc']:8.4f} {m['brier']:9.5f} {m['ece']:8.4f}")

t = xb.bootstrap_pareado(y_te, ens["TF"], ens["DS"], partidas_te,
                         fn=xb.brier, n=2000, seed=0)
cruza = t["ic95"][0] <= 0 <= t["ic95"][1]
print(f"\nTF − DS: {t['diferenca']:+.5f}  IC95% [{t['ic95'][0]:+.5f}; "
      f"{t['ic95'][1]:+.5f}]  p={t['p_bilateral']:.3f}"
      f"  -> {'NAO estabelecida' if cruza else 'estabelecida'}")

# calibracao agregada na competicao nao vista
somas = np.array([ens["TF"][partidas_te == u].sum() for u in np.unique(partidas_te)])
gols = np.array([y_te[partidas_te == u].sum() for u in np.unique(partidas_te)])
vies = (somas.sum() / gols.sum() - 1) * 100
print(f"calibração agregada: xG {somas.sum():.1f} contra {int(gols.sum())} gols "
      f"({vies:+.1f}%)")

json.dump({"id": "EXP-012", "competicao_fora": FORA,
           "sementes": {k: list(v) for k, v in SEEDS_POR_MODELO.items()},
           "n_teste": int(D["te"].sum()), "gols": int(y_te.sum()),
           "n_partidas": int(len(np.unique(partidas_te))),
           "modelos": res,
           "teste_pareado": {k: v for k, v in t.items() if k != "distribuicao"},
           "vies_agregado_pct": float(vies)},
          open(os.path.join(RAIZ, "docs", "experimentos", "EXP-012-completo.json"),
               "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figura ----
# comparacao com o desempenho DENTRO da distribuicao (EXP-010)
e10 = json.load(open(os.path.join(RAIZ, "docs", "experimentos",
                                  "EXP-010-completo.json"), encoding="utf-8"))
DENTRO = {"B1": "B1 · logística", "B2": "B2 · logística + interação",
          "DS": "DS · Deep Sets (tokens)", "TF": "TF · Transformer (tokens)"}

fig, axs = viz.figura(largura=12.0, altura=4.0, colunas=2)
ORDEM = ["B1", "B2", "DS", "TF"]
for ax, met, titulo, menor in ((axs[0], "auc", "AUC", False),
                               (axs[1], "brier", "Brier", True)):
    ys = list(range(len(ORDEM)))[::-1]
    for k, y_ in zip(ORDEM, ys):
        v_fora = res[k][met]
        v_dentro = e10["modelos"][DENTRO[k]][met]
        ax.plot([v_dentro], [y_], "o", ms=8, color=viz.COR["suave"],
                mec=viz.COR["superficie"], mew=1.5, zorder=3)
        ax.plot([v_fora], [y_], "o", ms=10, color=viz.COR_MODELO[k],
                mec=viz.COR["superficie"], mew=2, zorder=4)
        ax.plot([v_dentro, v_fora], [y_, y_], color=viz.COR["grade"], lw=1.5, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(ORDEM if met == "auc" else [""] * 4)
    ax.set_ylim(-0.6, len(ORDEM) - 0.4)
    ax.set_title(titulo + ("  ·  menor é melhor" if menor else "  ·  maior é melhor"),
                 color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
axs[0].annotate("cinza = teste habitual (partidas sorteadas) · "
                "colorido = competição nunca vista",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle(f"EXP-012 · generalização para uma competição inteira fora do treino",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-012 · fora do treino: {FORA} · {int(D['te'].sum())} chutes em "
                f"{len(np.unique(partidas_te))} partidas · Deep Sets 3 sementes, Transformer 2.")
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-012-holdout.png"))

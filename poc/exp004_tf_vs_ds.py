# -*- coding: utf-8 -*-
"""
EXP-004 — a pergunta central do projeto.

    A atencao par a par acrescenta alguma coisa sobre o Deep Sets?

Deep Sets e Transformer recebem EXATAMENTE os mesmos tokens. A unica diferenca
e que o Transformer pode comparar jogadores entre si. A diferenca entre os dois
isola, portanto, o valor da atencao — e nada mais.

O EXP-000 mediu +0,0014 de AUC a favor do Transformer com 2 seeds e sem teste
estatistico: um numero que nao sustenta afirmacao nenhuma. Aqui:

  - 5 seeds por modelo, para separar sinal de ruido de inicializacao;
  - selecao por Brier de validacao (cartao 0001), e nao por AUC;
  - bootstrap pareado AGRUPADO POR PARTIDA, porque chutes do mesmo jogo sao
    correlacionados e reamostrar chutes daria um IC estreito demais.

Uso:  python exp004_tf_vs_ds.py     (CPU: cerca de uma hora)
"""
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import viz
import xg_base as xb

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-004")
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
os.makedirs(DIR_EXP, exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)

D = xb.carrega(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots_all.npz"))
y_te = D["goal"][D["te"]]
partidas_te = D["match_id"][D["te"]]
print(f"teste: {len(y_te)} chutes | {len(np.unique(partidas_te))} partidas | "
      f"{y_te.mean()*100:.1f}% gols")

# ------------------------------------------------------------- treinos ----
preds = {"DS": [], "TF": []}
for chave, classe in (("DS", xb.DeepSets), ("TF", xb.Former)):
    for s in SEEDS:
        t0 = time.time()
        p = xb.treina(classe(), D, s, criterio="brier")
        preds[chave].append(p)
        m = xb.metricas(y_te, p)
        print(f"  {chave} seed {s}: Brier={m['brier']:.5f}  AUC={m['auc']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

ens = {k: np.mean(v, axis=0) for k, v in preds.items()}
np.savez_compressed(os.path.join(DIR_EXP, "predicoes.npz"), y=y_te,
                    partidas=partidas_te,
                    **{f"{k}_seed{s}": p for k, ps in preds.items()
                       for s, p in zip(SEEDS, ps)},
                    **{f"{k}_ensemble": v for k, v in ens.items()})

# ------------------------------------------------------ teste estatistico ----
print("\nbootstrap pareado agrupado por partida (2000 reamostras)...")
teste_brier = xb.bootstrap_pareado(y_te, ens["TF"], ens["DS"], partidas_te,
                                   fn=xb.brier, n=2000, seed=0)
teste_auc = xb.bootstrap_pareado(y_te, ens["TF"], ens["DS"], partidas_te,
                                 fn=roc_auc_score, n=2000, seed=0)

resumo = {
    "id": "EXP-004", "seeds": list(SEEDS), "n_teste": int(len(y_te)),
    "n_partidas_teste": int(len(np.unique(partidas_te))),
    "criterio_early_stopping": "brier (cartao 0001)",
    "por_seed": {k: [xb.metricas(y_te, p) for p in ps] for k, ps in preds.items()},
    "ensemble": {k: xb.metricas(y_te, v) for k, v in ens.items()},
    "teste_pareado": {
        "brier_TF_menos_DS": {kk: vv for kk, vv in teste_brier.items() if kk != "distribuicao"},
        "auc_TF_menos_DS": {kk: vv for kk, vv in teste_auc.items() if kk != "distribuicao"},
    },
}
with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-004-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)

print("\n===== EXP-004 =====")
for k in ("DS", "TF"):
    bs = [m["brier"] for m in resumo["por_seed"][k]]
    au = [m["auc"] for m in resumo["por_seed"][k]]
    print(f"{k}: Brier {np.mean(bs):.5f} ±{np.std(bs):.5f} | "
          f"AUC {np.mean(au):.4f} ±{np.std(au):.4f}")
tb = resumo["teste_pareado"]["brier_TF_menos_DS"]
print(f"\nBrier(TF) - Brier(DS) = {tb['diferenca']:+.5f}  "
      f"IC95% [{tb['ic95'][0]:+.5f}, {tb['ic95'][1]:+.5f}]  p={tb['p_bilateral']:.3f}")
print("(negativo favorece o Transformer, pois Brier menor e melhor)")

# -------------------------------------------------------------- figuras ----
# Figura 1 — distribuicao por seed: o ganho e maior que o ruido?
fig, axs = viz.figura(largura=12.0, altura=4.0, colunas=2)
for ax, met, titulo, sub in (
        (axs[0], "brier", "Brier por semente", "menor é melhor · métrica que decide (cartão 0001)"),
        (axs[1], "auc", "AUC por semente", "maior é melhor · métrica de ordenação")):
    for i, k in enumerate(["DS", "TF"]):
        vals = [m[met] for m in resumo["por_seed"][k]]
        y_ = 1 - i
        ax.plot(vals, [y_] * len(vals), "o", ms=10, color=viz.COR_MODELO[k],
                mec=viz.COR["superficie"], mew=2, zorder=3, alpha=0.9)
        ax.plot([np.mean(vals)], [y_], "|", ms=26, mew=2.5,
                color=viz.COR["tinta"], zorder=4)
        ax.annotate(f"média {np.mean(vals):.5f}".replace(".", ","),
                    (np.mean(vals), y_), textcoords="offset points",
                    xytext=(0, -24), ha="center", fontsize=8.5,
                    fontweight="bold", color=viz.COR["tinta"])
    ax.set_yticks([1, 0]); ax.set_yticklabels(["DS · Deep Sets", "TF · Transformer"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_title(titulo, color=viz.COR["tinta"], pad=28)
    ax.annotate(sub, (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
fig.suptitle("EXP-004 · o ganho da atenção separa-se do ruído entre sementes?",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, f"EXP-004 · {len(SEEDS)} sementes por modelo · teste = {len(y_te):,} chutes · "
                "traço vertical = média. Fonte: docs/experimentos/EXP-004-completo.json"
           .replace(",", "."))
fig.tight_layout(rect=[0, 0, 1, 0.93])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-004-seeds.png"))

# Figura 2 — distribuicao bootstrap da diferenca
fig, ax = viz.figura(largura=8.6, altura=4.0)
difs = teste_brier["distribuicao"]
ax.hist(difs, bins=60, color=viz.COR_MODELO["TF"], alpha=0.85,
        edgecolor=viz.COR["superficie"], linewidth=0.5, zorder=2)
ax.axvline(0, color=viz.COR["tinta"], lw=1.6, zorder=3)
ax.annotate("sem diferença", (0, ax.get_ylim()[1] * 0.95),
            textcoords="offset points", xytext=(6, 0), fontsize=8.5,
            color=viz.COR["tinta_2"], va="top")
for v, rot in ((teste_brier["ic95"][0], "IC 2,5%"), (teste_brier["ic95"][1], "IC 97,5%")):
    ax.axvline(v, color=viz.COR["eixo"], lw=1.2, zorder=3)
    ax.annotate(rot, (v, ax.get_ylim()[1] * 0.55), textcoords="offset points",
                xytext=(4, 0), fontsize=7.5, color=viz.COR["suave"])
ax.set_title("Diferença de Brier: Transformer − Deep Sets",
             color=viz.COR["tinta"], pad=28)
ax.annotate("2.000 reamostras de PARTIDAS com reposição · "
            "à esquerda de zero = Transformer melhor",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.set_xlabel("Brier(TF) − Brier(DS)"); ax.set_ylabel("reamostras")
ax.grid(axis="y"); ax.set_axisbelow(True)
viz.rodape(fig, f"EXP-004 · diferença observada {teste_brier['diferenca']:+.5f} · "
                f"IC95% [{teste_brier['ic95'][0]:+.5f}, {teste_brier['ic95'][1]:+.5f}] · "
                f"p={teste_brier['p_bilateral']:.3f}. Se o IC cruza zero, a diferença "
                "não está estabelecida.".replace(".", ","))
fig.tight_layout()
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-004-bootstrap.png"))

print("\nEXP-004 concluido.")

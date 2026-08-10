# -*- coding: utf-8 -*-
"""
EXP-006 — calibração das probabilidades e calibração agregada.

Fecha dois pedidos do feedback do professor que seguiam abertos:

  nº 1  calibrar as probabilidades (Platt / isotônica) e reportar o efeito
  nº 3  comparar o xG TOTAL previsto por partida com os gols realmente marcados

A parte agregada e a mais proxima do uso real: um analista nao pergunta "o
modelo ordenou bem?", pergunta "o time criou 1,8 de xG e fez 1 gol — o numero
faz sentido?". E um teste que a AUC nao consegue nem formular.

IMPORTANTE: os calibradores sao ajustados na VALIDACAO e aplicados ao TESTE.
Ajustar no teste inflaria o resultado — seria o proprio vazamento que o split
por partida existe para evitar.

Uso:  python exp006_calibracao.py
"""
import json
import os

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import viz
import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
DIR_EXP = os.path.join(RAIZ, "experiments", "EXP-006")
DIR_FIG = os.path.join(RAIZ, "docs", "experimentos", "figuras")
os.makedirs(DIR_EXP, exist_ok=True)
SEED = 0

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
y_va, y_te = D["goal"][D["va"]], D["goal"][D["te"]]
partidas_te = D["match_id"][D["te"]]

# ------------------------------------------------ previsoes de val e teste ---
# Reaproveita os pesos ja salvos quando existirem (o TF veio do EXP-005).
MODELOS = {"DS": (xb.DeepSets, os.path.join(DIR_EXP, "deepsets_seed0.pt")),
           "TF": (xb.Former, os.path.join(RAIZ, "experiments", "EXP-005",
                                          "former_seed0.pt"))}

X = torch.tensor(D["tok"]); PAD = torch.tensor(D["pad"])
iva, ite = np.flatnonzero(D["va"]), np.flatnonzero(D["te"])

previsoes = {}
for chave, (classe, caminho) in MODELOS.items():
    modelo, do_cache = xb.treina_e_guarda(classe(), D, SEED, caminho, criterio="brier")
    print(f"{chave}: pesos " + ("do disco" if do_cache else "treinados e salvos"))
    with torch.no_grad():
        pv = torch.sigmoid(modelo(X[iva], PAD[iva])).numpy()
        pt = torch.sigmoid(modelo(X[ite], PAD[ite])).numpy()
    previsoes[chave] = (pv, pt)

# ------------------------------------------------------ Parte 1: calibrar ----
resultados = {}
print("\n=== efeito da calibração (ajuste na validação, avaliação no teste) ===")
print(f"{'modelo':22s} {'Brier':>9s} {'ECE':>8s}")
for chave, (pv, pt) in previsoes.items():
    variantes = {"sem calibração": pt}

    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(np.log(pv / (1 - pv)).reshape(-1, 1), y_va)     # Platt no logito
    variantes["Platt"] = platt.predict_proba(
        np.log(pt / (1 - pt)).reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, y_va)
    variantes["isotônica"] = iso.predict(pt)

    resultados[chave] = {}
    for nome, p in variantes.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        m = xb.metricas(y_te, p)
        resultados[chave][nome] = m
        print(f"{chave + ' · ' + nome:22s} {m['brier']:9.5f} {m['ece']:8.4f}")
    previsoes[chave] = (pv, pt, variantes)

# --------------------------------------------- Parte 2: calibração agregada --
# Por partida: soma do xG previsto contra gols efetivamente marcados.
uniq = np.unique(partidas_te)
agregado = {}
print("\n=== calibração agregada por partida ===")
for chave in MODELOS:
    pt = previsoes[chave][2]["isotônica"]
    somas = np.array([pt[partidas_te == u].sum() for u in uniq])
    gols = np.array([y_te[partidas_te == u].sum() for u in uniq])
    vies = somas.sum() - gols.sum()
    agregado[chave] = {
        "xg_total": float(somas.sum()), "gols_total": int(gols.sum()),
        "vies_total": float(vies),
        "vies_relativo": float(vies / gols.sum()),
        "erro_medio_absoluto_por_partida": float(np.abs(somas - gols).mean()),
        "somas": somas, "gols": gols,
    }
    a = agregado[chave]
    print(f"  {chave}: xG total {a['xg_total']:.1f} contra {a['gols_total']} gols "
          f"({a['vies_relativo']*100:+.1f}%) · erro médio por partida "
          f"{a['erro_medio_absoluto_por_partida']:.2f} gol")

with open(os.path.join(RAIZ, "docs", "experimentos", "EXP-006-completo.json"),
          "w", encoding="utf-8") as f:
    json.dump({"id": "EXP-006", "semente": SEED,
               "calibracao": resultados,
               "agregado": {k: {kk: vv for kk, vv in v.items()
                                if kk not in ("somas", "gols")}
                            for k, v in agregado.items()}},
              f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figuras ----
# Figura A — efeito da calibração no Brier e no ECE
fig, axs = viz.figura(largura=12.0, altura=4.0, colunas=2)
VAR = ["sem calibração", "Platt", "isotônica"]
for ax, met, titulo in ((axs[0], "brier", "Brier"), (axs[1], "ece", "ECE")):
    for i, chave in enumerate(["DS", "TF"]):
        vals = [resultados[chave][v][met] for v in VAR]
        ys = [len(VAR) - 1 - j + (0.16 if i else -0.16) for j in range(len(VAR))]
        ax.plot(vals, ys, "o", ms=9, color=viz.COR_MODELO[chave],
                mec=viz.COR["superficie"], mew=2, zorder=3,
                label="Deep Sets" if chave == "DS" else "Transformer")
    ax.set_yticks(range(len(VAR))[::-1]); ax.set_yticklabels(VAR)
    ax.set_ylim(-0.6, len(VAR) - 0.4)
    ax.set_title(titulo + "  ·  menor é melhor", color=viz.COR["tinta"], pad=28)
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right")
axs[0].annotate("recalibrar melhora a probabilidade sem tocar na arquitetura",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
fig.suptitle("EXP-006 · efeito da calibração (ajustada na validação)",
             x=0.0, ha="left", fontsize=13, fontweight="bold", color=viz.COR["tinta"])
viz.rodape(fig, "EXP-006 · semente 0 · avaliação no conjunto de teste. "
                "Ajustar o calibrador no teste inflaria o resultado.")
fig.tight_layout(rect=[0, 0, 1, 0.90])
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-006-calibracao.png"))

# Figura B — calibração agregada: xG somado por partida contra gols
fig, ax = viz.figura(largura=7.6, altura=5.2)
a = agregado["TF"]
lim = max(a["somas"].max(), a["gols"].max()) * 1.05
ax.plot([0, lim], [0, lim], color=viz.COR["eixo"], lw=1.4, zorder=1)
ax.annotate("previsão perfeita", (lim * 0.72, lim * 0.72), rotation=38,
            fontsize=8, color=viz.COR["suave"], ha="center", va="bottom")
jitter = np.random.default_rng(0).normal(0, 0.06, len(a["gols"]))
ax.plot(a["somas"], a["gols"] + jitter, "o", ms=5, alpha=0.35,
        color=viz.COR_MODELO["TF"], mec="none", zorder=2)
# media observada por faixa de xG previsto
bins = np.quantile(a["somas"], np.linspace(0, 1, 9))
cx, cy = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (a["somas"] >= lo) & (a["somas"] < hi)
    if m.sum():
        cx.append(a["somas"][m].mean()); cy.append(a["gols"][m].mean())
ax.plot(cx, cy, "-o", color=viz.COR["tinta"], lw=2, ms=7,
        mec=viz.COR["superficie"], mew=1.5, zorder=4, label="média por faixa")
ax.set_xlabel("xG total previsto na partida")
ax.set_ylabel("gols efetivamente marcados")
ax.set_title("Calibração agregada por partida", color=viz.COR["tinta"], pad=28)
ax.annotate("cada ponto é uma partida do conjunto de teste "
            "(deslocamento vertical para separar os pontos)",
            (0, 1), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 8), fontsize=8.5, color=viz.COR["tinta_2"])
ax.grid(True); ax.set_axisbelow(True); ax.legend(loc="upper left")
viz.rodape(fig, f"EXP-006 · Transformer calibrado (isotônica) · {len(uniq)} partidas · "
                f"xG total {a['xg_total']:.0f} contra {a['gols_total']} gols "
                f"({a['vies_relativo']*100:+.1f}%).")
fig.tight_layout()
viz.salvar(fig, os.path.join(DIR_FIG, "EXP-006-agregada.png"))

# -*- coding: utf-8 -*-
r"""
Confere cada numero do relatorio contra o JSON do experimento que o produziu.

A regra do projeto e "nenhum numero no relatorio sem um ID de experimento".
Este script torna a regra verificavel: se alguem editar o .tex a mao e trocar
um digito, ou se um experimento for reexecutado e o texto ficar para tras, a
divergencia aparece aqui em vez de aparecer na arguicao.

Uso:  python valida_numeros.py
"""
import json
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(RAIZ, "relatorio", "artigo.tex")
DOCS = os.path.join(RAIZ, "docs", "experimentos")

texto = open(TEX, encoding="utf-8").read()


def carrega(nome):
    p = os.path.join(DOCS, nome)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def br(x, casas):
    """Numero no formato do texto: virgula decimal, sem sinal de mais."""
    return f"{abs(x):.{casas}f}".replace(".", ",")


falhas, ok, ausentes = [], 0, []


def formas(valor, casas):
    """Formas aceitaveis do numero no texto, incluindo separador de milhar."""
    base = br(valor, casas)
    saida = {base}
    if casas == 0 and abs(valor) >= 1000:                 # 14935 -> 14.935
        saida.add(f"{int(abs(valor)):,}".replace(",", "."))
    return saida


def confere(rotulo, valor, casas, fonte, obrigatorio=True):
    """Verifica que `valor`, formatado com `casas`, aparece no .tex.

    obrigatorio=False: o numero pode legitimamente nao ser citado no texto
    (nem todo valor medido precisa ir para o relatorio). Nesse caso a ausencia
    e apenas informada, nao contada como divergencia. O que NUNCA e tolerado e
    o texto trazer um valor DIFERENTE do medido.
    """
    global ok
    alt = formas(valor, casas)
    if any(a in texto for a in alt):
        ok += 1
    elif obrigatorio:
        falhas.append(f"{rotulo}: esperado {br(valor, casas)} (de {fonte}) "
                      "— NAO ENCONTRADO no .tex")
    else:
        ausentes.append(f"{rotulo} ({br(valor, casas)}) — medido, não citado")


# ---------------------------------------------------- EXP-000b (escada) -----
e0 = carrega("EXP-000-completo.json")
if e0:
    for k in ("B1", "B2", "DS", "TF"):
        m = e0["modelos"][k]["media_seeds"]
        confere(f"EXP-000b {k} AUC", m["auc"], 4, "EXP-000-completo.json")
        confere(f"EXP-000b {k} Brier", m["brier"], 5, "EXP-000-completo.json")
        confere(f"EXP-000b {k} ECE", m["ece"], 4, "EXP-000-completo.json")
    confere("EXP-000b n teste", e0["n_teste"], 0, "EXP-000-completo.json")

# ------------------------------------------------- EXP-004b (central) -------
e4 = carrega("EXP-004-completo.json")
if e4:
    import statistics as st
    for k in ("DS", "TF"):
        bs = [m["brier"] for m in e4["por_seed"][k]]
        au = [m["auc"] for m in e4["por_seed"][k]]
        confere(f"EXP-004b {k} Brier medio", st.mean(bs), 5, "EXP-004-completo.json")
        confere(f"EXP-004b {k} AUC medio", st.mean(au), 4, "EXP-004-completo.json")
    tb = e4["teste_pareado"]["brier_TF_menos_DS"]
    ta = e4["teste_pareado"]["auc_TF_menos_DS"]
    confere("EXP-004b dif Brier", tb["diferenca"], 5, "EXP-004-completo.json")
    confere("EXP-004b IC inf Brier", tb["ic95"][0], 5, "EXP-004-completo.json")
    confere("EXP-004b IC sup Brier", tb["ic95"][1], 5, "EXP-004-completo.json")
    confere("EXP-004b dif AUC", ta["diferenca"], 5, "EXP-004-completo.json")
    confere("EXP-004b p AUC", ta["p_bilateral"], 3, "EXP-004-completo.json")
    confere("EXP-004b n partidas", e4["n_partidas_teste"], 0, "EXP-004-completo.json")

# --------------------------------------------------- EXP-005 (atencao) ------
e5 = carrega("EXP-005-completo.json")
if e5:
    for nome, g in e5["grupos"].items():
        confere(f"EXP-005 {nome} razao", g["razao_media"], 2, "EXP-005-completo.json")
        confere(f"EXP-005 {nome} mediana", g["razao_mediana"], 2, "EXP-005-completo.json")

# ------------------------------------------------ EXP-006 (calibracao) ------
e6 = carrega("EXP-006-completo.json")
if e6:
    for modelo, variantes in e6["calibracao"].items():
        for var, m in variantes.items():
            confere(f"EXP-006 {modelo}/{var} Brier", m["brier"], 5, "EXP-006-completo.json")
            confere(f"EXP-006 {modelo}/{var} ECE", m["ece"], 4, "EXP-006-completo.json")
    for modelo, a in e6["agregado"].items():
        # o texto cita o xG absoluto do TF e apenas o vies relativo do DS
        confere(f"EXP-006 {modelo} xG total", a["xg_total"], 1,
                "EXP-006-completo.json", obrigatorio=(modelo == "TF"))
        confere(f"EXP-006 {modelo} gols", a["gols_total"], 0, "EXP-006-completo.json")
        confere(f"EXP-006 {modelo} vies %", abs(a["vies_relativo"]) * 100, 1,
                "EXP-006-completo.json")

# ------------------------------------------------- EXP-007 (estudo caso) ----
e7 = carrega("EXP-007-completo.json")
if e7:
    confere("EXP-007 n chutes", e7["n_chutes"], 0, "EXP-007-completo.json")
    confere("EXP-007 gols", e7["gols"], 0, "EXP-007-completo.json")
    for c in e7["casos"]:
        confere(f"EXP-007 xG '{c['titulo'][:22]}'", c["xg"], 3, "EXP-007-completo.json")

# ------------------------------------------------------------- resultado ----
print(f"conferidos com sucesso: {ok}")
for a in ausentes:
    print("  -", a)
print(f"divergencias: {len(falhas)}")
for f in falhas:
    print("  !", f)
sys.exit(1 if falhas else 0)

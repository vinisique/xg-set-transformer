# -*- coding: utf-8 -*-
"""
Testa EMPIRICAMENTE a invariancia a permutacao — a propriedade que o relatorio
usa para justificar a ausencia de positional encoding.

Ate agora essa afirmacao era teorica: "a cena e um conjunto, logo embaralhar os
jogadores nao pode mudar a previsao". Um avaliador pode perguntar se isso vale
DE FATO na implementacao — mascara mal aplicada, ou qualquer dependencia de
indice, quebraria a propriedade sem aviso.

O teste embaralha os jogadores de cada cena e compara as previsoes. Tambem
verifica o Deep Sets, que deve ser invariante pelo mesmo motivo.

Uso:  python teste_invariancia.py
"""
import os

import numpy as np
import torch

import xg_base as xb

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
N = 2000          # cenas de teste amostradas
PERMS = 5         # embaralhamentos por cena

D = xb.carrega(os.path.join(AQUI, "shots_all.npz"))
ite = np.flatnonzero(D["te"])[:N]
tok = D["tok"][ite].copy()
pad = D["pad"][ite].copy()

MODELOS = [
    ("Transformer", xb.Former, os.path.join(RAIZ, "experiments", "EXP-005",
                                            "former_seed0.pt")),
    ("Deep Sets", xb.DeepSets, os.path.join(RAIZ, "experiments", "EXP-006",
                                            "deepsets_seed0.pt")),
]

rng = np.random.default_rng(0)
print(f"{N} cenas · {PERMS} permutações cada\n")

for nome, classe, caminho in MODELOS:
    if not os.path.exists(caminho):
        print(f"{nome}: pesos não encontrados em {caminho} — pulando")
        continue
    modelo = classe()
    modelo.load_state_dict(torch.load(caminho, weights_only=True))
    modelo.eval()

    with torch.no_grad():
        base = torch.sigmoid(modelo(torch.tensor(tok), torch.tensor(pad))).numpy()

    maior = 0.0
    for _ in range(PERMS):
        t2, p2 = tok.copy(), pad.copy()
        for i in range(len(t2)):
            # embaralha APENAS os jogadores do freeze-frame (indices 1..21);
            # o token 0 e o chutador e tem papel proprio
            ordem = rng.permutation(np.arange(1, t2.shape[1]))
            t2[i, 1:] = t2[i, ordem]
            p2[i, 1:] = p2[i, ordem]
        with torch.no_grad():
            novo = torch.sigmoid(modelo(torch.tensor(t2), torch.tensor(p2))).numpy()
        maior = max(maior, np.abs(novo - base).max())

    veredito = "INVARIANTE" if maior < 1e-5 else "NAO invariante"
    print(f"{nome:14s} maior diferença absoluta: {maior:.2e}   -> {veredito}")

print("\n(diferenças da ordem de 1e-7 são ruído de ponto flutuante, não violação)")

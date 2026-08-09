# -*- coding: utf-8 -*-
"""
Barra de progresso do EXP-004, deduzida do estado em disco.

NAO toca no processo em andamento. Le tres fontes observaveis:
  1. o cache de previsoes  -> quantas sementes ja terminaram
  2. a data de modificacao -> quando a ultima terminou
  3. o log de saida        -> quanto cada uma levou

A semente em andamento nao tem como ser medida por dentro (o treino so
imprime quando termina), entao seu progresso e estimado pelo tempo decorrido
contra a media historica. E estimativa, e a saida diz isso.

Uso:  python progresso.py           (uma leitura)
      python progresso.py --loop    (atualiza a cada 30s)
"""
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(AQUI, "..", "experiments", "EXP-004", "predicoes.npz")
BASE_LOG = (r"C:\Users\vinni\AppData\Local\Temp\claude"
            r"\C--Users-vinni-Downloads-projeto"
            r"\88c9b5df-c5dd-4d4a-ace8-ab7185fac0f6\scratchpad")
LOG = os.path.join(BASE_LOG, "exp004c.txt")           # execucao atual
LOGS_ANTIGOS = [os.path.join(BASE_LOG, n)             # tempos ja medidos
                for n in ("exp004.txt", "exp004b.txt")]

SEEDS = 5
TOTAL = 2 * SEEDS          # 5 Deep Sets + 5 Transformer
# medias historicas observadas nas execucoes anteriores (segundos)
ESTIMATIVA = {"DS": 350, "TF": 1250}


def barra(frac, largura=44):
    frac = max(0.0, min(1.0, frac))
    cheio = int(round(frac * largura))
    return "█" * cheio + "░" * (largura - cheio)


def hms(s):
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def estado():
    feitas, quando = [], None
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        feitas = sorted(k for k in d.files if "seed" in k)
        quando = os.path.getmtime(CACHE)

    tempos = {}
    for caminho in LOGS_ANTIGOS + [LOG]:              # o atual sobrescreve os antigos
        if not os.path.exists(caminho):
            continue
        texto = open(caminho, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"(DS|TF) seed (\d).*?\((\d+)s\)", texto):
            tempos[f"{m.group(1)}_seed{m.group(2)}"] = int(m.group(3))

    # A semente em andamento comecou no MAIS RECENTE entre: a ultima gravacao do
    # cache e o inicio desta execucao. Sem isso, uma execucao que aproveitou o
    # cache de uma anterior contaria o tempo desde a execucao antiga.
    inicio_execucao = os.path.getmtime(LOG) if os.path.exists(LOG) else 0
    if quando is None or inicio_execucao > quando:
        quando = inicio_execucao
    return feitas, quando, tempos


def render():
    feitas, quando, tempos = estado()
    n = len(feitas)
    agora = time.time()

    print("\n" + "=" * 62)
    print("  EXP-004 · Transformer vs Deep Sets · 5 sementes cada")
    print("=" * 62)

    for fam in ("DS", "TF"):
        rotulo = "Deep Sets  " if fam == "DS" else "Transformer"
        linha = []
        for s in range(SEEDS):
            chave = f"{fam}_seed{s}"
            if chave in feitas:
                t = tempos.get(chave)
                linha.append(f"[{hms(t)}]" if t else "[ok]")
            else:
                linha.append("[  ·  ]")
        print(f"  {rotulo}  " + " ".join(linha))

    # progresso: sementes concluidas + fracao estimada da que esta rodando
    if n < TOTAL and quando:
        fam_atual = "DS" if n < SEEDS else "TF"
        decorrido = agora - quando
        parcial = min(0.95, decorrido / ESTIMATIVA[fam_atual])
        restantes = TOTAL - n - 1
        falta = (ESTIMATIVA[fam_atual] * (1 - parcial)
                 + restantes * ESTIMATIVA["TF" if n >= SEEDS - 1 else "DS"] + 180)
    else:
        fam_atual, decorrido, parcial, falta = None, 0, 0, 0

    frac = (n + parcial) / TOTAL
    print()
    print(f"  {barra(frac)}  {frac*100:5.1f}%")
    print(f"  {n} de {TOTAL} sementes concluídas", end="")
    if fam_atual:
        print(f"  ·  rodando {fam_atual} semente {n % SEEDS} há {hms(decorrido)}")
        fim = datetime.fromtimestamp(agora + falta).strftime("%H:%M")
        print(f"  restam ~{hms(falta)}  ·  previsão de término ~{fim}  (estimativa)")
    else:
        print("  ·  treinos concluídos (bootstrap e figuras em seguida)")
    print("=" * 62)


if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            render()
            time.sleep(30)
    else:
        render()

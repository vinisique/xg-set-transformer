# -*- coding: utf-8 -*-
"""
PoC 3 — Baixa finalizacoes (com freeze-frame) da StatsBomb Open Data.
TODAS as competicoes/temporadas do repositorio.
Saida: poc/shots_all.npz
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
S = requests.Session()

comps = S.get(f"{BASE}/competitions.json", timeout=30).json()
sel = [(c["competition_id"], c["season_id"], c["competition_name"], c["season_name"])
       for c in comps]
print(f"{len(sel)} pares competicao/temporada")

matches, seen = [], set()
for cid, sid, cn, sn in sel:
    try:
        ms = S.get(f"{BASE}/matches/{cid}/{sid}.json", timeout=30).json()
    except Exception as e:
        print(f"  sem matches p/ {cn} {sn}: {e}")
        continue
    for m in ms:
        if m["match_id"] not in seen:
            seen.add(m["match_id"])
            matches.append((m["match_id"], f"{cn} {sn}"))
print(f"{len(matches)} partidas")

MAXP = 21  # jogadores no freeze-frame alem do chutador

def fetch(args):
    mid, comp = args
    try:
        evs = S.get(f"{BASE}/events/{mid}.json", timeout=60).json()
    except Exception as e:
        print(f"  falha {mid}: {e}")
        return []
    out = []
    for e in evs:
        if e.get("type", {}).get("name") != "Shot":
            continue
        sh = e["shot"]
        if sh.get("type", {}).get("name") == "Penalty":
            continue
        ff = sh.get("freeze_frame")
        if not ff:
            continue
        loc = e.get("location")
        if not loc:
            continue
        px = np.zeros(MAXP, np.float32); py = np.zeros(MAXP, np.float32)
        mate = np.zeros(MAXP, np.float32); gk = np.zeros(MAXP, np.float32)
        mask = np.zeros(MAXP, np.float32)
        for j, p in enumerate(ff[:MAXP]):
            px[j], py[j] = p["location"][0], p["location"][1]
            mate[j] = 1.0 if p["teammate"] else 0.0
            gk[j] = 1.0 if p.get("position", {}).get("name") == "Goalkeeper" else 0.0
            mask[j] = 1.0
        out.append(dict(
            sx=loc[0], sy=loc[1],
            goal=1 if sh.get("outcome", {}).get("name") == "Goal" else 0,
            px=px, py=py, mate=mate, gk=gk, mask=mask,
            match_id=mid, comp=comp,
            header=1 if sh.get("body_part", {}).get("name") == "Head" else 0,
        ))
    return out

shots = []
with ThreadPoolExecutor(16) as ex:
    for i, res in enumerate(ex.map(fetch, matches)):
        shots += res
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(matches)} partidas, {len(shots)} chutes")

print(f"Total: {len(shots)} finalizacoes | gols: {sum(s['goal'] for s in shots)}")
np.savez_compressed(
    "shots_all.npz",
    sx=np.array([s["sx"] for s in shots], np.float32),
    sy=np.array([s["sy"] for s in shots], np.float32),
    goal=np.array([s["goal"] for s in shots], np.int64),
    px=np.stack([s["px"] for s in shots]),
    py=np.stack([s["py"] for s in shots]),
    mate=np.stack([s["mate"] for s in shots]),
    gk=np.stack([s["gk"] for s in shots]),
    mask=np.stack([s["mask"] for s in shots]),
    header=np.array([s["header"] for s in shots], np.float32),
    match_id=np.array([s["match_id"] for s in shots]),
    comp=np.array([s["comp"] for s in shots]),
)
print("shots_all.npz salvo")

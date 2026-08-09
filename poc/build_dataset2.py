# -*- coding: utf-8 -*-
"""
PoC 2 — Dataset estendido com sinais de interacao "reais":
  - coords: posicao X,Y na formacao (quem joga perto de quem)
  - coplay: media de jogos passados de cada jogador com os 10 companheiros atuais
  - form:   pontos do time nos ultimos 5 jogos
  - pidx:   indice do jogador (para embedding residual por ID)
Saida: poc/dataset2.npz
"""
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

DB = "data/database.sqlite"
conn = sqlite3.connect(DB)

player_cols = [f"home_player_{i}" for i in range(1, 12)] + [f"away_player_{i}" for i in range(1, 12)]
xy_cols = []
for side in ("home", "away"):
    for i in range(1, 12):
        xy_cols += [f"{side}_player_X{i}", f"{side}_player_Y{i}"]
odds_cols = ["B365H", "B365D", "B365A"]

m = pd.read_sql(
    f"""SELECT id, season, date, home_team_api_id, away_team_api_id,
               home_team_goal, away_team_goal,
               {','.join(player_cols)}, {','.join(xy_cols)}, {','.join(odds_cols)}
        FROM Match""",
    conn,
)
m["date"] = pd.to_datetime(m["date"])
m = m.dropna(subset=player_cols).sort_values("date").reset_index(drop=True)
print(f"Partidas: {len(m)}")

# ---------- atributos datados (igual PoC 1) ----------
pa = pd.read_sql("SELECT * FROM Player_Attributes", conn)
pa["date"] = pd.to_datetime(pa["date"])
pa = pa.drop(columns=["id", "player_fifa_api_id", "preferred_foot",
                      "attacking_work_rate", "defensive_work_rate"])
feat_cols = [c for c in pa.columns if c not in ("player_api_id", "date")]

pl = pd.read_sql("SELECT player_api_id, birthday, height, weight FROM Player", conn)
pl["birthday"] = pd.to_datetime(pl["birthday"])

rows = []
for side, t in (("home", 0), ("away", 1)):
    for i in range(1, 12):
        sub = m[["id", "date", f"{side}_player_{i}",
                 f"{side}_player_X{i}", f"{side}_player_Y{i}"]].copy()
        sub.columns = ["match_id", "match_date", "player_api_id", "posX", "posY"]
        sub["team"], sub["slot"] = t, i - 1
        rows.append(sub)
long = pd.concat(rows, ignore_index=True)
long["player_api_id"] = long["player_api_id"].astype(int)

pa = pa.sort_values("date")
long = long.sort_values("match_date")
merged = pd.merge_asof(long, pa, left_on="match_date", right_on="date",
                       by="player_api_id", direction="backward")
miss = merged["overall_rating"].isna()
if miss.any():
    fb = pd.merge_asof(long[miss.values].sort_values("match_date"), pa,
                       left_on="match_date", right_on="date",
                       by="player_api_id", direction="forward")
    merged.loc[miss, feat_cols] = fb[feat_cols].values

merged = merged.merge(pl, on="player_api_id", how="left")
merged["age"] = (merged["match_date"] - merged["birthday"]).dt.days / 365.25
all_feats = feat_cols + ["height", "weight", "age"]
merged[all_feats] = merged[all_feats].fillna(merged[all_feats].median())
merged["posX"] = merged["posX"].fillna(5)
merged["posY"] = merged["posY"].fillna(5)
merged["role"] = np.select([merged.posY <= 1, merged.posY <= 4, merged.posY <= 7],
                           [0, 1, 2], default=3)

# vocabulario de jogadores -> indice
uniq = np.sort(merged["player_api_id"].unique())
pid2idx = {p: i for i, p in enumerate(uniq)}
merged["pidx"] = merged["player_api_id"].map(pid2idx)
print(f"Jogadores unicos: {len(uniq)}")

# ---------- tensores base (ordem cronologica de partidas) ----------
merged = merged.sort_values(["match_date", "match_id", "team", "slot"])
order = merged["match_id"].drop_duplicates().to_numpy()
n, F = len(order), len(all_feats)
X = merged[all_feats].to_numpy(np.float32).reshape(n, 22, F)
coords = merged[["posX", "posY"]].to_numpy(np.float32).reshape(n, 22, 2)
role = merged["role"].to_numpy(np.int64).reshape(n, 22)
team = merged["team"].to_numpy(np.int64).reshape(n, 22)
pidx = merged["pidx"].to_numpy(np.int64).reshape(n, 22)

mm = m.set_index("id").loc[order]
y = np.select([mm.home_team_goal > mm.away_team_goal,
               mm.home_team_goal == mm.away_team_goal], [0, 1], default=2).astype(np.int64)
odds = mm[odds_cols].to_numpy(np.float32)
season = mm["season"].to_numpy()

# ---------- coplay e forma recente, varrendo cronologicamente ----------
pair_count = defaultdict(int)   # (pid_a, pid_b) -> jogos juntos ate agora
team_pts = defaultdict(list)    # team_api_id -> pontos por jogo (cronologico)
pids_all = merged["player_api_id"].to_numpy().reshape(n, 22)
home_ids = mm["home_team_api_id"].to_numpy()
away_ids = mm["away_team_api_id"].to_numpy()
hg = mm["home_team_goal"].to_numpy()
ag = mm["away_team_goal"].to_numpy()

coplay = np.zeros((n, 22), np.float32)
form = np.zeros((n, 22), np.float32)
for k in range(n):
    for t0, tid in ((0, home_ids[k]), (11, away_ids[k])):
        squad = pids_all[k, t0:t0 + 11]
        for a in range(11):
            s = 0
            for b in range(11):
                if a != b:
                    key = (squad[a], squad[b]) if squad[a] < squad[b] else (squad[b], squad[a])
                    s += pair_count[key]
            coplay[k, t0 + a] = s / 10.0
        form[k, t0:t0 + 11] = sum(team_pts[tid][-5:])
    # atualiza contadores DEPOIS de gravar as features (sem vazamento)
    for t0 in (0, 11):
        squad = pids_all[k, t0:t0 + 11]
        for a in range(11):
            for b in range(a + 1, 11):
                key = (squad[a], squad[b]) if squad[a] < squad[b] else (squad[b], squad[a])
                pair_count[key] += 1
    ph = 3 if hg[k] > ag[k] else (1 if hg[k] == ag[k] else 0)
    team_pts[home_ids[k]].append(ph)
    team_pts[away_ids[k]].append(3 - ph if ph != 1 else 1)

np.savez_compressed(
    "dataset2.npz", X=X, coords=coords, coplay=coplay, form=form, pidx=pidx,
    role=role, team=team, y=y, season=season, odds=odds,
    feat_names=np.array(all_feats), n_players=len(uniq),
)
print(f"dataset2.npz salvo: X{X.shape} | coplay media={coplay.mean():.2f} | form media={form.mean():.2f}")

# -*- coding: utf-8 -*-
"""
PoC — Constroi o dataset jogador-token a partir do European Soccer Database.

Saida: poc/dataset.npz com
  X        [N, 22, F]  atributos FIFA de cada jogador na data (mais recente <= data do jogo)
  role     [N, 22]     0=GK 1=DEF 2=MID 3=ATT (derivado da coordenada Y da formacao)
  team     [N, 22]     0=mandante, 1=visitante
  y        [N]         0=vitoria mandante, 1=empate, 2=vitoria visitante
  season   [N]         string da temporada
  odds     [N, 3]      B365 H/D/A (NaN quando ausente)
"""
import sqlite3
import numpy as np
import pandas as pd

DB = "data/database.sqlite"

conn = sqlite3.connect(DB)

# ---------- Partidas com escalacao completa ----------
player_cols = [f"home_player_{i}" for i in range(1, 12)] + [f"away_player_{i}" for i in range(1, 12)]
y_cols = [f"home_player_Y{i}" for i in range(1, 12)] + [f"away_player_Y{i}" for i in range(1, 12)]
odds_cols = ["B365H", "B365D", "B365A"]

m = pd.read_sql(
    f"""SELECT id, season, date, home_team_goal, away_team_goal,
               {','.join(player_cols)}, {','.join(y_cols)}, {','.join(odds_cols)}
        FROM Match""",
    conn,
)
m["date"] = pd.to_datetime(m["date"])
m = m.dropna(subset=player_cols).reset_index(drop=True)
print(f"Partidas com escalacao completa: {len(m)}")

# ---------- Atributos dos jogadores (datados) ----------
pa = pd.read_sql("SELECT * FROM Player_Attributes", conn)
pa["date"] = pd.to_datetime(pa["date"])
drop = ["id", "player_fifa_api_id", "preferred_foot", "attacking_work_rate", "defensive_work_rate"]
pa = pa.drop(columns=drop)
feat_cols = [c for c in pa.columns if c not in ("player_api_id", "date")]
print(f"{len(feat_cols)} atributos por jogador: {feat_cols[:6]}...")

pl = pd.read_sql("SELECT player_api_id, birthday, height, weight FROM Player", conn)
pl["birthday"] = pd.to_datetime(pl["birthday"])

# ---------- Formato longo: uma linha por (partida, slot) ----------
rows = []
for side, prefix in (("home", "home_player_"), ("away", "away_player_")):
    for i in range(1, 12):
        sub = m[["id", "date", prefix + str(i), f"{side}_player_Y{i}"]].copy()
        sub.columns = ["match_id", "match_date", "player_api_id", "posY"]
        sub["team"] = 0 if side == "home" else 1
        sub["slot"] = i - 1
        rows.append(sub)
long = pd.concat(rows, ignore_index=True)
long["player_api_id"] = long["player_api_id"].astype(int)

# merge_asof: atributo mais recente ANTES da partida (sem vazamento de futuro)
pa = pa.sort_values("date")
long = long.sort_values("match_date")
merged = pd.merge_asof(
    long, pa, left_on="match_date", right_on="date",
    by="player_api_id", direction="backward",
)
# fallback p/ jogadores sem atributo anterior: usa o primeiro disponivel
missing = merged["overall_rating"].isna()
if missing.any():
    fb = pd.merge_asof(
        long[missing.values].sort_values("match_date"), pa,
        left_on="match_date", right_on="date",
        by="player_api_id", direction="forward",
    )
    merged.loc[missing, feat_cols] = fb[feat_cols].values
print(f"Slots sem atributos apos fallback: {merged['overall_rating'].isna().sum()} / {len(merged)}")

merged = merged.merge(pl, on="player_api_id", how="left")
merged["age"] = (merged["match_date"] - merged["birthday"]).dt.days / 365.25

all_feats = feat_cols + ["height", "weight", "age"]
merged[all_feats] = merged[all_feats].fillna(merged[all_feats].median())

# papel a partir da coordenada Y da formacao
posY = merged["posY"].fillna(5)
merged["role"] = np.select([posY <= 1, posY <= 4, posY <= 7], [0, 1, 2], default=3)

# ---------- Monta tensores ----------
merged = merged.sort_values(["match_id", "team", "slot"])
n = merged["match_id"].nunique()
F = len(all_feats)
X = merged[all_feats].to_numpy(np.float32).reshape(n, 22, F)
role = merged["role"].to_numpy(np.int64).reshape(n, 22)
team = merged["team"].to_numpy(np.int64).reshape(n, 22)

order = merged["match_id"].drop_duplicates().to_numpy()
mm = m.set_index("id").loc[order]
y = np.select(
    [mm.home_team_goal > mm.away_team_goal, mm.home_team_goal == mm.away_team_goal],
    [0, 1], default=2,
).astype(np.int64)
odds = mm[odds_cols].to_numpy(np.float32)
season = mm["season"].to_numpy()

np.savez_compressed(
    "dataset.npz", X=X, role=role, team=team, y=y, season=season, odds=odds,
    feat_names=np.array(all_feats),
)
print(f"dataset.npz salvo: X{X.shape}, classes={np.bincount(y)}, temporadas={sorted(set(season))}")

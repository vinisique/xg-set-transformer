# -*- coding: utf-8 -*-
"""
Base compartilhada dos experimentos de xG: dados, tokens, modelos, treino e
metricas. Existe para que cada experimento novo nao seja mais uma copia da
preparacao dos dados — copia e onde nascem as divergencias silenciosas entre
experimentos que deveriam ser comparaveis.

poc3_xg3.py e exp000_evidencia.py sao anteriores a este modulo e ficam como
estao: sao o registro do que ja foi executado.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

# =========================================================== metricas ======

def brier(y, p):
    """Erro quadratico medio da probabilidade. Enxerga calibracao; a AUC nao."""
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    """Expected Calibration Error em bins de QUANTIL — bins de largura fixa
    deixariam os ultimos quase vazios, ja que a maioria dos chutes tem xG baixo."""
    cortes = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    idx = np.digitize(p, cortes[1:-1])
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def curva_calibracao(y, p, n_bins=10):
    cortes = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    idx = np.digitize(p, cortes[1:-1])
    prev, obs, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            prev.append(p[m].mean()); obs.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(prev), np.array(obs), np.array(ns)


def metricas(y, p):
    return {"auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "brier": brier(y, p),
            "ece": ece(y, p)}


def bootstrap_pareado(y, p_a, p_b, grupos, fn=brier, n=2000, seed=0):
    """Bootstrap pareado AGRUPADO POR PARTIDA.

    Reamostra PARTIDAS com reposicao, nao chutes. Chutes da mesma partida sao
    correlacionados (mesmo time, mesmo adversario, mesmo contexto); reamostrar
    chutes individualmente trataria essa correlacao como informacao independente
    e produziria um intervalo de confianca estreito demais — exatamente o erro
    que o split por partida existe para evitar.

    Devolve a diferenca observada fn(A) - fn(B), o IC 95% e a fracao de
    reamostras em que a diferenca troca de sinal.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(grupos)
    uniq = np.unique(g)
    por_grupo = {u: np.flatnonzero(g == u) for u in uniq}

    obs = fn(y, p_a) - fn(y, p_b)
    difs = np.empty(n)
    for i in range(n):
        escolhidas = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([por_grupo[u] for u in escolhidas])
        difs[i] = fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx])

    lo, hi = np.percentile(difs, [2.5, 97.5])
    # p bilateral: com que frequencia a diferenca aparece do lado oposto ao observado
    p_val = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    return {"diferenca": float(obs), "ic95": [float(lo), float(hi)],
            "p_bilateral": float(min(1.0, p_val)),
            "distribuicao": difs}


# =============================================================== dados =====

def carrega(caminho="shots_all.npz", seed_split=0):
    """Carrega os chutes, monta os tokens e faz o split POR PARTIDA (70/15/15)."""
    d = np.load(caminho, allow_pickle=True)
    sx, sy, goal = d["sx"], d["sy"], d["goal"]
    px, py = d["px"], d["py"]
    mate, gk, mask, header = d["mate"], d["gk"], d["mask"], d["header"]
    match_id, comp = d["match_id"], d["comp"]
    N, P = px.shape

    dx = 120.0 - sx
    dist = np.hypot(dx, sy - 40.0)
    angle = np.abs(np.arctan2(8.0 * dx, dx**2 + (sy - 40.0)**2 - 16.0))

    rng = np.random.default_rng(seed_split)
    uniq = np.unique(match_id); rng.shuffle(uniq)
    n1, n2 = int(0.7 * len(uniq)), int(0.85 * len(uniq))
    tr = np.isin(match_id, uniq[:n1])
    va = np.isin(match_id, uniq[n1:n2])
    te = np.isin(match_id, uniq[n2:])

    opp = (mate < 0.5) & (mask > 0.5)

    def dentro(pxx, pyy):
        d1 = (36 - sy[:, None]) * (pxx - sx[:, None]) - (120 - sx[:, None]) * (pyy - sy[:, None])
        d2 = (44 - sy[:, None]) * (pxx - sx[:, None]) - (120 - sx[:, None]) * (pyy - sy[:, None])
        return (d1 * d2 < 0) & (pxx > sx[:, None])

    # --- features manuais (B2) ---
    gkm = opp & (gk > 0.5)
    gk_dist = np.nanmin(np.where(gkm, np.hypot(120 - px, py - 40), np.inf), 1)
    gk_dist[np.isinf(gk_dist)] = 25.0
    gkx = np.where(gkm, px, 0).sum(1) / np.maximum(gkm.sum(1), 1)
    gky = np.where(gkm, py, 0).sum(1) / np.maximum(gkm.sum(1), 1)
    t = np.clip(((gkx - sx) * dx + (gky - sy) * (40 - sy)) / (dist**2 + 1e-8), 0, 1)
    gk_off = np.hypot(sx + t * dx - gkx, sy + t * (40 - sy) - gky)
    blockers = (dentro(px, py) & opp & (gk < 0.5)).sum(1).astype(np.float32)
    near_opp = np.min(np.where(opp, np.hypot(px - sx[:, None], py - sy[:, None]), np.inf), 1)
    near_opp[np.isinf(near_opp)] = 30.0

    # --- tokens (DS/TF): geometria relativa ao chutador e a linha do chute ---
    pdist = np.hypot(120 - px, py - 40)
    dshoot = np.hypot(px - sx[:, None], py - sy[:, None])
    tline = np.clip(((px - sx[:, None]) * dx[:, None] + (py - sy[:, None]) * (40 - sy)[:, None])
                    / (dist[:, None] ** 2 + 1e-8), 0, 1)
    perp = np.hypot(sx[:, None] + tline * dx[:, None] - px,
                    sy[:, None] + tline * (40 - sy)[:, None] - py)

    Z, ZP = np.zeros(N), np.zeros((N, P))
    shooter = np.stack([sx / 120, sy / 80, dist / 50, angle, Z, Z, Z, Z, Z, Z,
                        np.ones(N), np.zeros(N), np.ones(N), header], 1)[:, None, :]
    others = np.stack([px / 120, py / 80, pdist / 50, ZP,
                       (px - sx[:, None]) / 40.0, (py - sy[:, None]) / 40.0,
                       dshoot / 40.0, tline, perp / 20.0,
                       dentro(px, py).astype(np.float32), mate, gk, ZP, ZP], -1)

    return {
        "goal": goal, "match_id": match_id, "comp": comp,
        "tr": tr, "va": va, "te": te,
        "F1": np.stack([dist, angle, header], 1),
        "F2": np.stack([dist, angle, header, gk_dist, gk_off, blockers, near_opp], 1),
        "tok": np.concatenate([shooter, others], 1).astype(np.float32),
        "pad": np.concatenate([np.zeros((N, 1)), 1 - mask], 1) > 0.5,
    }


def logisticas(D):
    """B1 e B2 — previsoes de teste."""
    out = {}
    for nome, F, it in (("B1", D["F1"], 1000), ("B2", D["F2"], 2000)):
        lr = LogisticRegression(max_iter=it).fit(F[D["tr"]], D["goal"][D["tr"]])
        out[nome] = lr.predict_proba(F[D["te"]])[:, 1]
    return out


# ============================================================== modelos ====

class DeepSets(nn.Module):
    """Processa cada jogador ISOLADAMENTE e agrega (media + maximo).
    Nunca compara dois jogadores entre si — por isso isola o valor da atencao."""

    def __init__(self, dim=48, drop=0.1, n_feat=14):
        super().__init__()
        self.proj = nn.Linear(n_feat, dim)
        self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(dim * 2, dim), nn.ReLU())
        self.head = nn.Linear(dim * 2, 1)

    def forward(self, x, pad):
        h = self.phi(self.proj(x))
        w = (~pad).float().unsqueeze(-1)
        mean = (h * w).sum(1) / w.sum(1)
        mx = h.masked_fill(pad.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


class Former(nn.Module):
    """Set Transformer: [CLS] + encoder SEM positional encoding (a cena e um
    conjunto, nao uma sequencia) + mascara de padding nas posicoes vazias."""

    def __init__(self, dim=48, drop=0.1, n_feat=14, camadas=2, cabecas=4):
        super().__init__()
        self.proj = nn.Linear(n_feat, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, cabecas, dim * 2, drop,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, camadas)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, pad):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.proj(x)], 1)
        p = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool), pad], 1)
        return self.head(self.tf(h, src_key_padding_mask=p)[:, 0]).squeeze(-1)


@torch.no_grad()
def atencao_do_cls(model, x, pad):
    """Devolve (logit, pesos_de_atencao) refazendo o forward do Former a mao.

    `nn.TransformerEncoder` nao expoe os pesos de atencao, entao replicamos o
    bloco pre-LN (norm_first=True) chamando `self_attn` com need_weights=True.

    Saidas:
      logit : [B]
      attn  : [B, camadas, cabecas, 23] — quanto o [CLS] olhou para cada token
              (indice 0 = o proprio CLS, 1 = chutador, 2..22 = freeze-frame)

    A funcao AFIRMA que o logit reproduzido bate com `model(x, pad)`. Se a
    replicacao divergir do forward real, a analise estaria descrevendo um modelo
    que nao e o avaliado — por isso a verificacao e obrigatoria, nao opcional.
    """
    model.eval()
    h = torch.cat([model.cls.expand(len(x), -1, -1), model.proj(x)], 1)
    p = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool), pad], 1)

    pesos = []
    for camada in model.tf.layers:
        normed = camada.norm1(h)
        saida, w = camada.self_attn(normed, normed, normed,
                                    key_padding_mask=p,
                                    need_weights=True, average_attn_weights=False)
        pesos.append(w[:, :, 0, :])          # linha do [CLS]: [B, cabecas, 23]
        h = h + saida                        # dropout e identidade em eval
        h = h + camada._ff_block(camada.norm2(h))

    logit = model.head(h[:, 0]).squeeze(-1)
    esperado = model(x, pad)
    erro = (logit - esperado).abs().max().item()
    assert erro < 1e-4, f"replicacao do forward divergiu do modelo (erro {erro:.2e})"
    return logit, torch.stack(pesos, 1)


def treina(model, D, seed, criterio="brier", epocas=60, paciencia=8, batch=256,
           tol_rel=1e-4):
    """Treina e devolve as previsoes de TESTE do melhor estado de validacao.

    criterio: 'brier' (cartao 0001) ou 'auc' (protocolo antigo do EXP-000).

    tol_rel: melhora minima para reiniciar a paciencia, RELATIVA a escala da
    metrica (`tol_rel * |melhor|`). Tolerancia absoluta nao serve aqui porque as
    duas metricas vivem em escalas diferentes: AUC perto de 0,81 e Brier perto
    de 0,076. No EXP-004 isso custou caro — um limiar absoluto de 1e-6 sobre o
    Brier era, em termos relativos, dez vezes mais apertado que o 1e-4 usado
    sobre a AUC. Quase toda flutuacao contava como melhora, a paciencia nunca
    disparava, o treino ia ate a ultima epoca e selecionava um estado ja
    sobreajustado — produzindo Brier PIOR justamente no criterio que deveria
    otimiza-lo.
    """
    X = torch.tensor(D["tok"]); PAD = torch.tensor(D["pad"])
    Y = torch.tensor(D["goal"], dtype=torch.float32)
    itr, iva, ite = (np.flatnonzero(D[k]) for k in ("tr", "va", "te"))
    y_va = D["goal"][iva]

    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    lf = nn.BCEWithLogitsLoss()

    melhor, estado, espera = None, None, 0
    for ep in range(epocas):
        model.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(itr)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lf(model(X[b], PAD[b]), Y[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            saida = model(X[iva], PAD[iva]).numpy()
        # Brier: menor e melhor -> usamos o negativo para "maior e melhor"
        score = (-brier(y_va, 1 / (1 + np.exp(-saida))) if criterio == "brier"
                 else roc_auc_score(y_va, saida))
        if melhor is None or score > melhor + tol_rel * abs(melhor):
            melhor, espera = score, 0
            estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break

    model.load_state_dict(estado)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X[ite], PAD[ite])).numpy()


def treina_e_guarda(model, D, seed, caminho, **kw):
    """Como treina(), mas grava os PESOS em disco e os reaproveita se existirem.

    Os experimentos anteriores guardavam so as previsoes. Para inspecionar
    atencao (EXP-005) e preciso o modelo em si — e retreinar 10 minutos a cada
    ajuste de figura seria desperdicio.
    """
    import os
    if os.path.exists(caminho):
        model.load_state_dict(torch.load(caminho, weights_only=True))
        model.eval()
        return model, True
    treina(model, D, seed, **kw)      # deixa o model com o melhor estado carregado
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    torch.save(model.state_dict(), caminho)
    return model, False

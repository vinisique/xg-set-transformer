# -*- coding: utf-8 -*-
"""
Modulo de visualizacao do projeto — estilo unico para todas as figuras.

Toda figura de experimento passa por aqui, para que o relatorio inteiro tenha
uma linguagem visual consistente. A paleta foi validada para daltonismo
(separacao CVD dE >= 8 em OKLab x100) antes de ser adotada.

Uso:
    import viz
    fig, ax = viz.figura(largura=7, altura=4)
    ...
    viz.salvar(fig, "docs/experimentos/figuras/EXP-000-escada.png")
"""
import matplotlib

matplotlib.use("Agg")  # sem janela: roda em background e escreve direto em disco
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paleta ----
# Categorica: ordem fixa, nunca ciclada. Cada modelo mantem sua cor em todas
# as figuras do projeto — cor segue a entidade, nao a posicao no ranking.
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

COR = {
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta_2": "#52514e",
    "suave": "#898781",
    "grade": "#e1e0d9",
    "eixo": "#c3c2b7",
}

# Cor fixa por modelo — usada em TODAS as figuras do projeto.
COR_MODELO = {
    "B1": SERIE[0],
    "B2": SERIE[1],
    "DS": SERIE[2],
    "TF": SERIE[3],
}

NOME_MODELO = {
    "B1": "B1 · regressão logística\n(distância, ângulo, cabeceio)",
    "B2": "B2 · + interação manual\n(goleiro, bloqueadores)",
    "DS": "DS · Deep Sets\n(tokens, sem atenção)",
    "TF": "TF · Transformer\n(tokens, com atenção)",
}


def estilo():
    """Aplica o estilo do projeto ao matplotlib. Chamado por figura()."""
    plt.rcParams.update({
        "figure.facecolor": COR["superficie"],
        "axes.facecolor": COR["superficie"],
        "savefig.facecolor": COR["superficie"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "text.color": COR["tinta"],
        "axes.labelcolor": COR["tinta_2"],
        "axes.edgecolor": COR["eixo"],
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": COR["suave"],
        "ytick.color": COR["suave"],
        "xtick.labelcolor": COR["tinta_2"],
        "ytick.labelcolor": COR["tinta_2"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        # grade: linha fina e solida, um tom acima da superficie. Nunca tracejada.
        "grid.color": COR["grade"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def figura(largura=7.0, altura=4.0, colunas=1):
    """Cria figura+eixos ja no estilo do projeto, sem as bordas superior/direita."""
    estilo()
    fig, axs = plt.subplots(1, colunas, figsize=(largura, altura))
    for ax in ([axs] if colunas == 1 else axs):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig, axs


def rodape(fig, texto):
    """Nota de rodape — de onde vieram os numeros. Rastreabilidade na figura."""
    fig.text(0.0, -0.02, texto, ha="left", va="top",
             fontsize=7.5, color=COR["suave"])


def salvar(fig, caminho):
    import os
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura salva: {caminho}")

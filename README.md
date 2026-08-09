# xG com set Transformer — o jogador como token

Estimativa de **Expected Goals (xG)** a partir da posição real de **todos os
jogadores no instante do chute**, usando um Transformer que trata **cada jogador
como um token**.

Projeto final da disciplina **CCM-109 — Deep Learning (Tópicos Avançados em
Inteligência Artificial)** · Universidade Federal do ABC · Prof. Ronaldo Prati.
Autor: **Vinícius Siqueira**.

---

## A pergunta

O xG é a métrica central da análise de futebol profissional. Os modelos clássicos
usam apenas atributos do chute — distância, ângulo, parte do corpo — e, quando
incorporam os outros jogadores, dependem de *features* desenhadas à mão: distância
do goleiro, número de defensores na linha do chute.

> **Um Transformer que trata cada jogador como um token aprende sozinho a
> geometria de interação que hoje exige engenharia manual de atributos?**

A cena de uma finalização é um **conjunto**, não uma sequência: a ordem em que os
jogadores são listados não carrega informação. Por isso o modelo é um *set
Transformer* — encoder com self-attention, token `[CLS]` e **sem positional
encoding** — o que o torna invariante a permutação dos jogadores.

## Dados

**[StatsBomb Open Data](https://github.com/statsbomb/open-data)** — pública, uso
não comercial com atribuição. Cada finalização traz o *freeze-frame*: coordenadas
(x, y) de todos os jogadores visíveis no momento do chute, com papel
(companheiro / adversário / goleiro) e desfecho.

| | |
|---|---|
| Finalizações | 99.746 (pênaltis excluídos) |
| Partidas | 3.961 |
| Competições-temporada | 80 |
| Taxa de gol | 10,26% |
| Jogadores visíveis por cena | média 13,1 · mínimo 1 · máximo 21 |

Split **por partida** (70/15/15): chutes do mesmo jogo nunca aparecem em splits
diferentes, o que evita vazamento entre lances correlacionados.

## Abordagem — escada de baselines

Cada degrau acrescenta exatamente uma capacidade, de modo que a diferença entre
degraus atribui o mérito a um componente específico:

| | Modelo | O que acrescenta |
|---|---|---|
| **B1** | Regressão logística com distância, ângulo e cabeceio | o xG "de livro" |
| **B2** | B1 + features manuais de interação (goleiro, bloqueadores) | interação **feita à mão** |
| **DS** | Deep Sets sobre os tokens de jogador | informação por jogador, **sem** comparação par a par |
| **TF** | Transformer sobre os mesmos tokens | interação **par a par aprendida** |

A comparação decisiva é **TF vs. DS**: os dois recebem exatamente os mesmos
tokens, e só o Transformer pode comparar jogadores entre si. A diferença isola o
valor da atenção.

## Reproduzindo

Requer Python 3.12+. As dependências são `torch` (CPU basta), `numpy`,
`scikit-learn`, `requests` e `matplotlib`.

```bash
git clone https://github.com/vinisique/xg-set-transformer.git
cd xg-set-transformer

python -m venv poc/.venv
poc/.venv/Scripts/activate          # Windows
# source poc/.venv/bin/activate     # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn requests matplotlib
```

**1. Baixar os dados** (~15 min; escreve `poc/shots_all.npz`, ~6 MB):

```bash
cd poc
python poc3_fetch_all.py
```

**2. Rodar a escada de baselines** (CPU, dezenas de minutos):

```bash
python poc3_xg3.py
```

Os dados brutos e o ambiente virtual **não** são versionados — são reconstruídos
pelos comandos acima. Ver `.gitignore`.

## Organização do repositório

```
docs/                    documentação do projeto — comece por docs/README.md
├── METODOLOGIA.md         como o trabalho é conduzido e registrado
├── ESTADO_ATUAL.md        diagnóstico: o que existe, o que falta, riscos
├── PLANO.md               marcos até a entrega
├── GLOSSARIO.md           todo termo técnico explicado
├── decisoes/              uma decisão técnica por arquivo, com alternativas
└── experimentos/          log append-only de resultados, com IDs

poc/                     prova de conceito
├── poc3_fetch_all.py      download das finalizações com freeze-frame
├── poc3_xg3.py            escada completa B1 / B2 / Deep Sets / Transformer
├── poc3_xg.py, xg2.py     versões anteriores (dataset menor)
└── build_dataset*.py,     linha abandonada: previsão de resultado 1X2
    poc2_*.py, train.py     a partir de escalações (European Soccer Database)
```

## Método

Duas regras organizam o repositório e existem para tornar o trabalho auditável:

1. **Nenhuma decisão técnica sem registro escrito** — cada escolha vive em
   `docs/decisoes/`, com as alternativas que foram descartadas e por quê.
2. **Nenhum número no relatório sem ID de experimento** em
   `docs/experimentos/RESULTS.md`, que é *append-only* — inclusive para os
   experimentos que não funcionaram.

## Licença e atribuição

Dados da **StatsBomb Open Data**, usados sob a licença de uso não comercial com
atribuição à fonte. O material didático da disciplina não é redistribuído aqui.

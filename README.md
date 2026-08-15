# xG com set Transformer: o jogador como token

Estimativa de **Expected Goals (xG)** a partir da posição real de **todos os
jogadores no instante do chute**, usando um Transformer que trata **cada jogador
como um token**.

Projeto final da disciplina **CCM-109, Deep Learning (Tópicos Avançados em
Inteligência Artificial)** · Universidade Federal do ABC · Prof. Ronaldo Prati.
Autor: **Vinícius Siqueira**.

Relatório final: [`relatorio/artigo.pdf`](relatorio/artigo.pdf).

---

## A pergunta

O xG é a métrica central da análise de futebol profissional. Os modelos clássicos
usam apenas atributos do chute (distância, ângulo, parte do corpo) e, quando
incorporam os outros jogadores, dependem de *features* desenhadas à mão: distância
do goleiro, número de defensores na linha do chute.

> **Um Transformer que trata cada jogador como um token aprende sozinho a
> geometria de interação que hoje exige engenharia manual de atributos?**

A cena de uma finalização é um **conjunto**, não uma sequência: a ordem em que os
jogadores são listados não carrega informação. Por isso o modelo é um *set
Transformer*, um encoder com self-attention, token `[CLS]` e **sem positional
encoding**, o que o torna invariante a permutação dos jogadores.

## Dados

**[StatsBomb Open Data](https://github.com/statsbomb/open-data)**, pública, uso
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

Baixados do ramo `master` em 03/07/2026. Esse ramo recebe novas competições ao
longo do tempo, então a data faz parte da especificação do conjunto.

Split **por partida** (70/15/15): chutes do mesmo jogo nunca aparecem em splits
diferentes, o que evita vazamento entre lances correlacionados.

## Abordagem: escada de baselines

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

## Resultados

No conjunto de teste, com 14.935 finalizações e combinação de 5 sementes:

| Modelo | AUC | Brier | ECE |
|---|---|---|---|
| B1 | 0,7653 | 0,08237 | 0,0076 |
| B2 | 0,7961 | 0,07846 | 0,0035 |
| Deep Sets | 0,8160 | 0,07635 | 0,0070 |
| Transformer | **0,8162** | **0,07579** | 0,0043 |

**A atenção par a par melhora a qualidade da probabilidade, mas não a capacidade
de ordenar as finalizações.** Em Brier a diferença é de -0,00056, com intervalo
de confiança [-0,00081; -0,00031] por bootstrap pareado agrupado por partida. Em
AUC ela é indistinguível de ruído (p = 0,797).

O efeito é pequeno e o trabalho declara três limites que impedem tratá-lo como
estabelecido:

* **Depende da combinação de sementes.** Semente a semente, o intervalo cruza
  zero nas cinco inicializações, com menor valor-p de 0,053.
* **Não se sustenta fora da distribuição.** Com a Premier League 2015/16 inteira
  fora do treino, todos os modelos perdem cerca de um quinto da perícia e a
  vantagem do Transformer deixa de ser estabelecida (p = 0,251). A calibração
  agregada, essa se mantém, com viés de -0,6%.
* **Metade do salto atribuído à representação era não-linearidade.** Um gradient
  boosting sobre os mesmos sete atributos manuais do B2 explica 54,5% do ganho em
  AUC e 64,6% em Brier do degrau B2 para Deep Sets.

Sobre o que a atenção observa, o token `[CLS]` concentra 2,76 vezes o acaso sobre
bloqueadores e 1,38 vezes sobre o goleiro. Uma ablação causal confirma
dependência: proibir o modelo de observar os bloqueadores degrada o Brier 7,9
vezes mais do que suprimir a mesma quantidade de jogadores sorteados. Vale a
ressalva registrada no relatório: o indicador "dentro do triângulo do chute" é um
atributo de entrada, então o resultado mostra o modelo **ponderando** essa pista,
e não descobrindo a geometria.

Duas explicações alternativas foram testadas e afastadas: equiparar a capacidade
do Deep Sets (38.977 parâmetros contra 38.737) não reproduz o ganho, e remover a
ReLU terminal do Deep Sets não o aproxima do Transformer.

O registro completo, incluindo os experimentos que não funcionaram, está em
[`docs/experimentos/RESULTS.md`](docs/experimentos/RESULTS.md).

## Reproduzindo

Requer **Python 3.14** (versão usada; 3.11 ou superior deve funcionar) e roda em
CPU, sem GPU.

```bash
git clone https://github.com/vinisique/xg-set-transformer.git
cd xg-set-transformer

python -m venv poc/.venv
poc/.venv/Scripts/activate          # Windows
# source poc/.venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

**1. Baixar os dados** (cerca de 15 min; escreve `poc/shots_all.npz`, 6 MB):

```bash
cd poc
python poc3_fetch_all.py
```

**2. Rodar os experimentos.** Cada script grava um JSON em `docs/experimentos/` e
a figura correspondente. Os números do relatório saem daqui, não do
`poc3_xg3.py`, que é a prova de conceito anterior e ficou como registro histórico.

```bash
python exp004_tf_vs_ds.py            # questão central: TF vs DS, 5 sementes
python exp010_baseline_nao_linear.py # escada completa, com o B2-GBM
python exp005_atencao.py             # atenção por papel na cena
python exp006_calibracao.py          # calibração e recalibração
python exp008_capacidade.py          # controle de capacidade
python exp009_ablacao.py             # ablação causal da atenção
python exp011_robustez.py            # por semente e por subgrupo
python exp012_holdout.py             # hold-out por competição
python exp013_relu_terminal.py       # controle da ReLU terminal
python exp007_estudo_caso.py         # estudo de caso da Eurocopa 2024
```

Os pesos e as previsões são salvos em `experiments/`, e cada script reaproveita o
que já estiver em disco. Rodar de novo não retreina o que já foi treinado.
Em CPU, o conjunto todo leva algumas horas na primeira execução.

**3. Conferir o relatório** contra os experimentos:

```bash
python valida_numeros.py     # todo número do .tex contra o JSON que o produziu
python checa_relatorio.py    # estrutura do .tex, bibliografia e contagem de páginas
```

Os dados brutos, os pesos e o ambiente virtual **não** são versionados. São
reconstruídos pelos comandos acima. Ver `.gitignore`.

## Organização do repositório

```
relatorio/               o artigo em LaTeX (template SBC)
├── artigo.pdf             versão entregue, 15 páginas
├── artigo.tex             fonte
└── figuras/               figuras usadas no artigo

poc/                     código dos modelos e dos experimentos
├── xg_base.py             base compartilhada: dados, tokens, modelos, treino
├── exp0NN_*.py            um arquivo por experimento, numerado por ID
├── fig_*.py               regeneram figuras a partir dos JSONs salvos
├── viz.py                 estilo visual único de todas as figuras
├── valida_numeros.py      confere cada número do relatório contra seu JSON
├── checa_relatorio.py     sanidade do .tex antes de compilar
├── poc3_fetch_all.py      download das finalizações com freeze-frame
├── poc3_xg3.py            prova de conceito anterior, mantida como registro
└── build_dataset*.py,     linha abandonada: previsão de resultado 1X2
    poc2_*.py, train.py      a partir de escalações (European Soccer Database)

docs/                    documentação do projeto, comece por docs/README.md
├── METODOLOGIA.md         como o trabalho é conduzido e registrado
├── ESTADO_ATUAL.md        diagnóstico: o que existe, o que falta, riscos
├── DIARIO.md              o que aconteceu em cada sessão de trabalho
├── GLOSSARIO.md           todo termo técnico explicado
├── decisoes/              uma decisão técnica por arquivo, com alternativas
└── experimentos/          log append-only de resultados, com IDs e figuras
```

## Método

Duas regras organizam o repositório e existem para tornar o trabalho auditável:

1. **Nenhuma decisão técnica sem registro escrito.** Cada escolha vive em
   `docs/decisoes/`, com as alternativas que foram descartadas e por quê.
2. **Nenhum número no relatório sem ID de experimento** em
   `docs/experimentos/RESULTS.md`, que é *append-only*, inclusive para os
   experimentos que não funcionaram.

A segunda regra é verificável por `poc/valida_numeros.py`, que confere cada valor
do `artigo.tex` contra o JSON do experimento que o produziu.

## Licença e atribuição

Dados da **StatsBomb Open Data**, usados sob a licença de uso não comercial com
atribuição à fonte. O material didático da disciplina não é redistribuído aqui.

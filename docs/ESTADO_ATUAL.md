# Estado atual do projeto — diagnóstico

Levantamento feito em **2026-08-09**, a partir da leitura completa do repositório:
proposta enviada, feedback do professor, os 10 arquivos da PoC, os dados baixados
e os 14 notebooks das 9 aulas.

Marcação de confiança: **[medido]** eu rodei e vi · **[código]** está escrito no
repositório · **[curso]** vem do material das aulas · **[palpite]** intuição minha,
não verificada.

---

## 1. O que já existe e funciona

### 1.1 Dados — o ponto mais forte do projeto

`poc/shots_all.npz` **[medido]**:

| | |
|---|---|
| Finalizações | **99.746** |
| Partidas | **3.961** |
| Taxa de gol | **10,26%** |
| Competições-temporada | **80** |
| Jogadores visíveis por cena | média 13,1 · mínimo 1 · máximo 21 |

A proposta prometia "escalar os dados ~10×" a partir dos 6.347 chutes da PoC.
**Isso já está feito.** O download completo (`poc/poc3_fetch_all.py`) rodou e o
arquivo está em disco. Não há trabalho de aquisição de dados pendente.

Campos disponíveis por chute: posição do chutador (`sx`, `sy`), gol (0/1),
posição de até 21 jogadores (`px`, `py`), papel (`mate`, `gk`), máscara de
visibilidade (`mask`), cabeceio (`header`), `match_id` e competição.

### 1.2 Código

| Arquivo | O que faz | Estado |
|---|---|---|
| `poc/poc3_fetch_all.py` | Baixa todos os chutes com freeze-frame da StatsBomb | Funciona; já executado |
| `poc/poc3_xg3.py` | Escada completa B1 → B2 → Deep Sets → Transformer | Funciona; é a base de tudo |
| `poc/poc3_xg.py`, `poc3_xg2.py` | Versões anteriores (dataset pequeno, tokens mais pobres) | Histórico |
| `poc/build_dataset*.py`, `poc2_*.py`, `train.py` | PoCs 1 e 2 — previsão de resultado 1X2 a partir de escalações (European Soccer Database) | **Linha abandonada.** Contexto histórico, não entra no projeto final |

A arquitetura em `poc3_xg3.py` **[código]** já contém o essencial: tokens de 14
features com geometria relativa à linha do chute, `nn.TransformerEncoderLayer`
com `norm_first=True` e `batch_first=True`, token `[CLS]`, `src_key_padding_mask`,
split por partida, early stopping por AUC de validação, gradient clipping, 2 seeds.

### 1.3 Alinhamento com a proposta enviada

Tudo o que a proposta prometeu como *desenho* está implementado: escada de
baselines, split por partida, exclusão de pênaltis, tokens com geometria relativa,
Transformer pequeno sem positional encoding com `[CLS]`, PyTorch.

---

## 2. O que falta — lacunas mapeadas contra o feedback do professor

O professor listou sete pontos. Situação de cada um **[código]**:

| # | Pedido do professor | Existe hoje? |
|---|---|---|
| 1 | Calibrar probabilidades (Platt / isotônica) | **Não** |
| 2 | Reportar Brier score e curva de calibração | **Não** — só AUC e log loss |
| 3 | xG total previsto vs. gols reais por partida | **Não** |
| 4 | Features consagradas da literatura no baseline manual | **Parcial** — há 4 features (distância do goleiro, desvio do goleiro, bloqueadores, adversário mais próximo), mas sem referência bibliográfica |
| 5 | Velocidade/direção dos jogadores, ou reconhecer a limitação | **Não** — nem tentado, nem documentado |
| 6 | Validar mapas de atenção com casos conhecidos | **Não** — nenhuma extração de atenção implementada |
| 7 | Copa 2026 como análise qualitativa, não avaliação | **Não** — nenhuma análise da Copa ainda |

### 2.1 Divergências entre a proposta e o código

Encontrei duas afirmações da proposta que o código **ainda não sustenta** — vale
resolver antes que virem pergunta na arguição:

1. **"testes estatísticos (bootstrap/McNemar)"** — a proposta afirma isso; o
   código não tem nenhum teste estatístico **[código]**. Sem isso, não dá para
   dizer que uma diferença de AUC entre dois modelos é real e não ruído.
2. **"múltiplas seeds"** — o código roda **2 seeds**. Duas seeds dão uma média,
   mas não sustentam afirmação sobre variabilidade **[palpite]**: com 2 pontos o
   desvio padrão é praticamente não informativo.

Também vale registrar: os números citados na proposta (B1 = 0,709 · B2 = 0,736 ·
neurais ≈ 0,71) vieram do **dataset pequeno de 6.347 chutes**, não da base
completa. Reproduzi-los em escala é o experimento EXP-000.

---

## 3. Riscos e gargalos

### 3.1 Prazo — 5 dias
Entrega 2 em **14 de agosto de 2026**: relatório SBC, até 15 páginas, com link
para o repositório. É o gargalo dominante e deve ditar o escopo.

### 3.2 Sem GPU **[medido]**
`torch 2.12.1+cpu`, `cuda disponível: False`, 12 núcleos. O modelo é pequeno
(2 camadas, dim 48, 22 tokens), então treinar em CPU é viável — mas **cada
experimento custa minutos, não segundos**. Isso limita quantas variações cabem
no prazo e é argumento a favor de escolher poucos experimentos bem definidos.

### 3.3 `matplotlib` não está instalado **[medido]**
O ambiente `poc/.venv` não tem matplotlib nem seaborn. **Sem eles não existe curva
de calibração, nem heatmap de atenção, nem qualquer figura do relatório.** É a
dependência mais urgente.

### 3.4 O projeto não está versionado **[medido]**
Não existe `.git`. A proposta afirma "código versionado no GitHub desde a PoC", e
o relatório exige link para o repositório. Reprodutibilidade vale 15% da nota.

### 3.5 Risco científico — a resposta pode ser "não"
Na PoC em escala pequena, o Transformer (~0,71) ficou **abaixo** do baseline com
features manuais (0,736). Se isso se mantiver na base completa, a resposta à
pergunta central é negativa. Isso não invalida o trabalho, mas **muda o que o
relatório precisa mostrar**: em vez de "o modelo aprendeu a geometria sozinho",
passa a ser "medimos que não aprendeu, e investigamos por quê". Essa decisão
precisa ser tomada com antecedência, não sob pressão.

### 3.6 Composição dos dados — ponto que o professor pode questionar **[medido]**
- **30,7% das finalizações vêm de competições femininas** (Liga F, NWSL, Serie A
  Women, FA WSL). As taxas de gol são quase idênticas (10,4% vs 10,2%), então
  misturar não é obviamente errado — mas é uma decisão que precisa ser
  **declarada e justificada**, não silenciosa.
- **Metade dos dados é de temporadas 2015/16** (Premier League, La Liga, Serie A,
  Ligue 1) e a outra metade é recente. Padrões táticos mudaram no período.
- **Em 0,1% das cenas o goleiro adversário não aparece** no freeze-frame. O código
  atual preenche a distância do goleiro com o valor fixo `25.0` **[código]** —
  um remendo que funciona, mas que precisa ou de justificativa ou de tratamento
  melhor.
- A média de 13 jogadores visíveis (não 22) significa que **a máscara de padding
  não é um detalhe**: é o caso comum, não a exceção.

---

## 4. O que o material do curso oferece

Levantamento completo dos 14 notebooks. O que é **diretamente aproveitável**:

| Recurso | Onde | Uso no projeto |
|---|---|---|
| `MultiHeadAttention` manual, batch-first, com máscara nos *keys* | aula 06 torch | Base do encoder, se quisermos versão própria em vez da do PyTorch |
| `EncoderBlock` (residual + FFN 4× + LayerNorm) | aula 06 torch | Idem |
| 3 funções de visualização de atenção (heatmap por cabeça; hooks para `atenção×V`; cosseno + PCA) | aula 06 torch | **Atende ao pedido 6 do professor** |
| `train_one_epoch` / `evaluate` / `run_training` | aula 03 torch | Esqueleto do loop de treino |
| Early stopping com `deepcopy(state_dict())` | aula 02 torch | Já usado na PoC |
| *Masked mean pooling* | aula 04 e aula 07 | Alternativa ao `[CLS]` para agregar os 22 tokens |
| Partial dependence manual (varre 2 fatores, plota P(evento)) | aula 07 | "Fixa a cena, varre distância × ângulo, plota xG" |
| AdamW + cosine schedule + grad clipping | aula 06/07 torch | Receita de treino |

O que **não existe em nenhum notebook** e teremos de escrever do zero:

- Projeção linear de features contínuas no lugar de `nn.Embedding`.
- Token `[CLS]` aprendível com extensão da máscara.
- **Qualquer métrica de probabilidade**: não há ROC-AUC, Brier, log loss como
  métrica, nem curva de calibração em nenhuma das 9 aulas. Todo o material reporta
  acurácia. As métricas que o professor pediu vêm de fora do curso.
- Tratamento de classe desbalanceada.

### Cuidado ao citar o curso no relatório
A aula 06 apresenta o positional encoding como **necessário** e **em momento
algum diz que é opcional ou removível**. Remover o PE é uma decisão *nossa*,
justificada pela natureza de conjunto da cena — e deve ser apresentada assim, com
argumento próprio, não como algo que a aula ensinou.

Detalhe elegante e legítimo de citar: a aula 04 mostra que o pooling médio ignora
a ordem das palavras e trata isso como **defeito** ("o cachorro mordeu o homem" =
"o homem mordeu o cachorro"). No nosso problema, a mesma invariância é a
**propriedade desejada**, porque a cena é um conjunto. Mesmo fenômeno, sinal
invertido — é um bom parágrafo de fundamentação.

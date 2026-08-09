# 0002 — O que conta como "tentei de verdade" antes de aceitar o resultado negativo

- **Status:** **APROVADA**
- **Data:** 2026-08-09
- **Decide:** Vinícius
- **Como foi aprovada:** as três decisões abertas foram apresentadas em detalhe,
  com alternativas e recomendação, e o Vinícius respondeu "vamos avançar o
  projeto". Registro isso como aprovação da recomendação. **Se não era essa a
  intenção em algum dos cartões, é só dizer — nada aqui é irreversível, e reverter
  agora custa minutos.**

## Contexto

Você decidiu (9/ago): aceitar um resultado negativo, **mas não desistir no
primeiro número ruim** — persistir e mostrar que a tentativa foi genuína.

O risco dessa posição, se ficar informal, é o oposto do pretendido: "persistir um
pouco" não tem fim definido, e a 3 dias do prazo isso vira ou pânico ou uma
sequência de ajustes improvisados que ninguém consegue justificar depois. Pior: se
mexermos em muitas coisas ao mesmo tempo e o número melhorar, **não saberemos por
quê** — e um ganho inexplicado é mais frágil na arguição que um resultado negativo
bem investigado.

Precisamos transformar "tentei bastante" em algo auditável.

## Alternativas

### A) Persistir por tempo ("mexo até dia 12")
**Custo:** nenhum de planejamento.
**Risco:** alto. Incentiva mexer em várias coisas de uma vez para "aproveitar o
tempo", que é exatamente o que impede atribuir causa. E não gera nada de citável
no relatório.

### B) Persistir por número de tentativas, cada uma com hipótese declarada
Fixamos **N tentativas de melhoria**. Antes de cada uma, escrevo uma linha:
*"hipótese: o Transformer está limitado por X; se eu mudar Y, o Brier melhora"*.
Rodo, registro no log com ID, e digo se a hipótese se confirmou.
**Custo:** poucos minutos por tentativa para escrever a hipótese.
**Risco:** N pode ser mal escolhido — mas isso é ajustável e visível.

### C) Persistir até estabilizar (parar quando K tentativas seguidas não melhorarem)
Mais rigoroso estatisticamente.
**Custo:** imprevisível em tempo — pode não convergir antes do prazo.
**Risco:** incompatível com sua prioridade nº 1, que é entregar dentro do prazo.

## Recomendação

**B, com N = 4.** As quatro tentativas, na ordem em que eu as faria — cada uma
ataca uma causa *diferente* e plausível para o Transformer não superar o baseline:

| # | Hipótese | O que muda | Por que essa hipótese é plausível |
|---|---|---|---|
| T1 | O modelo está mal calibrado, não mal informado | Platt / isotônica sobre a saída | AUC decente com Brier ruim é a assinatura clássica disso |
| T2 | O sinal está lá, mas o modelo é pequeno/regularizado demais | Aumentar dimensão e cabeças, ajustar dropout | 2 camadas × dim 48 é bem modesto para 70k amostras |
| T3 | Os tokens não carregam a informação certa | Revisar features por jogador (o que a atenção *pode* comparar) | A atenção só combina o que está no token; se a geometria relevante não está lá, nenhuma arquitetura a inventa |
| T4 | O desbalanceamento (10% de gols) está achatando o aprendizado | `pos_weight` na perda ou reamostragem | Evento raro é causa conhecida de modelo conservador |

Regra de ouro: **uma mudança por tentativa.** Se mudarmos duas coisas juntas e
melhorar, perdemos a capacidade de dizer qual funcionou — e é justamente essa
atribuição que vale nota em "análise crítica" (25%).

Se as quatro falharem, **aí sim** o resultado negativo está estabelecido, e o
relatório passa a ter algo muito mais forte que "não funcionou": tem quatro
hipóteses testadas e descartadas, com evidência de cada uma. Isso é uma seção de
Lições Aprendidas pronta.

## Como saberemos se foi a escolha certa

- Se alguma tentativa **melhorar** o Brier de teste além do intervalo de confiança
  por bootstrap: hipótese confirmada, e a explicação do porquê vira parágrafo do
  relatório.
- Se as quatro falharem: encerramos e escrevemos o resultado negativo, com as
  quatro hipóteses como corpo da análise.
- **Critério de parada rígido:** se chegarmos ao dia **12/ago** com tentativas
  pendentes, paramos onde estiver e escrevemos. Escrever o relatório nunca é
  sacrificado por mais um experimento — é ele que carrega 50% da nota entre
  clareza e análise crítica.

## Consequências

- Cada tentativa vira um ID (`EXP-006` a `EXP-009`) e uma linha no log, **inclusive
  as que falharem** — o professor pediu isso explicitamente.
- O relatório ganha uma tabela "hipótese → mudança → resultado" que é, sozinha, a
  evidência de que a investigação foi séria.
- Ficam **fora** deste contrato: trocar de arquitetura, mudar a tarefa ou buscar
  outra fonte de dados. Não cabem no prazo.

## Decisão

**Aprovada** conforme a recomendação (N = 4), em 2026-08-09.

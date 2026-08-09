# Metodologia de trabalho — projeto xG com set Transformer

Este documento é o **contrato entre nós dois**. Ele existe por um motivo único:
garantir que você entenda 100% do que está sendo construído e por quê, e que
consiga questionar, propor alternativas e identificar gargalos em qualquer ponto.

Se em algum momento eu violar uma das regras abaixo, isso é um **defeito de
processo** — me cobre, e a correção tem prioridade sobre qualquer código.

---

## 1. A regra que sustenta todas as outras

> **Nenhuma linha de código entra no projeto sem que exista, antes, uma decisão
> escrita que você leu e aprovou.**

Não é burocracia: é o mecanismo que impede que o projeto vire uma caixa-preta que
você "recebeu pronta". Você precisa poder defender cada escolha numa arguição, e
não dá para defender o que não se decidiu.

---

## 2. O ciclo de três etapas

Todo incremento do projeto — sem exceção — passa por três etapas nesta ordem.

### Etapa 1 — DECIDIR (antes de qualquer código)

Eu escrevo um **cartão de decisão** em `docs/decisoes/NNNN-titulo.md`, curto (1
página), com esta estrutura fixa:

- **Contexto** — que problema apareceu, e por que precisa ser resolvido agora.
- **Alternativas** — sempre **no mínimo duas**, com o custo e o risco de cada uma.
  Se eu só apresentar uma opção, é sinal de que não pensei o suficiente: me cobre.
- **Recomendação** — qual eu escolheria e por quê. Eu recomendo; **você decide**.
- **Como saberemos se foi a escolha certa** — o critério de sucesso, definido
  *antes* de rodar. Isso impede a racionalização depois do resultado.
- **Consequências** — o que essa escolha fecha, o que ela obriga a fazer depois.

Você responde de uma destas formas: **aprovo** / **aprovo com mudança X** /
**quero entender melhor Y** / **prefiro a alternativa B**. Só depois eu codifico.

### Etapa 2 — IMPLEMENTAR

- Código em módulos pequenos, com nomes que dizem a intenção.
- Comentários explicam **por quê**, nunca **o quê** (o código já diz o quê).
- Junto com o código, eu te entrego um **walkthrough**: explicação em prosa das
  partes não óbvias, com os *shapes* dos tensores em cada passo. Deep learning
  é, na prática, contabilidade de dimensões — se você acompanha os shapes, você
  entende o modelo.
- Toda função que faz matemática não trivial (geometria do campo, máscara de
  atenção, calibração) vem com um teste que você consegue ler e conferir na mão.

### Etapa 3 — EVIDENCIAR

- Rodo o experimento e registro em `docs/experimentos/RESULTS.md`, que é
  **append-only**: nunca apago nem edito uma linha antiga. Resultado ruim fica
  registrado — o professor pediu explicitamente que experimentos que não
  funcionaram sejam documentados, e eles valem nota.
- Cada experimento tem um **ID** (`EXP-001`, `EXP-002`, …). **Nenhum número entra
  no relatório final sem um ID que o produziu.** Se eu citar uma métrica sem ID,
  ela não existe.
- Comparo o resultado com o critério de sucesso definido na Etapa 1 e digo
  claramente: bateu, não bateu, ou o critério estava mal formulado.

---

## 3. Como eu falo com você

### 3.1 Semáforo de confiança

Toda afirmação técnica que eu fizer vem marcada:

| Marca | Significado | O que você deve fazer |
|---|---|---|
| **[medido]** | Eu rodei e vi o número. Tem ID de experimento. | Pode confiar; confira o ID. |
| **[literatura]** | Vem de artigo/documentação. Com a fonte junto. | Confiável, mas a fonte pode não se aplicar ao nosso caso. |
| **[curso]** | Vem do material das aulas do Prof. Prati. | Confiável e **citável no relatório** — vale ponto. |
| **[palpite]** | É minha intuição de engenharia. Não verificada. | **Duvide.** Peça para eu medir antes de construir em cima. |

O erro mais caro que eu poderia cometer com você é apresentar palpite com cara de
fato. O semáforo existe para tornar isso impossível de passar despercebido.

### 3.2 Explicação em três níveis

Quando eu explicar qualquer componente, faço em três camadas, e você escolhe onde
parar:

1. **Uma frase** — o que faz e por que existe.
2. **A intuição** — analogia ou desenho mental, sem matemática.
3. **A mecânica** — a matemática e os shapes, linha a linha.

Você nunca precisa fingir que entendeu o nível 3 para seguir adiante. Dizer
"parei no nível 2" é informação útil, não fraqueza.

### 3.3 Checagem de entendimento

Ao fim de cada bloco eu te faço **2 ou 3 perguntas curtas** sobre o que
construímos. Não é prova: é diagnóstico da minha explicação. Se você não
conseguir responder, **a explicação foi ruim e eu refaço** — a falha é minha,
não sua.

### 3.4 Nada de silêncio sobre limitação

Sempre que uma escolha tiver um custo escondido — um viés nos dados, uma métrica
que engana, uma simplificação que o professor pode questionar na arguição — eu
levanto **na hora**, não quando der problema.

---

## 4. Seus canais

| Arquivo | Para quê |
|---|---|
| `docs/PERGUNTAS.md` | Fila de dúvidas suas. Escreva a qualquer momento, sem formatar. Nada é respondido "depois e esquecido": toda pergunta é fechada com resposta escrita. |
| `docs/GLOSSARIO.md` | Todo termo técnico do projeto explicado no nível de quem está aprendendo. Se eu usar um termo que não está lá, é falha minha. |
| `docs/decisoes/` | O histórico de **por que** o projeto é como é. É daqui que sai boa parte do relatório final. |
| `docs/experimentos/RESULTS.md` | O histórico de **o que aconteceu** quando rodamos. |
| `docs/DIARIO.md` | Ordem cronológica das sessões: o que foi feito, o que ficou pendente. |

---

## 5. Reprodutibilidade (vale 15% da nota)

Regras técnicas inegociáveis:

1. **Um experimento = um comando.** Nada de "rode isso, depois edite aquela linha
   e rode de novo". Se não dá para reproduzir com um comando, não está pronto.
2. **Seed fixa e explícita** em todo experimento; resultados de modelo neural
   sempre com **múltiplas seeds**, reportando média e desvio.
3. **Configuração é dado, não código.** Hiperparâmetros ficam num arquivo de
   config versionado, não espalhados em números mágicos.
4. **Split por partida, sempre.** Chutes da mesma partida nunca aparecem em
   splits diferentes — é o que impede vazamento entre lances correlacionados.
5. **Saídas vão para `experiments/<ID>/`**, incluindo config, métricas e log.
   Rodar de novo não sobrescreve histórico.

---

## 6. Como lidamos com resultado negativo

A pergunta central do projeto é: *o Transformer aprende sozinho a geometria de
interação que hoje é feita à mão?* **A resposta pode ser "não".**

Se for, isso **não é fracasso do projeto** — é o resultado. O que decide a nota é
o rigor com que a resposta foi obtida e analisada, não o sinal dela. A regra:

- Nunca ajustamos o experimento até o número ficar bonito. Definimos o critério
  antes (Etapa 1) e reportamos o que deu.
- Um resultado negativo bem investigado ("o modelo não superou o baseline, e
  medimos que a causa provável é X") vale mais que um positivo inexplicado.
- Todo experimento que falhou entra no relatório, na seção de lições aprendidas.

---

## 7. O que eu **não** vou fazer

- Não vou escrever código de projeto sem cartão de decisão aprovado.
- Não vou usar biblioteca nova sem justificar por que a atual não serve.
- Não vou apresentar número sem ID de experimento.
- Não vou esconder um resultado ruim nem apagar uma tentativa fracassada.
- Não vou dizer "confia" — se eu não consigo explicar, é porque eu não entendi.

# 0001 — Qual métrica decide se um modelo é melhor que outro

- **Status:** proposta
- **Data:** 2026-08-09
- **Decide:** Vinícius

> Este é o primeiro cartão de decisão e serve também de exemplo do formato.
> Ele é curto de propósito: se um cartão não cabe em uma página, a decisão está
> mal recortada.

## Contexto

Todo o projeto é uma comparação entre quatro modelos (B1, B2, Deep Sets,
Transformer). Para dizer "o modelo X é melhor que o Y" precisamos de **uma**
métrica que decide. Hoje o código reporta AUC e log loss, e usa **AUC** tanto
para o early stopping quanto para escolher o melhor modelo **[código]**.

O problema: o professor apontou que xG é uma **probabilidade**, e pediu Brier
score e curva de calibração. Isso não é um detalhe de relatório — muda qual
modelo ganha. Um modelo pode ordenar chutes perfeitamente (AUC alta) e ainda
assim dizer "40% de gol" onde o correto seria "10%", o que torna o xG inútil na
prática.

Escolher a métrica **antes** de rodar os experimentos é o que impede a
racionalização depois ("olha, nesta outra métrica o meu modelo ganha").

## Alternativas

### A) Manter AUC como métrica de decisão
Reporta Brier e calibração apenas como métricas secundárias no relatório.
**Custo:** zero, já está implementado.
**Risco:** alto. Contraria diretamente o feedback do professor, e a AUC é cega
para o erro que mais importa em xG. Podemos escolher como "melhor" um modelo cuja
probabilidade é ruim.

### B) Log loss como métrica de decisão
É a própria função de perda do treino, então otimização e seleção ficam
coerentes. Penaliza confiança errada com força.
**Custo:** baixo, já é calculada.
**Risco:** é muito sensível a previsões extremas — um único chute com
probabilidade quase zero que vira gol pesa desproporcionalmente. Com evento raro
(10% de gols) isso pode gerar instabilidade entre seeds **[palpite]**.

### C) Brier score como métrica de decisão
Erro quadrático médio entre probabilidade prevista e resultado. Penaliza má
calibração, é limitado (não explode com um único caso extremo) e é a métrica
padrão da literatura de previsão probabilística esportiva.
**Custo:** baixo, uma função.
**Risco:** é menos sensível que log loss a diferenças na cauda — dois modelos
podem ter Brier parecido e comportamentos distintos nos chutes muito prováveis.

### D) Brier decide; AUC e calibração acompanham obrigatoriamente
Brier é o critério único de decisão. AUC entra sempre ao lado, para separar
"ordena bem" de "acerta o valor", e a curva de calibração entra como diagnóstico.
**Custo:** baixo.
**Risco:** exige disciplina para não trocar de critério no meio do caminho.

## Recomendação

**D.** O raciocínio: a AUC responde "o modelo sabe quais chutes são mais
perigosos?" e o Brier responde "o modelo sabe *quanto* perigosos?". A segunda é a
pergunta do xG, mas a primeira é a que separa dois tipos de fracasso — um modelo
com AUC boa e Brier ruim precisa só de recalibração, enquanto AUC ruim significa
que não aprendeu nada. Reportar as duas juntas torna o diagnóstico possível;
eleger o Brier como juiz evita a ambiguidade de ter dois critérios.

Consequência prática imediata: **trocar o early stopping de AUC para Brier na
validação**, para que o modelo selecionado durante o treino seja o mesmo tipo de
modelo que declaramos melhor no fim.

## Como saberemos se foi a escolha certa

Em **EXP-001**, ao reavaliar os quatro modelos com todas as métricas:

- Se o ranking por Brier e por AUC for **o mesmo**, a decisão foi inócua — e isso
  também é informação útil para o relatório.
- Se os rankings **divergirem**, a decisão foi decisiva, e a curva de calibração
  vai mostrar exatamente por quê. Esse caso vira figura e parágrafo no relatório.

## Consequências

- O early stopping passa a monitorar Brier de validação; EXP-000 (que usou AUC)
  fica como registro histórico e não é comparável linha a linha com os próximos.
- Todas as tabelas do relatório passam a ter as colunas: AUC · log loss · Brier ·
  ECE.
- Se adotarmos calibração (Platt/isotônica) em `0002`, a comparação passa a ser
  **antes vs. depois** da calibração, e o Brier é o que mede o ganho.

## Decisão

*(a preencher por você: aprovo / aprovo com mudança X / prefiro a alternativa B /
quero entender melhor Y)*

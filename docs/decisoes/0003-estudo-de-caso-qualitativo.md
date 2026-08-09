# 0003 — Qual torneio serve de estudo de caso qualitativo

- **Status:** **APROVADA**
- **Data:** 2026-08-09
- **Decide:** Vinícius
- **Como foi aprovada:** apresentada em detalhe junto com os cartões 0001 e 0002;
  o Vinícius respondeu "vamos avançar o projeto". Registro como aprovação da
  recomendação. Reversível a qualquer momento.

## Contexto

A proposta enviada promete a **Copa do Mundo de 2026** como estudo de caso
qualitativo, e o professor comentou o ponto favoravelmente ("ótimo gancho
qualitativo — deixar claro que é análise, não avaliação quantitativa").

**A Copa de 2026 não existe na StatsBomb Open Data** [medido, 2026-08-09]:
consultei o `competitions.json` do repositório público e a competição mais recente
é a Eurocopa Feminina de 2025; as Copas do Mundo masculinas vão até 2022. Não é
falha de download — o dado não foi publicado.

Isso é uma promessa da proposta que o projeto **não tem como cumprir**. Silenciar
é o pior caminho: se o professor abrir o repositório procurando a Copa de 2026 e
não encontrar, a pergunta vem sem que a resposta esteja preparada.

## Alternativas

### A) Trocar por Eurocopa 2024 ou Copa América 2024
Torneios de seleções mais recentes disponíveis: 1.304 e 741 finalizações.
**Custo:** nenhum — o dado já está baixado.
**Risco:** nenhum técnico. Exige um parágrafo honesto explicando a troca.

### B) Manter a Copa de 2026 como trabalho futuro declarado
Faz a análise qualitativa sobre lances de qualquer competição, sem recorte de
torneio.
**Custo:** nenhum.
**Risco:** perde o enquadramento de "estudo de caso", que era o que dava unidade
narrativa à análise qualitativa.

### C) Buscar outra fonte para a Copa de 2026
**Custo:** alto e imprevisível, a 5 dias do prazo.
**Risco:** provavelmente fora da licença de uso que a proposta declarou.
Incompatível com a prioridade de entregar dentro do prazo.

## Recomendação

**A — Eurocopa 2024.** Mantém a análise qualitativa que o professor elogiou, usa
dado real e já disponível, e o torneio é de seleções, como a Copa seria. A
explicação da troca é uma **limitação declarada**, e limitação declarada
costuma contar a favor: mostra que o autor conferiu a fonte em vez de assumir.

Redação sugerida para o relatório: *"A proposta previa a Copa do Mundo de 2026
como estudo de caso. Em consulta à StatsBomb Open Data em 09/08/2026, verificou-se
que a competição ainda não havia sido publicada — a mais recente disponível é a
Eurocopa Feminina de 2025. O estudo de caso foi então conduzido sobre a Eurocopa
2024 (1.304 finalizações), o torneio de seleções mais recente na base."*

## Como saberemos se foi a escolha certa

A análise qualitativa precisa produzir pelo menos **um lance em que o mapa de
atenção seja legível e defensável** — por exemplo, um chute em que o modelo
atribua peso alto ao goleiro adiantado ou a um defensor na linha do chute. Se
nenhum lance da Eurocopa 2024 produzir isso, o problema é do modelo, não do
torneio, e a troca terá sido irrelevante para o resultado.

## Consequências

- O estudo de caso passa a ser a **Eurocopa 2024**; a Copa de 2026 vira uma frase
  na seção de limitações e uma linha em trabalhos futuros.
- Depende do **EXP-005** (mapas de atenção). Se o EXP-005 for cortado por falta de
  tempo, este cartão cai junto — e a seção 3.5 do relatório sai.
- A proposta enviada fica com uma promessa não cumprida, **explicada**. Isso entra
  em Lições Aprendidas: verificar a disponibilidade do dado antes de prometê-lo.

## Decisão

**Aprovada** conforme a recomendação (Eurocopa 2024), em 2026-08-09.

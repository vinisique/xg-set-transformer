# Registro de decisões

Cada decisão técnica do projeto vira um arquivo numerado aqui. Este diretório é a
resposta à pergunta *"por que o projeto é assim?"* — e é a matéria-prima das
seções **Abordagem** e **Lições Aprendidas** do relatório final.

## Índice

| # | Decisão | Status | Data |
|---|---|---|---|
| [0001](0001-metrica-de-decisao.md) | Qual métrica decide se um modelo é melhor que outro | **proposta** — aguardando você | 2026-08-09 |
| [0002](0002-contrato-de-persistencia.md) | O que conta como "tentei de verdade" antes de aceitar resultado negativo | **proposta** — aguardando você | 2026-08-09 |

Status possíveis: `proposta` (aguardando sua resposta) · `aprovada` ·
`rejeitada` · `revista por NNNN` (uma decisão posterior mudou esta).

**Decisão nunca é apagada.** Se mudarmos de ideia, a antiga fica com status
`revista por NNNN` e a nova explica o que aprendemos no caminho. O histórico das
ideias abandonadas é justamente o que o professor pediu que fosse documentado.

---

## Template

Copie o bloco abaixo para `NNNN-titulo-curto.md`.

```markdown
# NNNN — <título>

- **Status:** proposta
- **Data:** AAAA-MM-DD
- **Decide:** Vinícius

## Contexto
Que problema apareceu e por que precisa ser resolvido agora.

## Alternativas

### A) <nome>
Como funciona. **Custo:** … **Risco:** …

### B) <nome>
Como funciona. **Custo:** … **Risco:** …

## Recomendação
Qual eu escolheria e por quê. (Eu recomendo, você decide.)

## Como saberemos se foi a escolha certa
Critério objetivo, definido ANTES de rodar. Ex.: "EXP-00X deve mostrar
Brier ≤ 0,075 no teste; se não, revisitamos esta decisão."

## Consequências
O que essa escolha fecha e o que ela obriga a fazer depois.

## Decisão
(preenchido por você: aprovo / aprovo com mudança X / prefiro B / quero entender Y)
```

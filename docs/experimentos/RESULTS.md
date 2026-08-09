# Log de experimentos

**Append-only.** Nenhuma linha é editada ou apagada depois de escrita, nem quando
o resultado é ruim — resultado ruim é dado, e o professor pediu explicitamente
que experimentos malsucedidos fossem documentados.

**Regra de ouro:** nenhum número entra no relatório final sem um ID desta tabela.

Convenções:
- **AUC** — só ordenação; não enxerga calibração. Maior é melhor.
- **Log loss** e **Brier** — qualidade da probabilidade. Menor é melhor.
- Modelos neurais: sempre média ± desvio sobre múltiplas seeds.
- Split sempre **por partida** (70/15/15), semente do split fixa em 0.

---

## Índice

| ID | O que testou | Dados | Resultado em uma linha |
|---|---|---|---|
| EXP-000 | Linha de base herdada da PoC (B1/B2/DeepSets/Transformer) | `shots_all.npz` | *(rodando — a preencher)* |

---

## EXP-000 — linha de base herdada da PoC

- **Data:** 2026-08-09
- **Script:** `poc/poc3_xg3.py` (código da PoC, sem alterações)
- **Dados:** `poc/shots_all.npz` — 99.746 finalizações, 3.961 partidas, 10,26% gols
- **Objetivo:** estabelecer o ponto de partida real em dados completos. Os números
  citados na proposta (B1 = 0,709 · B2 = 0,736 · neurais ≈ 0,71) vieram do dataset
  pequeno de 6.347 chutes, e **nunca foram reproduzidos na base completa**.
- **Configuração:** dim 48, 2 camadas, 4 cabeças, dropout 0,1, AdamW lr 1e-3,
  weight decay 1e-3, batch 256, early stopping por AUC de validação (paciência 8),
  máximo 60 épocas, seeds 0 e 1.

**Resultado:** *(a preencher quando a execução terminar)*

**Observação:** este experimento serve de marco zero. Ele **não** inclui
calibração, Brier score nem teste estatístico — exatamente as lacunas que o
feedback do professor apontou.

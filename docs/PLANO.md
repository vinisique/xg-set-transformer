# Plano até a Entrega 2 (14 de agosto de 2026)

Restam **5 dias**. Este plano é uma proposta — cada etapa vira um cartão de
decisão que você aprova antes de eu codificar.

Princípio que organiza tudo: **primeiro medir direito, depois melhorar.** Não
adianta ajustar arquitetura enquanto as métricas que o professor pediu não
existem — sem Brier e curva de calibração não temos como saber se uma mudança
melhorou ou piorou o que importa.

---

## Decisões tomadas em 9/ago (respostas em `PERGUNTAS.md`)

1. **Objetivo: trabalho sólido dentro do prazo.** Escopo conservador; a ordem de
   corte no fim deste documento vale como está.
2. **Não reescrevemos o modelo.** Seguimos com o `forward` de atenção mascarada
   que já existe em `poc/poc3_xg3.py`. O entendimento vem por walkthrough escrito,
   em paralelo, sem custar dias de reimplementação.
3. **Resultado negativo é aceitável, desistir cedo não é.** Persistimos com um
   número fixo de tentativas, cada uma com hipótese declarada antes de rodar.
4. **GitHub público**, sem material de terceiros e sem coautoria de IA nos commits.

## Marco 0 — Fundação (hoje, 9/ago)

| Tarefa | Por quê | Estado |
|---|---|---|
| Instalar `matplotlib` | Sem ele não existe figura nenhuma no relatório | **Feito** (3.11.1) |
| `git init` + `.gitignore` + primeiro commit | Vale 15% da nota e o relatório exige o link | **Feito** |
| Criar o repositório público no GitHub e dar `push` | Precisa da sua conta | Pendente (você) |
| EXP-000 — linha de base em escala real | Saber de onde partimos de fato | Rodando |
| Cartão `0001` — métrica de decisão | Define quem "ganha" nas comparações | Aguardando você |
| Cartão `0002` — contrato de persistência | Define o que conta como "tentei de verdade" | A escrever |
| Decisão sobre a Copa de 2026 | O dado não existe na fonte | Aguardando você |

## Marco 1 — Avaliação correta (10/ago)

O módulo de métricas é a peça mais importante do projeto, porque **todo o resto é
julgado por ele**.

- Implementar: AUC, log loss, **Brier score**, **curva de calibração**, **ECE**,
  **calibração agregada** (xG somado por partida vs. gols reais) e **intervalo de
  confiança por bootstrap**.
- Reavaliar os quatro modelos existentes com esse módulo → **EXP-001**.
- Cartão de decisão: qual métrica é a **métrica de decisão** do projeto. Minha
  recomendação é log loss ou Brier, não AUC — porque xG é probabilidade, e a AUC
  é cega para calibração.

## Marco 2 — Calibração e baseline honesto (11/ago)

- Platt scaling e regressão isotônica ajustados **na validação**, nunca no teste
  → **EXP-002**. Atende aos pedidos 1 e 2 do professor.
- Revisar o baseline de features manuais com as features consagradas da
  literatura de xG, **com citação** → **EXP-003**. Atende ao pedido 4.
- Registrar a limitação de velocidade/direção: o freeze-frame da StatsBomb dá
  posição, não velocidade. Atende ao pedido 5.

## Marco 3 — O experimento científico central (12/ago)

Aqui respondemos a pergunta do projeto.

- Transformer vs. Deep Sets vs. B2, com **5 ou mais seeds** e **teste estatístico**
  (bootstrap pareado) → **EXP-004**. Fecha a divergência entre a proposta, que
  promete teste estatístico, e o código, que não tem nenhum.
- **Mapas de atenção**: extrair os pesos do `[CLS]` e verificar se o modelo olha
  para o goleiro e para os bloqueadores sem nunca ter recebido isso como feature
  → **EXP-005**. Atende ao pedido 6. As três funções de visualização da aula 06
  são reaproveitáveis aqui.

## Marco 4 — Escrita (13/ago)

- Relatório no template SBC, até 15 páginas.
- Figuras: curva de calibração, comparação de modelos, mapas de atenção.
- Seção de lições aprendidas, alimentada por `docs/decisoes/` e pelos
  experimentos que não deram certo.

## Marco 5 — Revisão e entrega (14/ago)

- Conferir que **todo número do relatório tem um ID de experimento**.
- README do repositório com instruções de reprodução.

---

## Escopo mínimo viável

Se o tempo apertar, esta é a ordem em que eu cortaria — do menos ao mais
sacrificável:

1. **Nunca corta:** métricas de calibração + escada de baselines + relatório.
   É o núcleo do que o professor pediu.
2. **Corta se precisar:** mapas de atenção (vira "trabalho futuro").
3. **Corta primeiro:** qualquer ajuste de arquitetura além do que já existe.

---

## Problema que precisa de decisão sua

### A Copa de 2026 não está disponível **[medido, 9/ago]**

A proposta enviada promete a Copa do Mundo de 2026 como estudo de caso
qualitativo, e o professor comentou o ponto de forma favorável. Só que a
**StatsBomb Open Data não publicou a Copa de 2026**: consultei o `competitions.json`
hoje e a competição mais recente do repositório é a Eurocopa Feminina de 2025. As
Copas disponíveis vão até 2022.

Não é problema de download nosso — o dado não existe na fonte pública.

Alternativas:

- **A) Trocar o estudo de caso** por Eurocopa 2024 ou Copa América 2024 (torneios
  de seleções mais recentes disponíveis, com 1.304 e 741 finalizações), e
  explicar no relatório por que a Copa de 2026 não pôde ser usada.
- **B) Manter a Copa de 2026** como trabalho futuro declarado, e fazer a análise
  qualitativa sobre lances escolhidos de qualquer competição.
- **C) Procurar outra fonte** para a Copa de 2026 — improvável em 5 dias e fora do
  escopo da licença que a proposta declarou **[palpite]**.

**Minha recomendação: A.** Mantém a análise qualitativa que o professor elogiou,
usa dado real e disponível, e a explicação da troca é honesta — o tipo de
limitação declarada que costuma contar a favor, não contra.

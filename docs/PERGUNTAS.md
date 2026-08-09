# Fila de perguntas

Escreva aqui qualquer dúvida, a qualquer momento, sem se preocupar com formato.
Uma linha solta serve. **Nenhuma pergunta é respondida "de passagem" e esquecida:**
toda pergunta é fechada com resposta escrita neste arquivo.

Não existe pergunta boba. Se você precisou perguntar, a explicação anterior falhou
— e consertar isso é responsabilidade minha, não sua.

---

## Abertas

*(vazio — escreva abaixo)*

---

## Respondidas

*(as perguntas fechadas migram para cá, com a resposta junto, para virar
material de consulta)*

---

## Perguntas que EU tinha para você — RESPONDIDAS em 2026-08-09

### 1. Objetivo da Entrega 2
**Resposta: entregar um trabalho sólido dentro do prazo.**
→ Consequência: escopo fechado e conservador. Nada de explorar arquitetura além
do necessário. A ordem de corte do `PLANO.md` vale como está.

### 2. Conforto com PyTorch / reconstruir do zero?
**Resposta: seguir com o `forward` de atenção mascarada que já existe, mesmo que
o entendimento não seja 100% imediato.**
→ Consequência: **não reescrevemos o modelo.** Aproveitamos `poc/poc3_xg3.py`,
que já funciona. Em compensação, assumo a dívida de explicar o `forward` linha a
linha com os shapes, num walkthrough escrito — entendimento construído *ao longo*
do caminho, sem custar dias de reimplementação. Ver `docs/WALKTHROUGH-MODELO.md`
(a escrever no Marco 1).

### 3. E se o resultado for negativo?
**Resposta: seguir com o resultado negativo — mas não desistir no primeiro
número ruim. Persistir e demonstrar que foi genuinamente tentado.**
→ Consequência: precisamos definir **antes** o que conta como "tentativa
genuína", senão "persistir" vira tempo indefinido às vésperas do prazo. Proposta
de contrato explícito (vira o cartão `0002`): um número fixo de tentativas de
melhoria, cada uma com hipótese declarada antes de rodar, cada uma virando linha
do log de experimentos — funcione ou não. O relatório mostra a sequência inteira.
Isso transforma "tentei bastante" em evidência auditável.

### 4. GitHub público ou privado?
**Resposta: público.**
→ Consequências: (a) material de terceiros fica **fora** do repositório — slides,
notebooks das aulas, enunciado e o texto do feedback do professor entraram no
`.gitignore`, porque republicá-los seria redistribuir obra de outra pessoa;
(b) os commits **não levam coautoria de IA** — pedido explícito seu, registrado
aqui para não se perder.

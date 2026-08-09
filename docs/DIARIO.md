# Diário de trabalho

Ordem cronológica do que foi feito, o que ficou pendente e o que travou. É a
memória do projeto — e boa parte da seção "Lições Aprendidas" do relatório sai
daqui.

---

## 2026-08-09 — Sessão 1: leitura completa e montagem do método

### Feito

- **Organização do repositório.** Material das aulas movido para `aulas/aulaNN/`
  (9 aulas, 24 arquivos). Proposta enviada confirmada **byte a byte idêntica** à
  versão do repositório — não havia divergência de versão.
- **Leitura completa:** proposta enviada, feedback do professor, os 10 scripts da
  PoC, os dados baixados e os 14 notebooks das 9 aulas.
- **Documentação de método criada:** `METODOLOGIA.md`, `ESTADO_ATUAL.md`,
  `PLANO.md`, `GLOSSARIO.md`, `PERGUNTAS.md`, `decisoes/` e `experimentos/`.
- **Primeiro cartão de decisão** escrito: `0001-metrica-de-decisao.md`.
- **EXP-000 disparado** — linha de base da PoC na base completa de 99.746 chutes.

### Descobertas que mudam o projeto

1. **A Copa do Mundo de 2026 não existe na StatsBomb Open Data.** Consultei o
   `competitions.json` hoje: a competição mais recente é a Eurocopa Feminina de
   2025, e as Copas vão até 2022. A proposta promete a Copa de 2026 como estudo de
   caso qualitativo, e o professor comentou o ponto favoravelmente. **Precisa de
   decisão sua** — ver `PLANO.md`.
2. **Os dados completos já estavam baixados** — 99.746 chutes, 3.961 partidas.
   O "escalar ~10×" prometido na proposta já está feito.
3. **A proposta promete teste estatístico (bootstrap/McNemar) que o código não
   tem**, e fala em "múltiplas seeds" onde o código roda 2.
4. **Composição dos dados é desbalanceada por origem:** 30,7% das finalizações
   vêm de competições femininas e cerca de 37% de apenas quatro temporadas
   2015/16. Precisa ser declarado no relatório.
5. **A média é de 13 jogadores visíveis por cena**, não 22 — a máscara de padding
   é o caso comum, não a exceção.
6. **`matplotlib` não está instalado** no ambiente. Sem ele não há nenhuma figura
   para o relatório.
7. **O projeto não está sob controle de versão** — não existe `.git`, apesar de a
   proposta afirmar que está no GitHub.

### Correção feita

O `aulas/README.md` que eu havia escrito antes afirmava que o projeto usa
"exatamente o encoder da aula 06 sem positional encoding". A leitura detalhada
mostrou que **a aula 06 nunca diz que o PE é opcional** — apresenta como
necessário. Corrigi o texto para deixar claro que remover o PE é decisão nossa,
com justificativa própria. Se essa frase tivesse ido para o relatório como
"conforme a aula 06", seria uma citação falsa.

### Decisões do Vinícius (mesma data)

As 4 perguntas foram respondidas — íntegra em `PERGUNTAS.md`. Em resumo: trabalho
sólido dentro do prazo; **não reescrever o modelo** (seguir com o `forward` de
atenção mascarada existente); aceitar resultado negativo, mas só depois de
tentativas genuínas e documentadas; **GitHub público**, sem coautoria de IA nos
commits.

### Feito depois disso

- `matplotlib` 3.11.1 instalado no `poc/.venv`.
- Repositório git inicializado com `.gitignore`; primeiro commit feito.
- **Material de terceiros excluído do repositório público:** `aulas/`,
  `project-assignment.pdf` e `feedback_proposta_entrega1.md`. São obra do
  professor; publicá-los seria redistribuir material de outra pessoa. Continuam
  no disco, apenas fora do git. Reversível removendo as linhas do `.gitignore`.

### Pendente para a próxima sessão

- Você criar o repositório público no GitHub e dar `push` (precisa da sua conta).
- Você decidir o cartão `0001` (métrica de decisão).
- Você decidir o que fazer com a Copa de 2026.
- Eu escrever o cartão `0002` (contrato de persistência) e o walkthrough do modelo.

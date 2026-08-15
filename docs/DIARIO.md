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

### EXP-000 concluído (mesma data)

Resultado completo em `experimentos/RESULTS.md`. Em uma frase: **em escala, os
modelos neurais superam o baseline manual — mas a atenção acrescenta quase nada
sobre Deep Sets.**

| Modelo | AUC | Log loss |
|---|---|---|
| B1 | 0,7653 | 0,2887 |
| B2 | 0,7961 | 0,2739 |
| Deep Sets | 0,8144 | 0,2659 |
| Transformer | 0,8158 | 0,2650 |

Isso reposiciona o projeto. O medo registrado na proposta ("os neurais ficam em
~0,71, abaixo do B2") **desapareceu com a escala** — era falta de dado, não de
arquitetura. Em compensação, a pergunta central ficou mais afiada e mais difícil:
o ganho de +0,0183 do B2 para Deep Sets mostra que a representação por token vale
muito, enquanto os +0,0014 de Deep Sets para Transformer sugerem que a **atenção
par a par** pode não estar acrescentando nada. Sem teste estatístico, porém, esse
número não sustenta afirmação nenhuma.

### Pendente para a próxima sessão

- Você criar o repositório público no GitHub e dar `push` (precisa da sua conta).
- Você decidir o cartão `0001` (métrica de decisão).
- Você decidir o que fazer com a Copa de 2026.
- Eu escrever o cartão `0002` (contrato de persistência) e o walkthrough do modelo.

---

## 2026-08-14 — Sessão final: fechamento do EXP-013 e do relatório

### O que aconteceu

**EXP-013 terminou e afastou a suspeita.** Remover a ReLU terminal do Deep Sets
não o aproximou do Transformer — piorou-o, com 3 de 3 sementes na mesma direção.
Era o experimento com maior potencial de derrubar a conclusão central; a
conclusão sobreviveu a um teste que poderia tê-la matado, que é a única forma
honesta de uma conclusão ficar mais forte.

**Um defeito do EXP-013 foi corrigido antes de o número entrar no relatório.** A
primeira execução comparava uma combinação de 3 sementes contra as de 5 do
EXP-004; parte da penalidade seria só a contagem de sementes. As combinações de
referência foram refeitas com as mesmas 3. A diferença passou de +0,00015 para
+0,00012 — pequena, mas o número anterior estava medindo duas coisas.

**EXP-012 e EXP-013 foram incorporados ao `artigo.tex`.** O parágrafo que dizia
que o *hold-out* por competição "não foi conduzido" virou a
Seção~*Generalização para uma competição não vista*. Resumo e *abstract* ganharam
a frase correspondente.

### Erro de ferramenta encontrado hoje

O estimador de páginas subestimava o texto: o regex `%.*` que remove comentários
também truncava a linha em cada `\%` — e o texto está cheio de `9,31\,\%`. Tudo
depois do primeiro `\%` da linha sumia da contagem. As estimativas anteriores
(14,1 páginas) estavam otimistas. Com `(?<!\)%` a estimativa foi para 15,2, e
foi preciso cortar de verdade.

Duas figuras saíram por redundância com as tabelas que já traziam os mesmos
números com mais precisão: a calibração agregada por partida (EXP-006) e a
distribuição *bootstrap* (EXP-004). Ambas continuam em
`docs/experimentos/figuras/`. Estimativa final: **14,8 páginas** — apertado, e só
o Overleaf decide.

### Ferramentas promovidas a versionadas

`checa_tex.py`, `checa_bib.py` e `paginas.py` viviam apenas no diretório temporário
da sessão. São o que garante que o `.tex` compila, e sumiriam com a limpeza da
máquina. Viraram `poc/checa_relatorio.py`, versionado, com três checagens novas:
figuras referenciadas que não existem em disco, o falso positivo de `latin1` (a
palavra aparecia no comentário que proíbe reinseri-la) e saída com código de erro.

O `valida_numeros.py` passou a cobrir EXP-012 e EXP-013, e aprendeu a reconhecer
a forma `0{,}251` que o LaTeX exige em modo matemático — sem isso, todo valor-*p*
do texto era reportado como divergência.

### Pendente — só você pode fazer

**Compilar no Overleaf** (`xg-set-transformer-relatorio.zip`, em `Downloads`):
pdfLaTeX → BibTeX → pdfLaTeX ×2. Se passar de 15 páginas, o primeiro corte
sugerido é a figura de calibração do EXP-000.

### Fecho do dia — compilação

Primeira compilação: **16 páginas**, uma acima do limite. A causa raiz não foi o
texto, foi a ferramenta: o estimador dava 14,9 para esse mesmo documento. Ele
soma texto, figuras e tabelas, mas ignora o branco que o LaTeX deixa ao empurrar
*float* para a página seguinte. Corrigido com a folga de 1,1 página medida contra
o PDF real — sem isso o script continuaria aprovando artigos que estouram.

Saíram três figuras, todas pelo mesmo critério já usado antes: **reproduziam uma
tabela que trazia os mesmos números com mais precisão** (escada → Tabela 2,
atenção por papel → Tabela 7, curvas de calibração → Tabelas 2 e 3). Ficaram as
duas que nenhuma tabela substitui: a arquitetura e os três casos da Eurocopa.

Achado de brinde na leitura do PDF: uma **frase duplicada** na Seção 3.6, sobra de
edição anterior. Só apareceu porque o PDF foi lido inteiro — nenhum dos
verificadores automáticos pega repetição de conteúdo.

Segunda compilação: **15 páginas**, no limite. Figuras e tabelas renumeradas sem
referência quebrada, nenhum `??`, bibliografia completa. `relatorio/artigo.pdf` é
a versão entregue.

# Relatório final — fonte LaTeX (template SBC)

Esta pasta contém o artigo no formato exigido pela disciplina: template SBC,
até 15 páginas, com link para o repositório.

```
artigo.tex        o texto do relatório
referencias.bib   bibliografia — só entra o que foi conferido
sbc-template.sty  template oficial da SBC
sbc.bst           estilo de bibliografia da SBC
figuras/          figuras dos experimentos (cópia de docs/experimentos/figuras/)
```

## Como compilar

**Não há LaTeX instalado nesta máquina.** As duas opções, em ordem de esforço:

### Overleaf (recomendado — sem instalação)

1. Acesse overleaf.com e crie um projeto em branco.
2. Faça upload de **todo o conteúdo desta pasta**, mantendo a subpasta `figuras/`.
3. Em *Menu → Compiler*, selecione **pdfLaTeX**.
4. Em *Menu → Main document*, selecione `artigo.tex`.
5. Compile. Para que as citações apareçam, compile na sequência
   pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX (o Overleaf costuma fazer isso
   sozinho ao clicar em *Recompile*).

### MiKTeX local

Instalar o MiKTeX (~2 GB de download) e rodar na pasta:

```bash
pdflatex artigo && bibtex artigo && pdflatex artigo && pdflatex artigo
```

## Uma armadilha do template oficial

O `sbc-template.tex` distribuído pela SBC traz **duas** linhas de `inputenc`:

```latex
\usepackage[utf8]{inputenc}
\usepackage[latin1]{inputenc}   % <- esta vence, e quebra todos os acentos
```

Em LaTeX a última declaração prevalece, então o arquivo original interpreta o
texto como latin1 — e todo acento em português sai corrompido. O `artigo.tex`
desta pasta mantém **apenas `utf8`**. Se você copiar trechos do template
original, não traga a linha do latin1 junto.

## Sincronizando as figuras

As figuras são geradas pelos experimentos em `docs/experimentos/figuras/`. Para
atualizar as cópias daqui depois de rodar um experimento novo:

```bash
cp docs/experimentos/figuras/*.png relatorio/figuras/
```

## Estado do texto

Todas as seções escritas. O texto passou por revisão crítica externa em 09/08/2026;
as correções apontadas estão aplicadas. Ver `docs/experimentos/RESULTS.md`.
Não há `TODO` pendente no `artigo.tex`. As lacunas conhecidas que permanecem
são experimentais, não de redação, e estão declaradas no próprio texto: falta o
controle de capacidade (um Deep Sets com ~39 mil parâmetros) e a ablação causal
da atenção.

# -*- coding: utf-8 -*-
r"""
Sanidade do relatorio antes de subir ao Overleaf: estrutura, bibliografia e
estimativa de paginas.

Complementa o valida_numeros.py, que confere os NUMEROS. Este aqui confere o
que quebraria a COMPILACAO ou a formatacao — e que so apareceria depois de
subir o projeto e esperar o build.

Uso:  python checa_relatorio.py
"""
import collections
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(RAIZ, "relatorio")
tex = open(os.path.join(REL, "artigo.tex"), encoding="utf-8").read()
bib = open(os.path.join(REL, "referencias.bib"), encoding="utf-8").read()

# Comentarios nao chegam ao PDF; varias checagens abaixo veriam falso positivo
# neles (o proprio aviso sobre latin1 contem a palavra "latin1").
sem_comentario = re.sub(r"(?<!\\)%.*", "", tex)

problemas = []

# ------------------------------------------------------------- estrutura ----
ab = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", sem_comentario))
fe = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", sem_comentario))
ruim = [k for k in set(ab) | set(fe) if ab[k] != fe[k]]
print("ambientes desbalanceados:", ruim or "nenhum")
if ruim:
    problemas.append(f"ambientes desbalanceados: {ruim}")

abre, fecha = sem_comentario.count("{"), sem_comentario.count("}")
print(f"chaves: {abre} abrem, {fecha} fecham ->",
      "OK" if abre == fecha else "DESBALANCEADO")
if abre != fecha:
    problemas.append("chaves desbalanceadas")

print(f"tabelas: {ab['table']} | figuras: {ab['figure']}")

# --------------------------------------------------- referencias cruzadas ---
rot = set(re.findall(r"\\label\{([^}]+)\}", sem_comentario))
ref = set(re.findall(r"\\ref\{([^}]+)\}", sem_comentario))
print("labels sem ref (inofensivo):", sorted(rot - ref) or "nenhum")
print("refs sem label (QUEBRA):", sorted(ref - rot) or "nenhuma")
if ref - rot:
    problemas.append(f"refs sem label: {sorted(ref - rot)}")

# arquivos de figura precisam existir: o Overleaf falha calado no PDF final
faltando = [f for f in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}",
                                  sem_comentario)
            if not os.path.exists(os.path.join(REL, f))
            and not os.path.exists(os.path.join(REL, f + ".png"))]
print("figuras ausentes em disco:", faltando or "nenhuma")
if faltando:
    problemas.append(f"figuras ausentes: {faltando}")

# ---------------------------------------------------------- bibliografia ----
chaves = set(re.findall(r"@\w+\{([^,]+),", bib))
usadas = set()
for g in re.findall(r"\\cite\{([^}]+)\}", sem_comentario):
    usadas.update(x.strip() for x in g.split(","))
print("no .bib nao citadas (nao entram no PDF):", sorted(chaves - usadas) or "nenhuma")
print("citadas sem entrada (QUEBRA a bibliografia):", sorted(usadas - chaves) or "nenhuma")
if usadas - chaves:
    problemas.append(f"citadas sem entrada: {sorted(usadas - chaves)}")

# --------------------------------------------------------------- encoding ---
# O template original da SBC declara inputenc duas vezes, utf8 e depois latin1.
# A ultima vence e todos os acentos quebram. So conta fora de comentario.
if re.search(r"\\usepackage\[[^]]*latin1[^]]*\]\{inputenc\}", sem_comentario):
    print("inputenc latin1: PRESENTE (quebra todos os acentos)")
    problemas.append("inputenc latin1 reintroduzido")
else:
    print("inputenc latin1: ausente (correto)")

# ------------------------------------------------------ estimativa de pag ---
corpo = sem_comentario.split(r"\begin{document}")[1]
for amb in ("table", "figure", "tabular"):
    corpo = re.sub(r"\\begin\{" + amb + r"\}.*?\\end\{" + amb + r"\}", "",
                   corpo, flags=re.S)
palavras = len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", corpo))

# A altura de uma figura escala com a largura pedida: 0,7\textwidth ocupa bem
# menos pagina que \textwidth.
larguras = [float(x) if x else 1.0 for x in
            re.findall(r"includegraphics\[width=([0-9.]*)\\?textwidth", sem_comentario)]
texto = palavras / 450.0
figuras = sum(0.42 * w for w in larguras)
tabelas = ab["table"] * 0.18
bruto = texto + figuras + tabelas + 1.0 + 0.8   # + titulo/resumos + referencias

# Calibracao medida contra o PDF real (Overleaf, 14/08/2026): a soma acima deu
# 14,9 para um PDF de 16 paginas. A conta ignora o espaco em branco que o LaTeX
# deixa ao empurrar floats para a proxima pagina e as quebras de secao, e por
# isso subestima. Sem esta correcao o script aprova um artigo que estoura.
FOLGA = 1.1
total = bruto + FOLGA

print(f"\nestimativa de paginas (SBC 12pt A4, ~450 palavras/pagina):")
print(f"  texto {texto:5.1f} | figuras {figuras:4.1f} | tabelas {tabelas:4.1f} "
      f"| titulo+resumos 1.0 | referencias 0.8 | folga de diagramacao {FOLGA}")
print(f"  TOTAL {total:5.1f}  (limite 15)  -> "
      + ("folga confortavel" if total < 14 else
         "apertado, conferir no Overleaf" if total <= 15 else "PASSOU DO LIMITE"))
if total > 15:
    problemas.append(f"estimativa de {total:.1f} paginas acima do limite")

print(f"\nproblemas que quebram a entrega: {len(problemas)}")
for p in problemas:
    print("  !", p)
sys.exit(1 if problemas else 0)

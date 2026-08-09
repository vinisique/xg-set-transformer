# -*- coding: utf-8 -*-
"""Gera proposta_entrega1.pdf (1 pagina) para a Entrega 1."""
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate

st_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13.5,
                          leading=17, alignment=TA_CENTER, spaceAfter=4)
st_auth = ParagraphStyle("a", fontName="Helvetica", fontSize=10.5,
                         leading=13, alignment=TA_CENTER, spaceAfter=14)
st_head = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11.5,
                         leading=14, spaceBefore=8, spaceAfter=3)
st_body = ParagraphStyle("b", fontName="Helvetica", fontSize=10.3,
                         leading=13.6, alignment=TA_JUSTIFY)

TITULO = ("Estimativa de Expected Goals (xG) com Transformers a partir das "
          "posições dos jogadores no instante da finalização")
AUTOR = "Vinícius Siqueira · Proposta de Projeto · Tópicos Especiais em Inteligência Artificial"

PROBLEMA = """
O projeto aborda a estimativa de <b>Expected Goals (xG)</b>: prever a probabilidade de uma
finalização terminar em gol a partir da posição real de todos os jogadores no instante do chute.
O xG é hoje a métrica central da análise de futebol profissional, usada por clubes, mídia e
departamentos de scouting para avaliar desempenho além do placar. Os modelos clássicos, porém,
utilizam apenas atributos do chute (distância, ângulo, parte do corpo) e, quando incorporam os
demais jogadores, dependem de <i>features</i> desenhadas à mão, como a distância do goleiro e o
número de defensores na linha do chute. A pergunta de pesquisa é: <b>um Transformer que trata
cada jogador como um token aprende automaticamente a geometria de interação</b> (cobertura do
goleiro, bloqueio de defensores, pressão sobre o chutador) <b>que hoje exige engenharia manual
de atributos?</b> Os mapas de atenção ainda oferecem interpretabilidade sobre o que o modelo
considera decisivo em uma finalização, e a Copa do Mundo de 2026, em andamento, fornece um
estudo de caso natural para análise qualitativa do modelo treinado.
"""

DADOS = """
Será utilizada a <b>StatsBomb Open Data</b>, base pública mantida pela empresa StatsBomb no
GitHub (github.com/statsbomb/open-data), com cerca de 3,9 mil partidas anotadas evento a
evento, incluindo as Copas do Mundo de 2018 e 2022, as Eurocopas de 2020 e 2024, a Copa
América de 2024, o Campeonato Espanhol de 2004 a 2021 e o Campeonato Inglês de 2015/16.
Para cada finalização, a base registra o <i>freeze-frame</i>: as coordenadas (x, y) de todos
os jogadores visíveis no momento do chute, com seus papéis (companheiro, adversário, goleiro)
e o desfecho do lance. No total são aproximadamente <b>100 mil finalizações</b>, das quais
cerca de 10% terminam em gol. Os dados serão obtidos por download direto do repositório
público, em formato JSON, respeitando a licença de uso não comercial com a devida atribuição
da fonte.
"""

ABORDAGEM = """
Cada finalização será representada como um <b>conjunto de até 22 tokens</b> (chutador e
jogadores do <i>freeze-frame</i>), cada token contendo geometria absoluta e relativa ao
chutador e à linha do chute (projeção, distância perpendicular, presença no triângulo do gol),
além de indicadores de papel. Os tokens alimentam um <b>Transformer encoder</b> compacto
(2 camadas, 4 cabeças, token [CLS], <b>sem positional encoding</b>, já que a cena é um conjunto
invariante a permutações), com saída sigmoide para a probabilidade de gol. No pré-processamento,
pênaltis serão excluídos e a divisão treino/validação/teste será feita <b>por partida</b>,
evitando vazamento entre chutes correlacionados do mesmo jogo. O Transformer será comparado com <i>baselines</i> de
complexidade crescente: (i) regressão logística com distância e ângulo, o xG clássico;
(ii) a mesma regressão acrescida de <i>features</i> manuais de interação (posição do goleiro,
defensores na linha do chute); e (iii) uma rede neural densa (MLP) treinada sobre esses mesmos
atributos agregados. As métricas serão AUC e <i>log loss</i>, com múltiplas execuções usando
sementes diferentes para verificar a estabilidade dos resultados. Completam o plano uma
análise qualitativa dos <b>mapas de atenção</b> (o modelo identifica o goleiro e os
bloqueadores sem recebê-los como atributo?) e a documentação de experimentos que não
funcionarem como parte da análise crítica, com implementação em PyTorch e código versionado
no GitHub desde o início do projeto.
"""

doc = SimpleDocTemplate("proposta_entrega1.pdf", pagesize=A4,
                        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                        topMargin=1.9 * cm, bottomMargin=1.9 * cm,
                        title="Proposta de Projeto — Entrega 1",
                        author="Vinícius Siqueira")
doc.build([
    Paragraph(TITULO, st_title),
    Paragraph(AUTOR, st_auth),
    Paragraph("Problema", st_head), Paragraph(PROBLEMA, st_body),
    Paragraph("Conjunto de dados", st_head), Paragraph(DADOS, st_body),
    Paragraph("Abordagem proposta", st_head), Paragraph(ABORDAGEM, st_body),
])
print("proposta_entrega1.pdf gerado")

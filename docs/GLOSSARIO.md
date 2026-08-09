# Glossário do projeto

Todo termo técnico que aparece no projeto, explicado no nível de quem está
aprendendo. **Se eu usar um termo que não está aqui, é falha minha — me cobre.**

Ordem: primeiro os termos do domínio (futebol/xG), depois os de modelagem.

---

## Domínio — futebol e xG

**xG (Expected Goals)**
A probabilidade de uma finalização terminar em gol. É um número entre 0 e 1: um
chute com xG de 0,25 significa "chutes como este viram gol 25% das vezes".
Consequência importante: **xG é uma probabilidade, não uma nota**. Por isso não
basta o modelo ordenar bem os chutes — ele precisa acertar o *valor*. Isso é
calibração (ver abaixo), e é o centro do feedback do professor.

**Freeze-frame**
A "fotografia" do posicionamento no instante exato do chute. A StatsBomb fornece,
para cada finalização, as coordenadas (x, y) de todos os jogadores visíveis, com
o papel de cada um: companheiro, adversário, goleiro. É a matéria-prima do
projeto — é o que permite olhar para a cena inteira em vez de só para o chute.

**Campo em coordenadas StatsBomb**
O campo vai de x=0 a x=120 e de y=0 a y=80. O gol atacado fica em x=120, com os
postes em y=36 e y=44 (8 unidades de largura). Toda a geometria do projeto —
distância, ângulo, "está na linha do chute?" — sai dessas convenções.

**Ângulo de chute**
A abertura angular da baliza vista do ponto do chute. Junto com a distância, é a
dupla que explica a maior parte do xG clássico: chute de frente e perto tem
ângulo grande; chute da linha de fundo tem ângulo quase zero mesmo estando perto.

**Bloqueador**
Adversário que está dentro do triângulo formado pelo chutador e os dois postes —
ou seja, no caminho da bola. Contar bloqueadores é uma *feature manual* clássica,
e é justamente o tipo de coisa que queremos descobrir se o Transformer aprende
sozinho.

---

## Modelagem

**Token**
Uma unidade de entrada do modelo. Em NLP (aula 4), um token é uma palavra. Aqui,
**um token é um jogador**: a cena do chute vira um conjunto de até 22 tokens, cada
um com suas coordenadas e papel. É a ideia central do projeto.

**Conjunto vs. sequência (permutation invariance)**
Numa frase, a ordem importa: "o cachorro mordeu o homem" ≠ "o homem mordeu o
cachorro". Numa cena de futebol, **a ordem em que listamos os jogadores não
significa nada** — é um conjunto, não uma sequência. Por isso o nosso modelo
*não* usa positional encoding: embaralhar a lista de jogadores tem de produzir
exatamente a mesma previsão. Essa propriedade se chama invariância à permutação.

> Detalhe bonito de citar no relatório: a aula 4 mostra a invariância à ordem
> como um **defeito** (o modelo não distingue quem mordeu quem). No nosso
> problema ela é exatamente a **propriedade desejada**. Mesmo fenômeno, sinal
> trocado, porque o dado é conjunto e não sequência.

**Self-attention (autoatenção)**
O mecanismo que deixa cada token "olhar" para todos os outros e decidir de quem
puxar informação. É o que permite ao modelo aprender relações **par a par** —
"este defensor está entre o chutador e o gol", "o goleiro está adiantado em
relação a este chute". É a diferença essencial entre o Transformer e o Deep Sets.

**Deep Sets**
Modelo que processa cada token isoladamente e depois agrega tudo (média, máximo).
Ele enxerga *informação por jogador*, mas **nunca compara dois jogadores entre
si**. Por isso é o baseline mais importante do projeto: a diferença
Transformer − Deep Sets isola exatamente o valor da interação par a par. Se o
Transformer não superar o Deep Sets, a atenção não está agregando nada.

**Token [CLS]**
Um token extra, artificial, que não corresponde a nenhum jogador. Ele participa
da atenção, "coleta" informação de todos os outros, e é a partir dele que fazemos
a previsão final. É a alternativa a agregar por média — vem do BERT (aula 7).

**Máscara de padding (padding mask)**
Nem toda cena tem 22 jogadores visíveis: **a média no nosso dado é 13, e há cenas
com apenas 1**. Como o tensor precisa ter tamanho fixo, sobram posições vazias. A
máscara diz ao modelo "ignore estas posições" — sem ela, o modelo aprende a
prestar atenção em jogadores que não existem. A aula 4 usa o mesmo conceito no
pooling mascarado, e a aula 5 no `padding_idx`.

**Log loss (entropia cruzada binária)**
Mede o quão erradas estão as *probabilidades*. Punir forte a confiança errada é o
ponto: dizer "95% de gol" num chute que não foi gol custa muito mais que dizer
"55%". É a função de perda que treinamos.

**AUC (área sob a curva ROC)**
Mede só a **ordenação**: a probabilidade de o modelo dar nota maior a um gol do
que a um não-gol. 0,5 é aleatório, 1,0 é perfeito. Limitação crucial: **a AUC não
enxerga calibração**. Um modelo que multiplica todas as previsões por 10 tem a
mesma AUC e um xG completamente inútil.

**Brier score**
A média do erro quadrático entre a probabilidade prevista e o resultado (0 ou 1).
Ao contrário da AUC, ele **penaliza probabilidade mal calibrada**. É uma das
métricas que o professor pediu explicitamente.

**Calibração**
Um modelo é calibrado quando, entre todos os chutes a que ele deu 20%, cerca de
20% viraram gol de fato. É a propriedade que torna o xG utilizável: sem ela o
número não é uma probabilidade, é só um ranking.
- **Curva de calibração (reliability diagram):** o gráfico que mostra isso —
  previsto no eixo x, observado no eixo y, diagonal = perfeito.
- **Platt scaling:** recalibra ajustando uma regressão logística sobre a saída do
  modelo. Simples, assume formato sigmoide.
- **Regressão isotônica:** recalibra com uma função apenas monótona, sem supor
  formato. Mais flexível, precisa de mais dados para não sobreajustar.
- **Calibração agregada:** somar o xG previsto de uma partida e comparar com os
  gols que realmente saíram. É o teste que um analista de futebol faria.

**Split por partida**
Dividir treino/validação/teste **por jogo**, não por chute. Dois chutes da mesma
partida compartilham time, adversário, clima e árbitro — se um cair no treino e o
outro no teste, o modelo "já viu" o contexto e a métrica sai inflada. Isso é
vazamento (*leakage*), e evitá-lo foi um dos pontos elogiados na proposta.

**Evento raro / desbalanceamento**
Só **10,3%** das finalizações viram gol. Consequências: acurácia é inútil (chutar
sempre "não é gol" acerta 90%), e o modelo tende a ser conservador. Por isso
usamos AUC, log loss, Brier e calibração — nunca acurácia.

**Seed**
O número que inicializa a aleatoriedade (pesos iniciais, embaralhamento). Fixar a
seed torna o resultado reproduzível; rodar com **várias** seeds mostra se a
diferença entre dois modelos é real ou apenas ruído de inicialização.

**Ablação**
Remover deliberadamente um componente para medir o quanto ele contribuía. Nossa
escada de baselines (B1 → B2 → Deep Sets → Transformer) é uma ablação: cada
degrau acrescenta exatamente uma capacidade, então a diferença entre degraus
atribui o mérito a um componente específico.

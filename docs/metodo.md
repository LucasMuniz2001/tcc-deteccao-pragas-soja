# Método

Detalhamento das decisões de preparação, treinamento e avaliação. O resumo está no
[README](../README.md).

---

## 1. Conversão Pascal VOC → YOLO

O formato Pascal VOC descreve cada caixa pelos valores absolutos de suas bordas,
em pixels. O formato exigido pela biblioteca de treinamento descreve a mesma caixa
por quatro números normalizados entre 0 e 1:

```
x_c = (x_min + x_max) / 2W        y_c = (y_min + y_max) / 2H
w   = (x_max − x_min) / W         h   = (y_max − y_min) / H
```

onde `W` e `H` são a largura e a altura da imagem. O índice da classe vem da lista
de classes, mantida na ordem original para que os resultados permaneçam comparáveis
aos da literatura.

Além da conversão, o procedimento de regeneração normaliza a extensão dos arquivos
de imagem, descarta caixas degeneradas, recorta ao retângulo da imagem eventuais
coordenadas fora dos limites, e grava um relatório com todas as ocorrências.

---

## 2. Partição em dobras estratificadas

A avaliação usa validação cruzada estratificada em **cinco dobras** — protocolo
equivalente ao adotado por Tetila et al. (2024) sobre a mesma base, o que permite
comparação direta.

Cada uma das cinco partes serve uma vez como conjunto de validação enquanto as
outras quatro formam o de treino; o resultado final é a média das cinco execuções.
A vantagem sobre uma divisão única é que todas as imagens são usadas para validação
exatamente uma vez, o que reduz a variância da estimativa — crítico em classes
representadas por poucas dezenas de exemplares.

### Por que não é uma estratificação simples

424 das 1.800 imagens contêm organismos de **mais de uma classe**. Uma imagem não
pertence a um rótulo só, e não existe divisão que equilibre todas as classes
simultaneamente. O problema é de estratificação multirrótulo.

Adota-se o algoritmo de **estratificação iterativa** (Sechidis, Tsoumakas e
Vlahavas, 2011): processam-se primeiro as classes mais raras e, a cada passo,
aloca-se a imagem na dobra que mais precisa daquela classe. Isso garante que as 23
anotações de *Nezara viridula* ninfa, por exemplo, sejam distribuídas entre as cinco
dobras em vez de se concentrarem em uma delas.

O algoritmo opera sobre a **presença da classe na imagem**, e não sobre a contagem
de caixas. A distinção tem efeito prático: equilibrar as contagens de caixas deixa
livre a quantidade de imagens por dobra, e a primeira execução feita dessa forma
produziu dobras de 338 a 376 imagens — dispersão que a validação cruzada não
pressupõe. Com a formulação por presença, as dobras ficaram entre 355 e 365
imagens, e as contagens de caixas permaneceram igualmente equilibradas.

### Resultado da execução

Implementada em
[`scripts/02_particiona_dobras.py`](../scripts/02_particiona_dobras.py).

| Dobra | Imagens de validação | Imagens de treino | Caixas de *N. viridula* ninfa |
|---|---|---|---|
| 1 | 359 | 1.441 | 5 |
| 2 | 365 | 1.435 | 5 |
| 3 | 361 | 1.439 | 4 |
| 4 | 355 | 1.445 | 5 |
| 5 | 360 | 1.440 | 4 |

A partição é determinística: a semente é fixa e gravada, junto da atribuição
completa de cada imagem, em `particao.json`. Duas execuções produzem arquivos
idênticos — verificado.

### Três salvaguardas

1. **Partição no nível da imagem, não da caixa.** Todos os organismos de uma imagem
   permanecem na mesma dobra. Sem isso, seria possível validar o modelo em um objeto
   cuja imagem de origem foi usada no treino.

2. **Quase-duplicatas agrupadas.** Imagens quase idênticas produziriam vazamento
   mesmo com a partição por imagem. Calculou-se o resumo perceptual de diferença
   (dHash) das 1.800 imagens e agruparam-se as que ficaram a uma distância de
   Hamming de até 10 bits: **6 grupos, envolvendo 13 imagens**, mantidos na mesma
   dobra.

3. **Partição gravada antes do primeiro treinamento** e reutilizada por todos os
   modelos, para que qualquer diferença observada seja atribuível ao modelo e não
   à partição.

---

## 3. Aumento de dados

Aplicado exclusivamente às imagens de treino — as de validação são apenas
redimensionadas, porque avaliar sobre imagens transformadas produziria estimativa
que não corresponde ao uso real do modelo.

| Operação | Faixa |
|---|---|
| Matiz, saturação e valor | ±0,015; ±0,7; ±0,4 |
| Espelhamento horizontal | probabilidade 0,5 |
| Espelhamento vertical | probabilidade 0,5 |
| Rotação | ±10° |
| Translação | ±10% da dimensão |
| Escala | ±50% |
| Mosaico de quatro imagens | probabilidade 1,0, desativado nas 10 épocas finais |
| Recorte e colagem, mistura de imagens | não utilizados |

Nenhuma dessas operações corrige o desbalanceamento: todas as classes se
transformam na mesma proporção. O desbalanceamento é tratado no protocolo de
avaliação, reportando o desempenho por classe.

A escolha das faixas respeita o tamanho dos alvos. Como 69,9% das caixas ocupam
menos de 1% da imagem, evitam-se operações que reduzam demais a escala dos objetos,
e a resolução de entrada é mantida em 640 × 640.

---

## 4. Modelos e hiperparâmetros

Três modelos da mesma família, em tamanhos crescentes. Diferem apenas pelos fatores
que multiplicam a profundidade e a largura da rede; estrutura, função de perda e
procedimento são idênticos. Como tudo mais é mantido constante, a comparação isola
o efeito da capacidade do modelo sobre a acurácia e sobre o custo de inferência.

| Modelo | Parâmetros (milhões) | Operações (bilhões) | mAP@0,5:0,95 em COCO (%) |
|---|---|---|---|
| YOLOv8n | 3,2 | 8,7 | 37,3 |
| YOLOv8s | 11,2 | 28,6 | 44,9 |
| YOLOv8m | 25,9 | 78,9 | 50,2 |

Valores publicados pela Ultralytics, com entrada de 640 × 640. Servem como
referência de porte dos modelos.

### Hiperparâmetros

Idênticos para os três modelos, pela razão acima. Partem dos padrões recomendados
pela biblioteca, com três ajustes deliberados: resolução de entrada mantida em 640
pelo tamanho dos alvos, lote fixado em 16 para acomodar o maior dos três modelos na
memória da GPU disponível, e parada antecipada para que modelos de capacidades
diferentes não sejam avaliados sob um número de épocas arbitrariamente igual.

| Hiperparâmetro | Valor |
|---|---|
| Resolução de entrada | 640 × 640 pixels |
| Épocas | 150 (máximo) |
| Paciência da parada antecipada | 30 épocas |
| Tamanho do lote | 16 |
| Otimizador | SGD com momento 0,937 |
| Taxa de aprendizado inicial e final | 0,01 e 0,0001 (decaimento linear) |
| Aquecimento | 3 épocas |
| Decaimento de pesos | 0,0005 |
| Pesos iniciais | Pré-treinados em COCO |
| Camadas congeladas | Nenhuma |
| Precisão mista | Ativada |
| Sementes | 0, 1 e 2 |
| Limiares na validação | confiança 0,001; sobreposição 0,7 |
| Limiares no protótipo | confiança 0,25; sobreposição 0,45 |

Nenhuma camada é congelada porque o domínio de destino — insetos sobre folhas em
campo — difere consideravelmente do domínio de origem, e o custo de ajustar a rede
inteira é aceitável no volume de dados envolvido.

---

## 5. Protocolo de avaliação

### Agregação

Cada execução produz, sobre a dobra de validação, as métricas no agregado e para
cada uma das doze classes, calculadas pela própria biblioteca — o que elimina
diferenças de implementação na comparação entre modelos.

A agregação entre dobras é a **média aritmética das cinco execuções, acompanhada do
desvio padrão** — e não a soma das contagens das cinco dobras. A média expõe a
dispersão entre partições, que é a informação de interesse nas classes com poucos
exemplares.

Nenhum resultado é apresentado como valor único: toda métrica vem com sua dispersão
e com o número de exemplares sobre o qual foi calculada.

### As três análises que são a contribuição

1. **Desempenho por classe cruzado com o número de exemplares**, em tabela única,
   de modo que se possa julgar a confiabilidade de cada valor. Espera-se que as
   classes com poucas dezenas de anotações apresentem revocação baixa e dispersão
   alta; o objetivo é quantificar esse efeito, não contorná-lo.

2. **Matriz de confusão entre as doze classes**, examinada à procura de três trocas
   previsíveis pela biologia dos organismos: entre adulto e ninfa de uma mesma
   espécie, entre percevejos de espécies distintas, e entre lagartas desfolhadoras.
   São erros de gravidade diferente — confundir o estágio de desenvolvimento tem
   consequência agronômica menor do que confundir espécies com níveis de ação
   distintos.

3. **Praga × organismo que não é praga.** As doze classes são agrupadas por papel
   agronômico e calcula-se separadamente a taxa de confusão entre os dois grupos.
   Classificar uma joaninha como percevejo tem consequência oposta à desejada pelo
   manejo, podendo induzir uma aplicação de defensivo justamente onde há controle
   biológico em curso — erro que a precisão média global não distingue de qualquer
   outro. O grupo dos organismos que não são praga contém apenas Coccinellidae.
   *Lagria villosa* fica entre as pragas: é secundária, mas pode causar dano, e
   tratá-la como inofensiva seria a escolha menos conservadora.

A matriz de confusão é montada com limiar de confiança de 0,25, o mesmo da
predição em uso, e IoU de 0,45 para casar detecção e anotação. As métricas de
precisão média continuam calculadas com o limiar de 0,001 da validação, sem o qual
a curva precisão-revocação fica truncada. A separação é necessária porque, na
versão 8.4.183, a biblioteca monta a matriz com o mesmo limiar das métricas, e a
0,001 quase toda saída da rede seria contada como detecção.

### Tempo de inferência

Medido **uma imagem por vez**, e não em lotes, por ser essa a condição de uso em
uma plataforma de monitoramento. Descartam-se as vinte primeiras inferências,
destinadas ao aquecimento do dispositivo, e cronometram-se as cem seguintes,
relatando-se a mediana e o intervalo interquartil.

O tempo é decomposto em pré-processamento, passagem pela rede e pós-processamento,
porque esta última parcela inclui a supressão não máxima, costuma ser subestimada e
depende dos limiares adotados.

A medição é feita em dois ambientes: na GPU do ambiente de nuvem, condição à qual se
refere a meta de 50 ms por imagem, e na CPU da máquina local com o modelo exportado
para formato de inferência otimizada, correspondente ao cenário embarcado.

---

## 6. Ameaças à validade

Registradas desde o desenho do experimento.

**Vazamento de informação entre as partições** — enfrentado pela partição no nível
da imagem, pelo agrupamento de quase-duplicatas e pela reutilização da mesma
partição por todos os modelos.

**Escolha do modelo com base nas mesmas dobras usadas para reportar o desempenho** —
torna o valor relatado para o modelo selecionado ligeiramente otimista. O efeito é
registrado explicitamente e mitigado pelas sementes adicionais da fase 2.

**Alcance das conclusões** — restringem-se às doze classes anotadas e às condições
de captura da base. As métricas medem o acerto da detecção, não a utilidade
agronômica da informação produzida, e nada se pode afirmar sobre espécies ausentes
do conjunto ou imagens capturadas por equipamentos muito distintos.

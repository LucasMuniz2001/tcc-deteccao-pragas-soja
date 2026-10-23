# Detecção automática de insetos-praga da soja por visão computacional

Código e documentação técnica do Trabalho de Conclusão de Curso em Engenharia da
Computação, Centro Universitário IBMEC – RJ.

**Autor:** Lucas Muniz da Costa Marques
**Orientador:** Prof. PhD. William de Souza Barbosa
**Ano:** 2026

---

## O problema

A soja é a principal cultura agrícola brasileira. O Manejo Integrado de Pragas
depende do monitoramento periódico da lavoura, hoje feito por amostragem manual:
demorado, dependente de mão de obra treinada e sujeito a erro de identificação
entre espécies parecidas — em particular entre percevejos de espécies distintas e
entre os estágios adulto e ninfa de uma mesma espécie.

Este trabalho treina e avalia detectores de objetos da família YOLO sobre o
conjunto público INSECT12C, com o objetivo de localizar e classificar doze
organismos associados à cultura da soja em imagens digitais.

**Metas declaradas:** mAP@0,5 ≥ 0,70 e tempo de inferência < 50 ms por imagem em GPU.

São metas de projeto, não resultados esperados. O trabalho reportará o desempenho
obtido, tenha ele atingido ou não os limiares.

---

## O que é autoral e o que vem de biblioteca

Esta seção existe porque a distinção é legítima e deve ser explícita.

O modelo não é contribuição deste trabalho. A YOLOv8 é usada na implementação da
Ultralytics (Jocher, Chaurasia e Qiu, 2023), sobre PyTorch. Dela vêm a construção
da rede, a função de perda, a atribuição de alvos, a supressão não máxima, o
aumento de dados, o cálculo das métricas e a exportação para ONNX.

**A contribuição está no método de avaliação, não no modelo.** O que é autoral:

| Componente | Descrição |
|---|---|
| Auditoria das anotações | Verificação dos 1.800 XML contra os rótulos distribuídos. Encontrou 6 defeitos reais no conjunto publicado — ver [docs/auditoria.md](docs/auditoria.md). |
| Regeneração dos rótulos | Pipeline que reconstrói os rótulos YOLO a partir do XML, que é a anotação primária, tornando o conjunto internamente consistente e reprodutível. |
| Partição estratificada multirrótulo | Estratificação iterativa em 5 dobras com 424 imagens multiclasse, mais controle de vazamento por quase-duplicatas (dHash). Ver [docs/metodo.md](docs/metodo.md). |
| Análise por classe | Desempenho cruzado com o número de exemplares, sob desbalanceamento de 36:1. |
| Análise praga × inimigo natural | Taxa de confusão entre os dois grupos, calculada separadamente. A métrica agregada trata como equivalente o erro que induziria aplicação de defensivo onde há controle biológico em curso. |
| Custo de inferência | Medição em GPU e em CPU com modelo exportado, decomposta em pré-processamento, rede e pós-processamento. |

Os pseudocódigos dos três procedimentos autorais estão no Apêndice A do trabalho
escrito e são a especificação que os scripts deste repositório implementam.

## Como reproduzir a preparação dos dados

Requer Python 3, Pillow e NumPy. O conjunto INSECT12C deve estar em um diretório
próprio, obtido com os autores. A preparação escreve em diretório novo e **não
altera o conjunto original.**

```
python -P scripts/01_regenera_rotulos.py  <INSECT12C-Dataset>  dados_yolo
python -P scripts/02_particiona_dobras.py  dados_yolo
```

A primeira etapa reconstrói os 1.800 rótulos a partir dos XML e grava
`relatorio_auditoria.csv`. A segunda calcula os resumos perceptuais, agrupa as
quase-duplicatas, estratifica em cinco dobras e grava as listas de treino e
validação, os cinco arquivos `data_dobraN.yaml` e o registro `particao.json`.

A partição é determinística: com a mesma semente, duas execuções produzem a
mesma atribuição. Para gerar os arquivos de configuração já apontando para o
ambiente de treinamento, use `--raiz /content/dados_yolo`.

As listas gravadas em `dobras/` trazem caminhos na forma `./../images/imgN.jpg`.
A forma importa: a Ultralytics resolve o prefixo `./` contra o diretório do
próprio arquivo de lista, e não contra o campo `path` do YAML. Com
`./images/imgN.jpg` ela procuraria as imagens em `dobras/images/`, não as
encontraria e, por tratar caminho inexistente como aviso e não como erro,
iniciaria o treino sobre um conjunto vazio.

## Como executar o treinamento

[`notebooks/treinamento_colab.ipynb`](notebooks/treinamento_colab.ipynb) executa o
procedimento experimental da Seção 2.5.3: a fase 1 compara os três modelos nas
cinco dobras com semente fixa, e a fase 2 repete o modelo escolhido com duas
sementes adicionais.

Envie `dados_yolo` compactado para o Drive, abra o notebook no Colab e **selecione
a GPU T4** — o notebook verifica isso e avisa, porque o capítulo 2 declara a T4
como a condição à qual a meta de 50 ms se refere.

O notebook é idempotente. Ao fim de cada época, os pesos, o estado do otimizador
e as métricas são copiados para o Drive; o treino em si roda no disco local da
máquina virtual, que é rápido. Uma execução interrompida retoma do último ponto
salvo, e uma execução já concluída é reconhecida e saltada. Isso existe porque as
~25 horas de GPU estimadas não cabem em nenhuma sessão única do Colab.

Antes de treinar, o notebook confere que o pacote recebido tem 1.800 imagens,
1.800 rótulos, 12 classes e 2.757 caixas. Se algum número divergir, o pacote não
é o que foi auditado e a execução para ali.

### Saída obtida

| Verificação | Resultado |
|---|---|
| Rótulos regenerados | 1.800 |
| Caixas gravadas | 2.757 — uma caixa degenerada descartada das 2.758 anotadas |
| Ocorrências no relatório | 63 — 6 defeitos de anotação e 57 extensões não canônicas |
| Grupos de quase-duplicatas | 6, envolvendo 13 imagens, mantidos na mesma dobra |
| Imagens por dobra de validação | 359, 365, 361, 355 e 360 |
| Caixas da classe mais rara por dobra | 5, 5, 4, 5 e 4 (*Nezara viridula* ninfa, 23 no total) |

---

## O conjunto de dados

**INSECT12C** (Tetila et al., 2024) — 1.800 imagens, 2.758 organismos anotados,
12 classes. Imagens em JPEG, 600 × 800 ou 800 × 600. Cada imagem tem um XML no
padrão Pascal VOC e um arquivo de texto no formato YOLO.

O conjunto **não é redistribuído aqui.** Este repositório contém o código que o
processa.

Características que condicionam decisões de método:

- **Desbalanceamento de 36:1** — a classe mais frequente tem 836 caixas; a menos
  frequente, 23. As duas classes de *Euschistus heros* respondem por 59,4% das
  anotações.
- **Objetos pequenos** — a mediana da área das caixas é 0,59% da área da imagem;
  69,9% das caixas ocupam menos de 1%.
- **Multirrótulo** — 424 imagens contêm organismos de mais de uma classe.
- **Nem toda classe é praga** — Coccinellidae é inimigo natural e Gastropoda é
  praga ocasional (molusco). Confundi-los com percevejos tem consequência
  agronômica oposta à desejada.

As doze classes:

```
Anticarsia_gemmatalis          Spodoptera_albula
Coccinellidae                  Lagria_villosa
Diabrotica_speciosa            Nezara_viridula_adulto
Edessa_meditabunda             Nezara_viridula_ninfa
Euschistus_heros_adulto        Rhammatocerus_schistocercoides
Euschistus_heros_ninfa         Gastropoda
```

---

## Método

Resumo. O detalhamento está em [docs/metodo.md](docs/metodo.md).

1. **Auditoria** das anotações distribuídas contra os XML.
2. **Regeneração** dos rótulos a partir do XML, com relatório de ocorrências.
3. **Partição** em 5 dobras estratificadas, no nível da imagem, com quase-duplicatas
   mantidas na mesma dobra. Gravada antes do primeiro treinamento e reutilizada por
   todos os modelos.
4. **Treinamento** de YOLOv8 n, s e m por transferência de aprendizado a partir dos
   pesos do COCO, sem congelar camadas.
5. **Avaliação** por validação cruzada em 5 dobras, com métricas agregadas e por
   classe, matriz de confusão e tempo de inferência.
6. **Protótipo** local de demonstração.

O experimento tem duas fases: os três modelos nas cinco dobras com semente fixa
(15 execuções), e depois o modelo escolhido nas cinco dobras com mais duas sementes
(10 execuções). Resultados relatados como média e desvio padrão — nunca como o
melhor valor obtido.

---

## Ambiente

Ver [docs/ambiente.md](docs/ambiente.md).

Preparação dos dados e análise na máquina local; treinamento no Google Colab, em
GPU NVIDIA T4. O orçamento estimado é de cerca de 25 horas de GPU, o que excede uma
única sessão — daí o salvamento de pesos e estado do otimizador ao fim de cada
época, com retomada a partir do último ponto salvo.

---

## Estado atual

| Etapa | Situação |
|---|---|
| Introdução (Capítulo 1) | Escrita |
| Materiais e métodos (Capítulo 2) | Escrito, em revisão pelo professor |
| Auditoria do conjunto de dados | Executada — 6 ocorrências encontradas |
| Análise de quase-duplicatas | Executada — 6 grupos, 13 imagens |
| Regeneração dos rótulos | Implementada e executada — `scripts/01_regenera_rotulos.py` |
| Partição em dobras | Implementada e executada — `scripts/02_particiona_dobras.py` |
| Notebook de treinamento | Implementado — `notebooks/treinamento_colab.ipynb` |
| Treinamento | Não iniciado |
| Resultados (Capítulo 3) | Não iniciado |
| Conclusão (Capítulo 4) | Não iniciado |

Os scripts serão publicados aqui conforme forem implementados. Este repositório
nasceu com a documentação do método porque a especificação precede a implementação,
e não porque o código já existisse.

---

## Referências principais

- JOCHER, G.; CHAURASIA, A.; QIU, J. **Ultralytics YOLOv8.** 2023.
  Disponível em: https://github.com/ultralytics/ultralytics
- TETILA, E. C. et al. **YOLO performance analysis for real-time detection of
  soybean pests.** Smart Agricultural Technology, 2024.
- TETILA, E. C. et al. **Detection and classification of soybean pests using deep
  learning with UAV images.** Computers and Electronics in Agriculture, 2020.
- SECHIDIS, K.; TSOUMAKAS, G.; VLAHAVAS, I. **On the stratification of multi-label
  data.** ECML PKDD, 2011.
- REDMON, J. et al. **You only look once: unified, real-time object detection.**
  CVPR, 2016.

A lista completa está nas referências do trabalho escrito.

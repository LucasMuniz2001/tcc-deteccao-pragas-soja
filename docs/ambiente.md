# Ambiente computacional

O trabalho usa dois ambientes com funções distintas. A separação tem consequência
metodológica: o tempo de inferência é medido nos dois, porque cada um responde a
uma pergunta diferente sobre a viabilidade de uso do modelo.

## Máquina local

Preparação dos dados, análise dos resultados e execução do protótipo.

| Componente | Especificação |
|---|---|
| Processador | AMD Ryzen 7 7730U, 8 núcleos e 16 processadores lógicos |
| Memória | 16 GB |
| Gráficos | Radeon integrado |
| Sistema | Windows 11 |

**Não há GPU NVIDIA.** É o motivo de o treinamento ocorrer em nuvem, e é também o
que define o cenário de medição em CPU: o modelo exportado para ONNX roda aqui, e
esse é o número que corresponde ao embarque em hardware modesto.

## Ambiente de treinamento

Google Colab, com GPU NVIDIA T4 de 16 GB.

O orçamento estimado é de cerca de **25 horas de GPU**:

| Fase | Execuções | Tempo por execução | Tempo total |
|---|---|---|---|
| Fase 1 — YOLOv8n em 5 dobras | 5 | ≈ 30 min | ≈ 2,5 h |
| Fase 1 — YOLOv8s em 5 dobras | 5 | ≈ 55 min | ≈ 4,6 h |
| Fase 1 — YOLOv8m em 5 dobras | 5 | ≈ 105 min | ≈ 8,8 h |
| Fase 2 — modelo escolhido, 2 sementes | 10 | ≈ 55 min | ≈ 9,2 h |
| **Total** | **25** | | **≈ 25 h** |

Estimativas para 1.440 imagens de treino por dobra, lote de 16 e 150 épocas, sem
considerar a parada antecipada.

### Por que isso exige retomada

25 horas excedem qualquer sessão única do Colab. A sessão cai por inatividade, por
limite de uso, ou simplesmente porque o navegador foi fechado.

Por isso o treinamento é organizado em **sessões sucessivas**, com os pesos e o
estado do otimizador salvos ao fim de cada época em armazenamento persistente, e as
execuções retomadas a partir do último ponto salvo. Reconectar custa minutos, não
um treino inteiro.

### Fluxo de uma sessão

1. Montar o Google Drive.
2. Copiar o conjunto preparado do Drive para o disco local da máquina virtual e
   descompactar ali — ler as imagens direto do Drive durante o treino é lento o
   bastante para dominar o tempo de época.
3. Instalar a biblioteca de treinamento.
4. Executar o laço: para cada modelo, para cada dobra, treinar com o arquivo de
   configuração correspondente àquela dobra, retomando se já houver ponto salvo.
5. Gravar pesos, métricas e logs de volta no Drive ao fim de cada época.

O conjunto preparado ocupa aproximadamente 260 MB, enviado uma única vez.

## Ferramentas

| Função | Ferramenta |
|---|---|
| Linguagem | Python 3 |
| Treinamento e inferência | Ultralytics YOLOv8 sobre PyTorch |
| Aumento de dados | Operações nativas da biblioteca, complementadas pelo Albumentations |
| Imagens, análise e gráficos | OpenCV, Pillow, NumPy, scikit-learn, Matplotlib |
| Inferência otimizada | Exportação para ONNX |
| Controle de versão | Git |

Todo o ferramental é de código aberto e sem custo de licença. O único recurso a
contratar é o acesso à GPU em nuvem, porque o orçamento de 25 horas excede o que a
modalidade gratuita oferece de forma previsível.

## Nota sobre licenciamento

A Ultralytics YOLOv8 é distribuída sob AGPL-3.0. Para uso acadêmico e para o
protótipo de demonstração descrito neste trabalho não há restrição. A licença
passaria a importar caso o modelo fosse distribuído dentro de um produto de código
fechado.

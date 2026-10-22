# Auditoria das anotações do INSECT12C

O conjunto INSECT12C distribui, para cada imagem, uma anotação em XML no padrão
Pascal VOC e um arquivo de texto no formato YOLO. Os dois deveriam descrever as
mesmas caixas.

Antes de qualquer treinamento, verificou-se se descrevem.

## Procedimento

O procedimento percorre os 1.800 arquivos XML e, para cada um:

1. extrai as caixas e as classes declaradas;
2. converte cada caixa para o formato YOLO, usando as dimensões declaradas no
   próprio XML;
3. compara o resultado com o arquivo de texto correspondente, admitida tolerância
   de 10⁻⁴ nas coordenadas normalizadas;
4. verifica a existência dos três arquivos de cada amostra;
5. verifica a validade geométrica de cada caixa.

## Resultado

Seis ocorrências. Poucas em termos absolutos, mas duas alterariam a contagem de
objetos usada no treinamento e uma produziria uma caixa de área nula — que a
maioria das implementações descarta silenciosamente, sem avisar.

| Amostra | Divergência | Tratamento |
|---|---|---|
| `img1146` | Possui imagem e XML, mas não possui o rótulo no formato YOLO | Rótulo gerado a partir do XML |
| `img1272` | Possui rótulo adicional com dupla extensão (`img1272.xml.txt`), de conteúdo idêntico ao rótulo correto | Arquivo redundante ignorado |
| `img182` | O XML declara dois organismos e o rótulo contém apenas um | Rótulo regenerado com os dois organismos |
| `img528` | O XML declara um organismo e o rótulo contém dois | Rótulo regenerado com um organismo |
| `img824` | As coordenadas do rótulo divergem das do XML além da tolerância adotada | Rótulo regenerado a partir do XML |
| `img631` | O XML contém uma caixa de largura nula, com abscissa mínima igual à máxima, na borda da imagem | Caixa descartada e ocorrência registrada |

## Decisão

O XML é a anotação primária e o arquivo de texto deriva dele. Adotou-se o XML como
referência e **todos** os rótulos foram regenerados a partir dele — não apenas os
seis divergentes.

A decisão torna o conjunto internamente consistente e, sobretudo, reprodutível: o
estado dos rótulos usados no experimento passa a ser resultado de um procedimento
documentado, e não do arquivo recebido.

## Observação adicional

Além das seis ocorrências, 57 dos 1.800 arquivos de imagem usam extensão `.JPG`
em maiúsculo, contra 1.743 em minúsculo. Não é um defeito de anotação, mas quebra
o pareamento por nome em sistemas sensíveis a maiúsculas — a extensão é normalizada
na cópia de trabalho.

O conjunto original não é alterado em nenhum momento. Toda a preparação escreve em
diretório novo.

## Execução

O procedimento está implementado em
[`scripts/01_regenera_rotulos.py`](../scripts/01_regenera_rotulos.py) e o relatório
é gravado como `relatorio_auditoria.csv` no diretório de saída.

A execução sobre as 1.800 amostras registrou **63 ocorrências**: as 6 divergências
de anotação da tabela acima e as 57 extensões não canônicas. Foram gravados 1.800
rótulos e **2.757 caixas** — uma menos do que as 2.758 anotadas, pela caixa
degenerada de `img631`.

Duas notas sobre a implementação, ambas consequência de detalhes que distorceriam
a contagem se passassem em branco:

**A extensão é consultada na listagem do diretório, não por tentativa de abertura.**
O sistema de arquivos do Windows ignora a caixa das letras, de modo que um teste
por `img1.jpg` encontra `img1.JPG` e as 57 divergências de extensão passariam
despercebidas.

**O descarte de uma caixa inválida não é relatado duas vezes.** A comparação de
contagem usa o número de organismos declarados no XML *antes* do descarte. Sem
isso, `img631` apareceria também como divergência de contagem — o mesmo defeito
contado como dois.

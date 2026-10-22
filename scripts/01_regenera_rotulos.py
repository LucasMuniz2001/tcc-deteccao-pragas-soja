"""Regenera os rotulos YOLO do INSECT12C a partir das anotacoes XML.

O XML no padrao Pascal VOC e a anotacao primaria; o arquivo de texto
distribuido com o conjunto deriva dele. Este procedimento adota o XML como
referencia, reconstroi todos os rotulos, e registra em relatorio toda
divergencia encontrada entre as duas versoes.

O conjunto original nao e alterado. Toda a escrita ocorre no diretorio de
saida, que e criado do zero.

Uso:
    python -I 01_regenera_rotulos.py <dir_origem> <dir_saida>
"""

import csv
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

TOLERANCIA = 1e-4

EXTENSOES = {".jpg", ".jpeg", ".png"}


def indexa_imagens(origem):
    """Mapeia cada amostra ao nome real do arquivo de imagem no disco.

    A consulta e feita sobre a listagem do diretorio, e nao por tentativa de
    abertura, porque o sistema de arquivos do Windows ignora a caixa das letras:
    um teste por "img1.jpg" encontraria "img1.JPG" e a divergencia de extensao
    passaria despercebida.
    """
    indice = {}
    for caminho in origem.iterdir():
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES:
            indice[caminho.stem] = caminho
    return indice


def caixas_do_xml(caminho, ocorrencias, amostra, largura_real, altura_real):
    """Extrai as caixas de um XML e as converte para o formato YOLO."""
    raiz = ET.parse(caminho).getroot()

    tam = raiz.find("size")
    largura = int(float(tam.find("width").text))
    altura = int(float(tam.find("height").text))

    if (largura, altura) != (largura_real, altura_real):
        ocorrencias.append((
            amostra, "dimensao_divergente",
            f"XML declara {largura}x{altura}; a imagem tem {largura_real}x{altura_real}",
            "Adotadas as dimensoes reais da imagem",
        ))
        largura, altura = largura_real, altura_real

    caixas = []
    descartadas = 0
    for obj in raiz.findall("object"):
        nome = obj.find("name").text.strip()
        cx = obj.find("bndbox")
        x_min = float(cx.find("xmin").text)
        y_min = float(cx.find("ymin").text)
        x_max = float(cx.find("xmax").text)
        y_max = float(cx.find("ymax").text)

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        # recorta ao retangulo da imagem
        x0, y0 = max(0.0, x_min), max(0.0, y_min)
        x1, y1 = min(float(largura), x_max), min(float(altura), y_max)
        if (x0, y0, x1, y1) != (x_min, y_min, x_max, y_max):
            ocorrencias.append((
                amostra, "caixa_fora_dos_limites",
                f"{nome}: ({x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}) "
                f"excede {largura}x{altura}",
                "Caixa recortada ao retangulo da imagem",
            ))

        if x1 - x0 <= 0 or y1 - y0 <= 0:
            ocorrencias.append((
                amostra, "caixa_degenerada",
                f"{nome}: largura {x1 - x0:.1f} e altura {y1 - y0:.1f} apos o recorte",
                "Caixa descartada",
            ))
            descartadas += 1
            continue

        caixas.append((
            nome,
            (x0 + x1) / (2 * largura),
            (y0 + y1) / (2 * altura),
            (x1 - x0) / largura,
            (y1 - y0) / altura,
        ))

    return caixas, descartadas


def le_rotulo_distribuido(caminho):
    """Le um rotulo no formato YOLO, ignorando linhas vazias."""
    linhas = []
    texto = caminho.read_text(encoding="utf-8", errors="replace")
    for linha in texto.splitlines():
        campos = linha.split()
        if len(campos) == 5:
            linhas.append((int(campos[0]), *(float(v) for v in campos[1:])))
    return linhas


def compara(amostra, geradas, descartadas, distribuidas, classes, ocorrencias):
    """Compara as caixas geradas com as distribuidas e registra divergencias.

    A contagem comparada e a do XML antes do descarte de caixas invalidas: uma
    caixa degenerada ja foi registrada como ocorrencia propria, e relata-la
    outra vez como divergencia de contagem duplicaria o mesmo defeito.
    """
    declaradas = len(geradas) + descartadas
    if declaradas != len(distribuidas):
        ocorrencias.append((
            amostra, "contagem_divergente",
            f"XML declara {declaradas} organismo(s); o rotulo distribuido "
            f"contem {len(distribuidas)}",
            "Rotulo regenerado a partir do XML",
        ))
        return
    if descartadas:
        return

    restantes = list(distribuidas)
    for nome, xc, yc, w, h in geradas:
        indice = classes.index(nome)
        alvo = None
        for k, (ic, ixc, iyc, iw, ih) in enumerate(restantes):
            if ic != indice:
                continue
            desvio = max(abs(ixc - xc), abs(iyc - yc), abs(iw - w), abs(ih - h))
            if desvio <= TOLERANCIA:
                alvo = k
                break
        if alvo is None:
            ocorrencias.append((
                amostra, "coordenada_divergente",
                f"{nome}: ({xc:.6f},{yc:.6f},{w:.6f},{h:.6f}) sem par no rotulo "
                f"distribuido dentro da tolerancia de {TOLERANCIA:g}",
                "Rotulo regenerado a partir do XML",
            ))
            return
        restantes.pop(alvo)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)

    origem = Path(sys.argv[1])
    saida = Path(sys.argv[2])

    texto_classes = (origem / "classes.txt").read_text(encoding="utf-8")
    classes = [linha.strip() for linha in texto_classes.splitlines() if linha.strip()]

    if saida.exists():
        shutil.rmtree(saida)
    dir_imagens = saida / "images"
    dir_rotulos = saida / "labels"
    dir_imagens.mkdir(parents=True)
    dir_rotulos.mkdir(parents=True)

    ocorrencias = []
    imagens = indexa_imagens(origem)
    amostras = sorted({p.stem for p in origem.glob("*.xml")}, key=lambda s: (len(s), s))

    total_caixas = 0
    por_classe = {nome: 0 for nome in classes}

    for amostra in amostras:
        xml = origem / f"{amostra}.xml"

        imagem = imagens.get(amostra)
        if imagem is None:
            ocorrencias.append((
                amostra, "imagem_ausente", "Nenhum arquivo de imagem encontrado",
                "Amostra descartada",
            ))
            continue
        if imagem.suffix != ".jpg":
            ocorrencias.append((
                amostra, "extensao_nao_canonica", f"Extensao {imagem.suffix}",
                "Extensao normalizada para .jpg na copia de trabalho",
            ))

        with Image.open(imagem) as img:
            largura_real, altura_real = img.size

        geradas, descartadas = caixas_do_xml(
            xml, ocorrencias, amostra, largura_real, altura_real
        )

        desconhecidas = sorted({n for n, *_ in geradas if n not in por_classe})
        if desconhecidas:
            ocorrencias.append((
                amostra, "classe_desconhecida",
                f"Classe(s) ausente(s) de classes.txt: {', '.join(desconhecidas)}",
                "Amostra descartada",
            ))
            continue

        txt = origem / f"{amostra}.txt"
        if not txt.exists():
            ocorrencias.append((
                amostra, "rotulo_ausente",
                "Possui imagem e XML, mas nao possui o rotulo no formato YOLO",
                "Rotulo gerado a partir do XML",
            ))
        else:
            compara(
                amostra, geradas, descartadas, le_rotulo_distribuido(txt),
                classes, ocorrencias,
            )

        duplo = origem / f"{amostra}.xml.txt"
        if duplo.exists():
            ocorrencias.append((
                amostra, "rotulo_com_dupla_extensao", f"Existe tambem {duplo.name}",
                "Arquivo redundante ignorado",
            ))

        shutil.copy2(imagem, dir_imagens / f"{amostra}.jpg")
        conteudo = "".join(
            f"{classes.index(n)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
            for n, xc, yc, w, h in geradas
        )
        (dir_rotulos / f"{amostra}.txt").write_text(conteudo, encoding="utf-8")

        total_caixas += len(geradas)
        for nome, *_ in geradas:
            por_classe[nome] += 1

    linhas_classes = "".join(f"{nome}\n" for nome in classes)
    (saida / "classes.txt").write_text(linhas_classes, encoding="utf-8")

    relatorio = saida / "relatorio_auditoria.csv"
    with relatorio.open("w", newline="", encoding="utf-8-sig") as fh:
        escritor = csv.writer(fh, delimiter=";")
        escritor.writerow(["amostra", "ocorrencia", "detalhe", "tratamento"])
        escritor.writerows(ocorrencias)

    n_imagens = len(list(dir_imagens.glob("*.jpg")))
    n_rotulos = len(list(dir_rotulos.glob("*.txt")))
    print(f"amostras com XML .......... {len(amostras)}")
    print(f"imagens gravadas .......... {n_imagens}")
    print(f"rotulos gravados .......... {n_rotulos}")
    print(f"caixas gravadas ........... {total_caixas}")
    print(f"ocorrencias registradas ... {len(ocorrencias)}")
    print()

    tipos = {}
    for _, tipo, *_ in ocorrencias:
        tipos[tipo] = tipos.get(tipo, 0) + 1
    for tipo, n in sorted(tipos.items(), key=lambda kv: -kv[1]):
        print(f"  {tipo:<28} {n}")
    print()

    for nome in classes:
        print(f"  {nome:<34} {por_classe[nome]:>4}")
    print(f"\nrelatorio: {relatorio}")


if __name__ == "__main__":
    main()

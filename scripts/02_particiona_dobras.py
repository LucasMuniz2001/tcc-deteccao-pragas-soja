"""Particiona o conjunto preparado em cinco dobras estratificadas multirrotulo.

A particao ocorre no nivel da imagem, nunca da caixa: todos os organismos de
uma imagem permanecem na mesma dobra. Imagens quase identicas sao agrupadas
previamente pelo resumo perceptual de diferenca (dHash) e tratadas como uma
unidade indivisivel, para que uma nao caia no treino enquanto sua quase-copia
cai na validacao.

Sobre as unidades resultantes aplica-se a estratificacao iterativa de Sechidis,
Tsoumakas e Vlahavas (2011): processam-se primeiro as classes mais raras e, a
cada passo, aloca-se a unidade na dobra que mais precisa daquela classe. Uma
estratificacao simples nao serve porque 424 das 1.800 imagens contem organismos
de mais de uma classe, e nao existe particao que equilibre todas as classes
simultaneamente.

A particao e deterministica: a semente e fixa e registrada na saida.

Uso:
    python -P 02_particiona_dobras.py <dir_dados> [--dobras 5] [--semente 0]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

LIMIAR_HAMMING = 10


# --------------------------------------------------------------------------
# Quase-duplicatas
# --------------------------------------------------------------------------

def dhash(caminho, lado=8):
    """Resumo perceptual de diferenca, em 64 bits.

    Reduz a imagem a (lado+1) x lado em tons de cinza e compara cada pixel ao
    vizinho da direita. O resultado depende da estrutura da imagem, nao de
    compressao, escala ou pequenas variacoes de brilho.
    """
    with Image.open(caminho) as img:
        reduzida = img.convert("L").resize((lado + 1, lado), Image.LANCZOS)
    matriz = np.asarray(reduzida, dtype=np.int16)
    bits = (matriz[:, 1:] > matriz[:, :-1]).flatten()
    valor = 0
    for bit in bits:
        valor = (valor << 1) | int(bit)
    return valor


def agrupa_quase_duplicatas(resumos, limiar=LIMIAR_HAMMING):
    """Agrupa amostras cujos resumos distam ate `limiar` bits.

    Comparacao exaustiva entre pares. Com 1.800 amostras sao cerca de 1,6
    milhao de comparacoes de inteiros, o que executa em poucos segundos e
    dispensa estrutura de indexacao.
    """
    nomes = list(resumos)
    pai = {nome: nome for nome in nomes}

    def raiz(a):
        while pai[a] != a:
            pai[a] = pai[pai[a]]
            a = pai[a]
        return a

    def une(a, b):
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            pai[rb] = ra

    pares = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            distancia = bin(resumos[a] ^ resumos[b]).count("1")
            if distancia <= limiar:
                pares.append((a, b, distancia))
                une(a, b)

    grupos = {}
    for nome in nomes:
        grupos.setdefault(raiz(nome), []).append(nome)

    return [sorted(membros) for membros in grupos.values() if len(membros) > 1], pares


# --------------------------------------------------------------------------
# Estratificacao iterativa
# --------------------------------------------------------------------------

def estratifica(unidades, pesos, n_classes, k, semente):
    """Estratificacao iterativa multirrotulo.

    `unidades[i][c]` e o numero de imagens da unidade i que contem a classe c, e
    `pesos[i]` o numero de imagens da unidade. A formulacao e a de Sechidis,
    Tsoumakas e Vlahavas (2011), que opera sobre a presenca da classe na imagem,
    e nao sobre a contagem de caixas: equilibrar contagens de caixas deixaria a
    quantidade de imagens por dobra livre para variar, e a validacao cruzada
    pressupoe dobras de tamanho comparavel.

    Devolve o indice da dobra de cada unidade.
    """
    sorteio = random.Random(semente)
    n = len(unidades)

    # quanto cada dobra ainda deve receber, no total e por classe
    falta_total = [sum(pesos) / k] * k
    totais = [sum(u[c] for u in unidades) for c in range(n_classes)]
    falta_classe = [[t / k for _ in range(k)] for t in totais]

    restantes = [sum(u) for u in unidades]
    pendentes = set(range(n))
    dobra = [-1] * n

    # unidades sem nenhuma anotacao nao influenciam a estratificacao por classe
    sem_rotulo = sorted(i for i in pendentes if restantes[i] == 0)

    def atribui(indice, destino):
        dobra[indice] = destino
        pendentes.discard(indice)
        falta_total[destino] -= pesos[indice]
        for c in range(n_classes):
            if unidades[indice][c]:
                falta_classe[c][destino] -= unidades[indice][c]

    while True:
        # classe mais rara entre as ainda pendentes
        contagem = [0] * n_classes
        for i in pendentes:
            if restantes[i] == 0:
                continue
            for c in range(n_classes):
                contagem[c] += unidades[i][c]
        if not any(contagem):
            break
        classe = min(
            (c for c in range(n_classes) if contagem[c] > 0),
            key=lambda c: (contagem[c], c),
        )

        candidatas = sorted(i for i in pendentes if unidades[i][classe])
        sorteio.shuffle(candidatas)
        for indice in candidatas:
            melhor = max(falta_classe[classe])
            empatadas = [m for m in range(k) if falta_classe[classe][m] == melhor]
            if len(empatadas) > 1:
                melhor_total = max(falta_total[m] for m in empatadas)
                empatadas = [m for m in empatadas if falta_total[m] == melhor_total]
            atribui(indice, sorteio.choice(empatadas))

    # o que sobrou nao tem anotacao: equilibra apenas a contagem de imagens
    for indice in sem_rotulo:
        if dobra[indice] != -1:
            continue
        melhor_total = max(falta_total)
        empatadas = [m for m in range(k) if falta_total[m] == melhor_total]
        atribui(indice, sorteio.choice(empatadas))

    assert not pendentes, "restaram unidades sem dobra atribuida"
    return dobra


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dados", type=Path, help="diretorio produzido pela etapa 1")
    ap.add_argument("--dobras", type=int, default=5)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument(
        "--raiz",
        default=None,
        help="caminho a declarar no campo 'path' dos arquivos de configuracao; "
             "informe o caminho do ambiente de treinamento quando ele diferir "
             "deste (por exemplo /content/dados_yolo no Colab)",
    )
    args = ap.parse_args()

    dados = args.dados
    k = args.dobras
    classes = [
        linha.strip()
        for linha in (dados / "classes.txt").read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]

    amostras = sorted(
        (p.stem for p in (dados / "images").glob("*.jpg")),
        key=lambda s: (len(s), s),
    )

    # contagem de caixas por classe, por amostra
    contagens = {}
    for amostra in amostras:
        vetor = [0] * len(classes)
        texto = (dados / "labels" / f"{amostra}.txt").read_text(encoding="utf-8")
        for linha in texto.splitlines():
            campos = linha.split()
            if campos:
                vetor[int(campos[0])] += 1
        contagens[amostra] = vetor

    print(f"amostras .................. {len(amostras)}")
    print(f"caixas .................... {sum(sum(v) for v in contagens.values())}")
    multirrotulo = sum(1 for v in contagens.values() if sum(1 for c in v if c) > 1)
    print(f"imagens multiclasse ....... {multirrotulo}")

    print("\ncalculando resumos perceptuais...")
    resumos = {a: dhash(dados / "images" / f"{a}.jpg") for a in amostras}
    grupos, pares = agrupa_quase_duplicatas(resumos)
    envolvidas = sum(len(g) for g in grupos)
    print(f"grupos de quase-duplicatas  {len(grupos)} ({envolvidas} imagens)")
    for g in grupos:
        print(f"  {', '.join(g)}")

    # unidades indivisiveis: cada grupo e uma, cada amostra solta e uma
    unidades = [list(g) for g in grupos]
    agrupadas = {a for g in grupos for a in g}
    unidades += [[a] for a in amostras if a not in agrupadas]
    unidades.sort(key=lambda u: (len(u[0]), u[0]))

    # presenca da classe na imagem, somada sobre os membros da unidade
    vetores = [
        [sum(1 for a in unidade if contagens[a][c]) for c in range(len(classes))]
        for unidade in unidades
    ]
    pesos = [len(unidade) for unidade in unidades]

    print(f"\nunidades a particionar .... {len(unidades)}")
    print(f"estratificando em {k} dobras, semente {args.semente}...")
    atribuicao = estratifica(vetores, pesos, len(classes), k, args.semente)

    membros = [[] for _ in range(k)]
    for unidade, destino in zip(unidades, atribuicao):
        membros[destino].extend(unidade)
    for lista in membros:
        lista.sort(key=lambda s: (len(s), s))

    # ------------------------------------------------------------------
    # verificacao
    # ------------------------------------------------------------------
    print("\ndistribuicao das caixas por classe e dobra")
    cabecalho = "  " + "classe".ljust(34) + "".join(f"{f'd{m+1}':>7}" for m in range(k))
    print(cabecalho + f"{'total':>8}")
    for c, nome in enumerate(classes):
        por_dobra = [sum(contagens[a][c] for a in membros[m]) for m in range(k)]
        print(
            "  " + nome.ljust(34)
            + "".join(f"{v:>7}" for v in por_dobra)
            + f"{sum(por_dobra):>8}"
        )
    print(
        "  " + "imagens".ljust(34)
        + "".join(f"{len(membros[m]):>7}" for m in range(k))
        + f"{sum(len(m) for m in membros):>8}"
    )

    todas = [a for lista in membros for a in lista]
    assert len(todas) == len(set(todas)) == len(amostras), "dobras inconsistentes"
    for grupo in grupos:
        destinos = {atribuicao[unidades.index(sorted(grupo))]}
        assert len(destinos) == 1

    # ------------------------------------------------------------------
    # gravacao
    # ------------------------------------------------------------------
    dir_dobras = dados / "dobras"
    dir_dobras.mkdir(exist_ok=True)
    raiz = args.raiz if args.raiz else dados.resolve().as_posix()

    for m in range(k):
        treino = [a for j in range(k) if j != m for a in membros[j]]
        treino.sort(key=lambda s: (len(s), s))
        for sufixo, lista in (("train", treino), ("val", membros[m])):
            # O caminho e relativo ao proprio arquivo de lista, e nao ao campo
            # 'path' do YAML: a Ultralytics substitui o prefixo "./" pelo
            # diretorio onde a lista esta. Como as listas ficam em dobras/, a
            # subida de um nivel e obrigatoria; sem ela a biblioteca procura as
            # imagens em dobras/images/ e rejeita as 1.800 como ilegiveis.
            linhas = "".join(f"./../images/{a}.jpg\n" for a in lista)
            (dir_dobras / f"dobra{m+1}_{sufixo}.txt").write_text(linhas, encoding="utf-8")

        nomes = "\n".join(f"  {c}: {nome}" for c, nome in enumerate(classes))
        yaml = (
            f"# Dobra {m+1} de {k} - gerado por 02_particiona_dobras.py\n"
            f"# semente {args.semente}; nao editar a mao\n"
            f"path: {raiz}\n"
            f"train: dobras/dobra{m+1}_train.txt\n"
            f"val: dobras/dobra{m+1}_val.txt\n"
            f"nc: {len(classes)}\n"
            f"names:\n{nomes}\n"
        )
        (dados / f"data_dobra{m+1}.yaml").write_text(yaml, encoding="utf-8")

    registro = {
        "dobras": k,
        "semente": args.semente,
        "limiar_hamming": LIMIAR_HAMMING,
        "amostras": len(amostras),
        "unidades": len(unidades),
        "grupos_quase_duplicatas": grupos,
        "pares_detectados": [[a, b, d] for a, b, d in pares],
        "atribuicao": {a: m + 1 for m in range(k) for a in membros[m]},
    }
    destino = dados / "particao.json"
    destino.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nlistas .... {dir_dobras}")
    print(f"particao .. {destino}")


if __name__ == "__main__":
    main()

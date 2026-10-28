"""Agrega a avaliacao das execucoes nas tres analises da Secao 2.6.2.

Le os JSON gravados pelo notebook de avaliacao, um por execucao e por peso
(best ou last), e produz:

1. Desempenho por classe cruzado com o numero de exemplares: media e desvio
   padrao entre execucoes de cada metrica, ao lado do numero de instancias da
   classe na dobra de validacao.
2. Matriz de confusao entre as doze classes, somada sobre as execucoes e
   normalizada pela classe verdadeira, com destaque para as tres trocas
   previsiveis pela biologia: adulto x ninfa da mesma especie, percevejos de
   especies distintas e lagartas desfolhadoras entre si.
3. Praga x nao praga: as classes sao agrupadas por papel agronomico e
   calcula-se a taxa de confusao entre os dois grupos, em cada direcao.

As matrizes do JSON estao em contagens brutas, com linhas = classe predita,
colunas = classe verdadeira e o ultimo indice = fundo. Somar contagens entre
execucoes e legitimo porque cada execucao avalia uma dobra disjunta; as taxas
por execucao sao relatadas tambem, com media e desvio padrao, para expor a
dispersao.

Uso:
    python -P 03_analisa_avaliacao.py <dir_json> [--modelo yolov8m] [--peso best]
"""

import argparse
import json
from pathlib import Path

import numpy as np

METRICAS = ("precisao", "revocacao", "mAP50", "mAP50_95")

# Papel agronomico (Secao 2.6.2). A Lagria villosa fica entre as pragas: e
# secundaria, mas pode causar dano, e trata-la como inofensiva seria a escolha
# menos conservadora.
NAO_PRAGA = {"Coccinellidae"}

# Trocas previsiveis pela biologia, por nome de classe.
ADULTO_NINFA = [
    ("Euschistus_heros_adulto", "Euschistus_heros_ninfa"),
    ("Nezara_viridula_adulto", "Nezara_viridula_ninfa"),
]
PERCEVEJOS = [
    "Edessa_meditabunda", "Euschistus_heros_adulto", "Euschistus_heros_ninfa",
    "Nezara_viridula_adulto", "Nezara_viridula_ninfa",
]
LAGARTAS = ["Anticarsia_gemmatalis", "Spodoptera_albula"]


def carrega(diretorio, modelo, peso):
    registros = []
    for arq in sorted(Path(diretorio).glob(f"{modelo}_dobra*_s*_{peso}.json")):
        registros.append(json.loads(arq.read_text(encoding="utf-8")))
    if not registros:
        raise SystemExit(f"nenhum JSON de {modelo} ({peso}) em {diretorio}")
    return registros


def media_dp(valores):
    v = np.asarray(valores, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def por_classe(registros):
    classes = [c["classe"] for c in registros[0]["classes"]]
    linhas = []
    for i, nome in enumerate(classes):
        itens = [r["classes"][i] for r in registros]
        linha = {"classe": nome,
                 "instancias_por_dobra": media_dp([c["instancias"] for c in itens])[0]}
        for m in METRICAS:
            linha[m] = media_dp([c.get(m, np.nan) for c in itens])
        linhas.append(linha)
    return classes, linhas


def matriz_somada(registros):
    return sum(np.asarray(r["matriz"], dtype=float) for r in registros)


def normaliza_por_verdadeira(matriz):
    """Cada coluna (classe verdadeira) passa a somar 1."""
    totais = matriz.sum(axis=0, keepdims=True)
    return np.divide(matriz, totais, out=np.zeros_like(matriz), where=totais > 0)


def confusao_entre(matriz, classes, verdadeiras, preditas):
    """Fracao das instancias de 'verdadeiras' detectadas como alguma de 'preditas'."""
    iv = [classes.index(c) for c in verdadeiras]
    ip = [classes.index(c) for c in preditas]
    total = matriz[:, iv].sum()
    return matriz[np.ix_(ip, iv)].sum() / total if total else np.nan


def praga_x_nao_praga(registros, classes):
    nao = [c for c in classes if c in NAO_PRAGA]
    praga = [c for c in classes if c not in NAO_PRAGA]
    por_execucao = {"nao_praga_como_praga": [], "praga_como_nao_praga": []}
    for r in registros:
        m = np.asarray(r["matriz"], dtype=float)
        por_execucao["nao_praga_como_praga"].append(confusao_entre(m, classes, nao, praga))
        por_execucao["praga_como_nao_praga"].append(confusao_entre(m, classes, praga, nao))
    soma = matriz_somada(registros)
    return {
        chave: dict(somada=confusao_entre(soma, classes,
                                          nao if chave.startswith("nao") else praga,
                                          praga if chave.startswith("nao") else nao),
                    por_execucao=media_dp(valores))
        for chave, valores in por_execucao.items()
    }


def fmt(par, casas=3):
    return f"{par[0]:.{casas}f} ± {par[1]:.{casas}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_json", type=Path)
    ap.add_argument("--modelo", default="yolov8m")
    ap.add_argument("--peso", choices=("best", "last"), default="best")
    ap.add_argument("--saida", type=Path, help="grava o relatorio em Markdown aqui")
    args = ap.parse_args()

    registros = carrega(args.dir_json, args.modelo, args.peso)
    classes, linhas = por_classe(registros)
    nomes = classes + ["fundo"]
    soma = matriz_somada(registros)
    norm = normaliza_por_verdadeira(soma)
    saida = []

    saida.append(f"# {args.modelo} — {args.peso}.pt — {len(registros)} execuções\n")
    conf = {r["matriz_conf"] for r in registros}
    saida.append(f"Matriz de confusão com conf = {', '.join(map(str, conf))} e IoU de "
                 f"casamento = {registros[0]['matriz_iou']}.\n")

    saida.append("## 1. Desempenho por classe\n")
    saida.append("| classe | instâncias por dobra | precisão | revocação | mAP@0,5 | mAP@0,5:0,95 |")
    saida.append("|---|---|---|---|---|---|")
    for l in sorted(linhas, key=lambda l: l["instancias_por_dobra"]):
        saida.append(f"| {l['classe']} | {l['instancias_por_dobra']:.1f} | "
                     + " | ".join(fmt(l[m]) for m in METRICAS) + " |")

    saida.append("\n## 2. Matriz de confusão (fração da classe verdadeira)\n")
    saida.append("Maiores trocas fora da diagonal, entre classes e com o fundo:\n")
    trocas = []
    for v in range(len(nomes)):
        for p in range(len(nomes)):
            if p != v and v < len(classes) and soma[p, v] > 0:
                trocas.append((norm[p, v], int(soma[p, v]), nomes[v], nomes[p]))
    saida.append("| verdadeira | predita como | fração | contagem |")
    saida.append("|---|---|---|---|")
    for frac, cont, v, p in sorted(trocas, reverse=True)[:15]:
        saida.append(f"| {v} | {p} | {frac:.3f} | {cont} |")

    saida.append("\n### Trocas previsíveis pela biologia\n")
    saida.append("| troca | fração |")
    saida.append("|---|---|")
    for a, b in ADULTO_NINFA:
        saida.append(f"| {a} → {b} | {confusao_entre(soma, classes, [a], [b]):.3f} |")
        saida.append(f"| {b} → {a} | {confusao_entre(soma, classes, [b], [a]):.3f} |")
    especie = lambda c: c.rsplit("_", 1)[0] if c.endswith(("_adulto", "_ninfa")) else c
    entre_especies = sum(
        soma[classes.index(p), classes.index(v)]
        for v in PERCEVEJOS for p in PERCEVEJOS if especie(v) != especie(p)
    ) / soma[:, [classes.index(c) for c in PERCEVEJOS]].sum()
    saida.append(f"| percevejos de espécies distintas | {entre_especies:.3f} |")
    a, b = LAGARTAS
    lagartas = (soma[classes.index(b), classes.index(a)] + soma[classes.index(a), classes.index(b)]) \
        / soma[:, [classes.index(a), classes.index(b)]].sum()
    saida.append(f"| entre lagartas ({a} ↔ {b}) | {lagartas:.3f} |")

    saida.append("\n## 3. Praga × não praga\n")
    saida.append(f"Não praga: {', '.join(sorted(NAO_PRAGA))}. Praga: as demais "
                 f"{len(classes) - len(NAO_PRAGA)} classes.\n")
    saida.append("| direção | somada | média ± dp por execução |")
    saida.append("|---|---|---|")
    rotulos = {"nao_praga_como_praga": "não praga detectada como praga",
               "praga_como_nao_praga": "praga detectada como não praga"}
    for chave, v in praga_x_nao_praga(registros, classes).items():
        saida.append(f"| {rotulos[chave]} | {v['somada']:.4f} | {fmt(v['por_execucao'], 4)} |")

    texto = "\n".join(saida) + "\n"
    print(texto)
    if args.saida:
        args.saida.write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Mede o tempo de inferencia em CPU com os modelos exportados para ONNX.

Repete na maquina local o procedimento da Secao 2.6.3 usado na T4: uma imagem
por vez, as vinte primeiras descartadas para aquecimento e as cem seguintes
cronometradas, com o tempo decomposto em pre-processamento, rede e
pos-processamento. As imagens sao as mesmas da medicao na T4 (as 120 primeiras
da validacao da dobra 1) e sao lidas do disco antes da medicao.

O modelo ONNX e executado pela propria Ultralytics, que usa o ONNX Runtime por
baixo. Assim o pre e o pos-processamento sao identicos aos da medicao em GPU, e
a unica diferenca entre os dois numeros e o dispositivo que executa a rede.

Uso:
    python -P 04_tempo_cpu.py <dir_dados> <dir_onnx> <saida.csv>
"""

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from ultralytics import YOLO

AQUECIMENTO, MEDIDAS = 20, 100
MODELOS = ["yolov8n", "yolov8s", "yolov8m"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_dados", type=Path)
    ap.add_argument("dir_onnx", type=Path)
    ap.add_argument("saida", type=Path)
    args = ap.parse_args()

    lista = (args.dir_dados / "dobras" / "dobra1_val.txt").read_text(encoding="utf-8").split()
    quadros = [cv2.imread(str(args.dir_dados / "images" / Path(l).name))
               for l in lista[:AQUECIMENTO + MEDIDAS]]
    assert all(q is not None for q in quadros)

    print(f"processador ... {platform.processor()} ({os.cpu_count()} processadores logicos)")
    print(f"onnxruntime ... {onnxruntime.__version__}, provedores {onnxruntime.get_available_providers()}")

    linhas = []
    for modelo in MODELOS:
        rede = YOLO(str(args.dir_onnx / f"{modelo}_dobra1_s0_best.onnx"), task="detect")
        for i, quadro in enumerate(quadros):
            t0 = time.perf_counter()
            r = rede.predict(quadro, imgsz=640, device="cpu", verbose=False)[0]
            total = (time.perf_counter() - t0) * 1000
            if i >= AQUECIMENTO:
                linhas.append(dict(modelo=modelo, imagem=i, **r.speed, total=total))

    with args.saida.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    # Mediana e intervalo interquartil, como na medicao da T4.
    print(f"\n{'modelo':8} {'etapa':12} {'p25':>8} {'mediana':>8} {'p75':>8}  (ms)")
    for modelo in MODELOS:
        for etapa in ("preprocess", "inference", "postprocess", "total"):
            v = np.array([l[etapa] for l in linhas if l["modelo"] == modelo])
            p25, p50, p75 = np.percentile(v, [25, 50, 75])
            print(f"{modelo:8} {etapa:12} {p25:8.1f} {p50:8.1f} {p75:8.1f}")


if __name__ == "__main__":
    main()

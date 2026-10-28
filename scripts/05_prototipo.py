"""Prototipo de demonstracao descrito na Secao 2.7.

Aplicacao local de pagina unica: o usuario envia uma imagem e recebe as caixas
detectadas, a contagem de organismos por classe com o papel agronomico de cada
um e o tempo gasto, decomposto em pre-processamento, rede e pos-processamento.

O modelo e carregado uma unica vez, na inicializacao, e os limiares sao os do
Quadro 4 (confianca 0,25 e sobreposicao 0,45). Os pesos indicados devem ser os
da execucao de desempenho mediano, e nao os da melhor, para que a demonstracao
nao seja otimista em relacao ao desempenho relatado.

Nada e gravado em disco: a imagem enviada vive apenas na memoria da sessao.

Uso:
    python -P 05_prototipo.py <pesos.pt> [--porta 7860] [--exemplos img1.jpg img2.jpg ...]
"""

import argparse
import time

import gradio as gr
from ultralytics import YOLO

CONFIANCA, SOBREPOSICAO = 0.25, 0.45

# Nome de exibicao e papel agronomico (Tabela 1), na ordem das classes do modelo.
CLASSES = {
    "Anticarsia_gemmatalis": ("Anticarsia gemmatalis", "praga desfolhadora"),
    "Coccinellidae": ("Coccinellidae (joaninha)", "inimigo natural"),
    "Diabrotica_speciosa": ("Diabrotica speciosa", "praga desfolhadora"),
    "Edessa_meditabunda": ("Edessa meditabunda", "praga sugadora"),
    "Euschistus_heros_adulto": ("Euschistus heros (adulto)", "praga sugadora"),
    "Euschistus_heros_ninfa": ("Euschistus heros (ninfa)", "praga sugadora"),
    "Gastropoda": ("Gastropoda (lesma)", "praga ocasional"),
    "Lagria_villosa": ("Lagria villosa", "praga ocasional"),
    "Nezara_viridula_adulto": ("Nezara viridula (adulto)", "praga sugadora"),
    "Nezara_viridula_ninfa": ("Nezara viridula (ninfa)", "praga sugadora"),
    "Rhammatocerus_schistocercoides": ("Rhammatocerus schistocercoides", "praga desfolhadora"),
    "Spodoptera_albula": ("Spodoptera albula", "praga desfolhadora"),
}


def monta_interface(modelo, exemplos):
    def detecta(imagem):
        if imagem is None:
            return None, [], "Envie uma imagem."
        t0 = time.perf_counter()
        r = modelo.predict(imagem[:, :, ::-1], imgsz=640, conf=CONFIANCA, iou=SOBREPOSICAO,
                           device="cpu", verbose=False)[0]
        total = (time.perf_counter() - t0) * 1000

        contagem = {}
        for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            nome = modelo.names[int(c)]
            n, maior = contagem.get(nome, (0, 0.0))
            contagem[nome] = (n + 1, max(maior, p))
        tabela = [[CLASSES[n][0], CLASSES[n][1], k, f"{p:.2f}"]
                  for n, (k, p) in sorted(contagem.items(), key=lambda x: -x[1][0])]

        s = r.speed
        resumo = (f"**{len(r.boxes)} organismo(s) detectado(s)** em {total:.0f} ms "
                  f"(pré-processamento {s['preprocess']:.0f} ms, rede {s['inference']:.0f} ms, "
                  f"pós-processamento {s['postprocess']:.0f} ms).")
        if "Coccinellidae" in contagem:
            resumo += ("\n\n**Atenção:** há inimigo natural na imagem. A presença de joaninhas indica "
                       "controle biológico em curso e deve ser considerada antes de qualquer aplicação.")
        return r.plot(line_width=2)[:, :, ::-1], tabela, resumo

    with gr.Blocks(title="Detecção de insetos-praga da soja") as app:
        gr.Markdown("## Detecção de insetos-praga da soja\n"
                    "Protótipo de demonstração do TCC — YOLOv8m treinado no conjunto INSECT12C. "
                    f"Limiares: confiança {CONFIANCA}, sobreposição {SOBREPOSICAO}.")
        with gr.Row():
            entrada = gr.Image(label="Imagem da lavoura", type="numpy")
            saida = gr.Image(label="Detecções")
        botao = gr.Button("Detectar", variant="primary")
        resumo = gr.Markdown()
        tabela = gr.Dataframe(headers=["Organismo", "Papel agronômico", "Quantidade", "Maior confiança"],
                              label="Contagem por classe", interactive=False)
        if exemplos:
            gr.Examples(exemplos, entrada, label="Imagens de exemplo")
        botao.click(detecta, entrada, [saida, tabela, resumo])
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pesos")
    ap.add_argument("--porta", type=int, default=7860)
    ap.add_argument("--exemplos", nargs="*", default=[],
                    help="imagens oferecidas como exemplo; use as da dobra de validacao dos pesos")
    args = ap.parse_args()

    modelo = YOLO(args.pesos)
    assert set(modelo.names.values()) == set(CLASSES), "pesos com classes diferentes das do INSECT12C"
    monta_interface(modelo, args.exemplos).launch(server_name="127.0.0.1", server_port=args.porta)


if __name__ == "__main__":
    main()

"""Utilitários de visualização compartilhados (estilo e salvamento de figuras)."""
import shutil

import matplotlib
matplotlib.use("Agg")  # backend sem janela, para gerar arquivos
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETA = "colorblind"


def salvar(fig, nome):
    """Salva a figura em results/figures e copia para paper/figures (PDF + PNG)."""
    config.garantir_diretorios()
    for ext in ("png", "pdf"):
        caminho = config.DIR_FIGURAS / f"{nome}.{ext}"
        fig.savefig(caminho, bbox_inches="tight", dpi=160)
    # cópia em PDF para o artigo
    shutil.copy(config.DIR_FIGURAS / f"{nome}.pdf", config.DIR_FIG_ARTIGO / f"{nome}.pdf")
    plt.close(fig)
    print(f"  figura: {nome}.pdf/.png")

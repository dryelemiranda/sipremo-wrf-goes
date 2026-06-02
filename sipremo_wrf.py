"""
SIPREMO-WRF — Sistema Integrado de Previsão Meteorológica Operacional
======================================================================
Módulos:
  - config         : parâmetros globais em um único lugar
  - logger         : log formatado com timestamps e níveis
  - goes           : busca e download do GOES-19 (S3 anônimo)
  - wrf_reader     : leitura robusta do wrfout (via memória RAM)
  - plotter        : geração de mapas (validação e alertas)
  - pipeline       : orquestra tudo em sequência

Uso:
  python sipremo_wrf.py                     # roda pipeline completo
  python sipremo_wrf.py --modo validacao    # só mapa de validação
  python sipremo_wrf.py --modo alertas      # só mapa de alertas
  python sipremo_wrf.py --modo ambos        # ambos (padrão)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import s3fs
except ImportError:
    s3fs = None  # detectado em runtime


# ============================================================
# MÓDULO: CONFIG
# ============================================================

@dataclass
class Config:
    """Todos os parâmetros do sistema em um único lugar."""

    # --- Caminhos WRF ---
    wrf_path: str = r"C:\Users\digit\OneDrive\Área de Trabalho\wrfconfiguracao"
    wrf_file: str = "wrfout_d01_2026-04-29_00:00:00"

    # --- Pastas de saída ---
    dir_satelite: str = "dados_satelite"
    dir_validacao: str = "produtos_sipremo/validacao"
    dir_alertas: str   = "produtos_sipremo/alertas"

    # --- GOES-19 ---
    goes_bucket_template: str = "noaa-goes19/ABI-L2-CMIPF/{ano}/{juliano}/{hora}"
    goes_banda: str = "M6C13"
    goes_atraso_min: int = 45          # minutos de atraso para garantir upload na NOAA

    # --- Domínio geográfico ---
    extent: tuple = (-46.5, -40.5, -24.5, -21.0)   # (lon_min, lon_max, lat_min, lat_max)

    # --- Limiares de nuvem ---
    limiar_validacao: tuple = (0.5, 0.9)             # contornos de fração de nuvem
    limiar_alerta_baixo: tuple = (0.3, 0.7, 1.0)    # faixas nuvem baixa
    limiar_medio: float = 0.6
    limiar_alto:  float = 0.5

    # --- Camadas verticais WRF (índices 0-based) ---
    camadas_baixas:  tuple = (0, 8)
    camadas_medias:  tuple = (9, 18)
    camadas_altas:   tuple = (19, None)

    # --- Plotagem ---
    dpi: int = 150
    figsize_validacao: tuple = (12, 10)
    figsize_alertas:   tuple = (14, 10)

    @property
    def full_path_wrf(self) -> str:
        return os.path.join(self.wrf_path, self.wrf_file)

    def criar_dirs(self) -> None:
        for d in [self.dir_satelite, self.dir_validacao, self.dir_alertas]:
            Path(d).mkdir(parents=True, exist_ok=True)


CFG = Config()


# ============================================================
# MÓDULO: LOGGER
# ============================================================

def criar_logger(nome: str = "SIPREMO") -> logging.Logger:
    logger = logging.getLogger(nome)
    if logger.handlers:
        return logger          # já configurado (evita handlers duplicados)

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Arquivo — DEBUG+ (um arquivo por execução)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"sipremo_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOG = criar_logger()


# ============================================================
# MÓDULO: GOES
# ============================================================

class ErroGOES(RuntimeError):
    pass


def _bucket_para_datetime(dt: datetime, cfg: Config) -> str:
    return cfg.goes_bucket_template.format(
        ano=dt.year,
        juliano=dt.strftime("%j"),
        hora=dt.strftime("%H"),
    )


def buscar_arquivo_goes(cfg: Config = CFG) -> Path:
    """
    Localiza o arquivo mais recente da banda 13 do GOES-19 no S3 da NOAA.
    Tenta a hora atual e, se vazio, a hora anterior.
    Retorna o Path local do arquivo (baixando se necessário).
    """
    if s3fs is None:
        raise ErroGOES("Biblioteca 's3fs' não instalada. Execute: pip install s3fs")

    agora = datetime.now(timezone.utc) - timedelta(minutes=cfg.goes_atraso_min)
    s3 = s3fs.S3FileSystem(anon=True)

    for tentativa in range(3):
        bucket = _bucket_para_datetime(agora, cfg)
        LOG.info(f"[GOES] Buscando em: s3://{bucket}")

        try:
            arquivos = s3.ls(bucket)
        except FileNotFoundError:
            arquivos = []

        candidatos = [f for f in arquivos if cfg.goes_banda in f]

        if candidatos:
            break

        LOG.warning(f"[GOES] Nenhum arquivo encontrado. Tentativa {tentativa+1}/3 — recuando 1 hora.")
        agora -= timedelta(hours=1)
    else:
        raise ErroGOES("Não foi possível localizar arquivo do GOES-19 nas últimas 3 horas.")

    item = candidatos[-1]
    nome_arquivo = item.split("/")[-1]
    destino = Path(cfg.dir_satelite) / nome_arquivo

    if destino.exists():
        LOG.info(f"[GOES] Arquivo já existe localmente: {destino}")
    else:
        LOG.info(f"[GOES] Baixando → {destino}")
        t0 = time.time()
        s3.get(item, str(destino))
        LOG.info(f"[GOES] Download concluído em {time.time()-t0:.1f}s")

    return destino, agora


# ============================================================
# MÓDULO: WRF READER
# ============================================================

class ErroWRF(RuntimeError):
    pass


def ler_wrf(cfg: Config = CFG) -> xr.Dataset:
    """
    Lê o arquivo wrfout via memória RAM (evita erros de DLL no Windows/WSL).
    Retorna dataset com Time=0 selecionado.
    """
    caminho = cfg.full_path_wrf
    LOG.info(f"[WRF] Lendo: {caminho}")

    if not os.path.exists(caminho):
        raise ErroWRF(f"Arquivo WRF não encontrado: {caminho}")

    tamanho_mb = os.path.getsize(caminho) / (1024 ** 2)
    LOG.debug(f"[WRF] Tamanho do arquivo: {tamanho_mb:.1f} MB")

    t0 = time.time()
    try:
        with open(caminho, "rb") as f:
            dados = f.read()
    except PermissionError as e:
        raise ErroWRF(f"Sem permissão para ler o arquivo WRF: {e}") from e

    try:
        ds = xr.open_dataset(io.BytesIO(dados), engine="h5netcdf")
    except Exception as e:
        raise ErroWRF(f"Falha ao abrir dataset WRF com h5netcdf: {e}") from e

    LOG.info(f"[WRF] Leitura concluída em {time.time()-t0:.1f}s")
    LOG.debug(f"[WRF] Variáveis disponíveis: {list(ds.data_vars)}")
    LOG.debug(f"[WRF] Dimensões: {dict(ds.sizes)}")

    # Validação mínima
    for var in ("CLDFRA", "XLAT", "XLONG"):
        if var not in ds:
            raise ErroWRF(f"Variável obrigatória '{var}' não encontrada no dataset WRF.")

    return ds.isel(Time=0)


# ============================================================
# MÓDULO: GOES READER
# ============================================================

def ler_goes(caminho: Path) -> xr.Dataset:
    LOG.info(f"[GOES] Abrindo dataset: {caminho.name}")
    try:
        ds = xr.open_dataset(str(caminho), engine="h5netcdf")
    except Exception as e:
        raise ErroGOES(f"Falha ao abrir dataset GOES: {e}") from e

    for var in ("CMI", "goes_imager_projection", "x", "y"):
        if var not in ds and var not in ds.coords:
            raise ErroGOES(f"Campo obrigatório '{var}' ausente no arquivo GOES.")

    return ds


# ============================================================
# MÓDULO: PLOTTER
# ============================================================

def _estilo_mapa(ax, cfg: Config) -> None:
    """Aplica feições geográficas padrão ao eixo."""
    ax.set_extent(list(cfg.extent), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, edgecolor="cyan", linewidth=1.2)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="cyan", alpha=0.5, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, edgecolor="cyan", alpha=0.4, linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.4, color="white", alpha=0.4,
                 linestyle="--", x_inline=False, y_inline=False)


def _projecao_goes(ds_ir: xr.Dataset):
    """Retorna projeção Cartopy e extent para imagem do GOES."""
    dat = ds_ir.goes_imager_projection
    h = float(dat.perspective_point_height)
    proj = ccrs.Geostationary(
        central_longitude=float(dat.longitude_of_projection_origin),
        satellite_height=h,
        sweep_axis=str(dat.sweep_angle_axis),
    )
    img_extent = (
        float(ds_ir.x.min()) * h,
        float(ds_ir.x.max()) * h,
        float(ds_ir.y.min()) * h,
        float(ds_ir.y.max()) * h,
    )
    return proj, img_extent


def gerar_mapa_validacao(
    ds_wrf: xr.Dataset,
    ds_ir: xr.Dataset,
    agora: datetime,
    cfg: Config = CFG,
) -> Path:
    """
    Gera mapa de validação: satélite GOES como fundo + contornos do modelo WRF.
    Salva em dir_validacao e retorna o Path do arquivo.
    """
    LOG.info("[PLOT] Gerando mapa de validação...")

    lats = ds_wrf["XLAT"]
    lons = ds_wrf["XLONG"]
    cld_max = ds_wrf["CLDFRA"].max(dim="bottom_top")

    proj_sat, img_extent = _projecao_goes(ds_ir)

    fig = plt.figure(figsize=cfg.figsize_validacao, facecolor="#0d1117")
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor="#0d1117")
    _estilo_mapa(ax, cfg)

    # Fundo: imagem IR do GOES
    ax.imshow(
        ds_ir["CMI"],
        origin="upper",
        extent=img_extent,
        transform=proj_sat,
        cmap="Greys_r",
        vmin=190,
        vmax=300,
    )

    # Contornos do modelo
    niveis = list(cfg.limiar_validacao)
    cores_contorno = ["yellow", "magenta"]
    cs = ax.contour(
        lons, lats, cld_max,
        levels=niveis,
        colors=cores_contorno,
        linewidths=2.0,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt={0.5: "50%", 0.9: "90%"}, fontsize=8, colors="white")

    # Legenda
    legenda = [
        Line2D([0], [0], color="yellow",  lw=2.5, label="Modelo: Nuvem > 50%"),
        Line2D([0], [0], color="magenta", lw=2.5, label="Modelo: Teto Fechado > 90%"),
        Line2D([0], [0], color="white",   lw=1.0, label="GOES-19 · Canal IR (10.3 µm)"),
    ]
    ax.legend(handles=legenda, loc="lower right", facecolor="#1a1f2e",
              edgecolor="gray", labelcolor="white", fontsize=9)

    # Títulos
    fig.suptitle("MONITORAMENTO SIPREMO-WRF", fontsize=15, fontweight="bold",
                 color="white", y=0.98)
    ax.set_title(
        f"Validação: Modelo vs GOES-19  |  Satélite: {agora.strftime('%d/%m/%Y %H:%M')} UTC",
        fontsize=9, color="lightgray", pad=6,
    )

    plt.tight_layout()

    ts = agora.strftime("%Y%m%d_%H%M")
    saida = Path(cfg.dir_validacao) / f"Validacao_Sipremo_{ts}.png"
    fig.savefig(saida, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOG.info(f"[PLOT] Validação salva → {saida}")
    return saida


def gerar_mapa_alertas(ds_wrf: xr.Dataset, cfg: Config = CFG) -> Path:
    """
    Gera mapa de alertas meteorológicos estratificado por altitude.
    Salva em dir_alertas e retorna o Path do arquivo.
    """
    LOG.info("[PLOT] Gerando mapa de alertas...")

    lats = ds_wrf["XLAT"]
    lons = ds_wrf["XLONG"]
    cld = ds_wrf["CLDFRA"]

    def _max_camada(ini, fim):
        sl = slice(ini, fim)
        return cld.isel(bottom_top=sl).max(dim="bottom_top")

    cld_baixa = _max_camada(*cfg.camadas_baixas)
    cld_media = _max_camada(*cfg.camadas_medias)
    cld_alta  = _max_camada(cfg.camadas_altas[0], cfg.camadas_altas[1])

    fig = plt.figure(figsize=cfg.figsize_alertas, facecolor="#0d1117")
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor="#111827")
    _estilo_mapa(ax, cfg)

    # --- Nuvens baixas (preenchido) ---
    niveis_baixas = list(cfg.limiar_alerta_baixo)
    cores_baixas = ["#4d1a00", "#cc4400", "#ff2200"]
    cf = ax.contourf(
        lons, lats, cld_baixa,
        levels=niveis_baixas,
        colors=cores_baixas,
        alpha=0.70,
        transform=ccrs.PlateCarree(),
    )

    # --- Nuvens médias (contorno tracejado amarelo) ---
    ax.contour(
        lons, lats, cld_media,
        levels=[cfg.limiar_medio],
        colors=["yellow"],
        linewidths=1.5,
        linestyles="--",
        transform=ccrs.PlateCarree(),
    )

    # --- Nuvens altas (contorno pontilhado ciano) ---
    ax.contour(
        lons, lats, cld_alta,
        levels=[cfg.limiar_alto],
        colors=["cyan"],
        linewidths=1.0,
        linestyles=":",
        transform=ccrs.PlateCarree(),
    )

    # Barra de cor para nuvens baixas
    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Fração de Nuvem Baixa", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Legenda de alertas
    legenda = [
        Patch(facecolor="#ff2200", alpha=0.8, label="ALERTA: Teto Baixo Fechado (> 70%)"),
        Patch(facecolor="#cc4400", alpha=0.8, label="Atenção: Nuvens Baixas Espessas (> 30%)"),
        Line2D([0], [0], color="yellow",  lw=1.5, ls="--", label="Nuvens Médias (> 60%)"),
        Line2D([0], [0], color="cyan",    lw=1.0, ls=":",  label="Nuvens Altas — Cirrus (> 50%)"),
    ]
    ax.legend(handles=legenda, loc="lower right", title="Legenda Operacional",
              facecolor="#1a1f2e", edgecolor="gray", labelcolor="white",
              title_fontproperties={"weight": "bold"}, fontsize=8)

    # Títulos
    fig.suptitle("SISTEMA DE ALERTA SIPREMO-WRF", fontsize=14, fontweight="bold",
                 color="white", y=0.98)
    ax.set_title(
        "Classificação Estratificada de Teto e Nebulosidade\n"
        f"Referência: {cfg.wrf_file}",
        fontsize=9, color="lightgray", pad=6,
    )

    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida = Path(cfg.dir_alertas) / f"Alerta_Sipremo_{ts}.png"
    fig.savefig(saida, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOG.info(f"[PLOT] Alertas salvo → {saida}")
    return saida


# ============================================================
# MÓDULO: PIPELINE
# ============================================================

def pipeline(modo: str = "ambos", cfg: Config = CFG) -> dict[str, Optional[Path]]:
    """
    Orquestra o pipeline completo:
      - modo='validacao' → mapa de validação (WRF + GOES)
      - modo='alertas'   → mapa de alertas estratificado
      - modo='ambos'     → ambos

    Retorna dict com as chaves 'validacao' e 'alertas' apontando
    para os Paths gerados (ou None se não solicitado / falhou).
    """
    resultados: dict[str, Optional[Path]] = {"validacao": None, "alertas": None}
    t_inicio = time.time()

    LOG.info("=" * 60)
    LOG.info(f"SIPREMO-WRF | modo={modo} | WRF={CFG.wrf_file}")
    LOG.info("=" * 60)

    cfg.criar_dirs()

    # 1. Leitura do WRF (sempre necessária)
    try:
        ds_wrf = ler_wrf(cfg)
    except ErroWRF as e:
        LOG.error(f"[WRF] FALHA CRÍTICA: {e}")
        LOG.error("Pipeline interrompido. Verifique o caminho e o arquivo WRF.")
        return resultados

    # 2. Mapa de Alertas (não depende do GOES)
    if modo in ("alertas", "ambos"):
        try:
            resultados["alertas"] = gerar_mapa_alertas(ds_wrf, cfg)
        except Exception as e:
            LOG.error(f"[ALERTAS] Falha ao gerar mapa de alertas: {e}", exc_info=True)

    # 3. Mapa de Validação (depende do GOES)
    if modo in ("validacao", "ambos"):
        try:
            caminho_goes, agora = buscar_arquivo_goes(cfg)
            ds_ir = ler_goes(caminho_goes)
            resultados["validacao"] = gerar_mapa_validacao(ds_wrf, ds_ir, agora, cfg)
        except ErroGOES as e:
            LOG.error(f"[GOES] {e}")
            LOG.warning("Mapa de validação não gerado. Satélite indisponível.")
        except Exception as e:
            LOG.error(f"[VALIDAÇÃO] Falha inesperada: {e}", exc_info=True)

    # Resumo final
    dur = time.time() - t_inicio
    LOG.info("-" * 60)
    LOG.info(f"Pipeline concluído em {dur:.1f}s")
    for nome, path in resultados.items():
        status = str(path) if path else "NÃO GERADO"
        LOG.info(f"  {nome:12s} → {status}")
    LOG.info("=" * 60)

    return resultados


# ============================================================
# ENTRYPOINT
# ============================================================

def _is_jupyter() -> bool:
    """Detecta se o código está rodando dentro do Jupyter/IPython."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except NameError:
        return False


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SIPREMO-WRF Pipeline")
    p.add_argument(
        "--modo",
        choices=["validacao", "alertas", "ambos"],
        default="ambos",
        help="Quais mapas gerar (padrão: ambos)",
    )
    p.add_argument(
        "--wrf-file",
        default=None,
        help="Nome do arquivo wrfout (substitui o padrão em Config)",
    )
    # No Jupyter, ignora argumentos desconhecidos (ex: --f=kernel.json)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _args()

    if args.wrf_file:
        CFG.wrf_file = args.wrf_file
        LOG.info(f"WRF override: {CFG.wrf_file}")

    pipeline(modo=args.modo)


# ── Atalho para rodar diretamente em células do Jupyter ──────────────
# Descomente e ajuste conforme necessário:
#
# CFG.wrf_file = "wrfout_d01_2026-04-29_00:00:00"
# pipeline(modo="ambos")   # ou "validacao" / "alertas"

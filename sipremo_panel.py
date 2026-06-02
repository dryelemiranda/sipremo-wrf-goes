
from __future__ import annotations
import io, sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr

try:
    import panel as pn
    import panel.widgets as pnw
    HAS_PANEL = True
except ImportError:
    HAS_PANEL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    pass

EXTENT = (-45.4, -40.6, -24.7, -20.2)
CAMADAS = {
    "baixas": {"slice": (0,  8),    "cor": "#E24B4A", "label": "Nuvens Baixas (0-2 km)"},
    "medias": {"slice": (9,  18),   "cor": "#EF9F27", "label": "Nuvens Medias (2-6 km)"},
    "altas":  {"slice": (19, None), "cor": "#22d3ee", "label": "Nuvens Altas (>6 km)"},
}

def _max_camada(cld, ini, fim):
    return cld.isel(bottom_top=slice(ini, fim)).max(dim="bottom_top")

def _cobertura(arr, limiar=0.3):
    return float((arr >= limiar).sum() / arr.size * 100)

def _listar_wrfouts(wrf_path):
    p = Path(wrf_path)
    return sorted(f.name for f in p.glob("wrfout_d01_*") if f.is_file())

def _ler_wrf(cfg, arquivo):
    caminho = str(Path(cfg.wrf_path) / arquivo)
    with open(caminho, "rb") as f:
        dados = f.read()
    return xr.open_dataset(io.BytesIO(dados), engine="h5netcdf").isel(Time=0)

def _projecao_goes(ds_ir):
    dat = ds_ir.goes_imager_projection
    h   = float(dat.perspective_point_height)
    proj = ccrs.Geostationary(
        central_longitude=float(dat.longitude_of_projection_origin),
        satellite_height=h, sweep_axis=str(dat.sweep_angle_axis))
    img_extent = (float(ds_ir.x.min())*h, float(ds_ir.x.max())*h,
                  float(ds_ir.y.min())*h, float(ds_ir.y.max())*h)
    return proj, img_extent

def _renderizar_mapa(ds_wrf, ds_ir, agora, camadas_ativas, limiar):
    lats = ds_wrf["XLAT"].values
    lons = ds_wrf["XLONG"].values
    cld  = ds_wrf["CLDFRA"]
    fig = plt.figure(figsize=(11, 9), facecolor="#0d1117")
    ax  = fig.add_subplot(111, projection=ccrs.PlateCarree(), facecolor="#111827")
    ax.set_extent(list(EXTENT), crs=ccrs.PlateCarree())
    if ds_ir is not None:
        proj_sat, img_extent = _projecao_goes(ds_ir)
        ax.imshow(ds_ir["CMI"], origin="upper", extent=img_extent,
                  transform=proj_sat, cmap="Greys_r", vmin=190, vmax=300,
                  interpolation="bilinear")
    handles = []
    for nome, cfg_c in CAMADAS.items():
        if not camadas_ativas.get(nome, True):
            continue
        ini, fim = cfg_c["slice"]
        dados_c  = _max_camada(cld, ini, fim)
        cor      = cfg_c["cor"]
        ax.contourf(lons, lats, dados_c, levels=[limiar, 1.01],
                    colors=[cor], alpha=0.22, transform=ccrs.PlateCarree())
        ax.contour(lons, lats, dados_c,
                   levels=[limiar, min(limiar+0.2, 0.95)],
                   colors=[cor], linewidths=[1.2, 2.0],
                   transform=ccrs.PlateCarree())
        handles.append(plt.Line2D([], [], color=cor, lw=2, label=cfg_c["label"]))
    ax.add_feature(cfeature.COASTLINE, edgecolor="#22d3ee", linewidth=1.1)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#22d3ee",
                   linewidth=0.6, alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="white",
                      alpha=0.3, linestyle="--", x_inline=False, y_inline=False)
    gl.xlabel_style = {"color": "#6b7280", "fontsize": 7}
    gl.ylabel_style = {"color": "#6b7280", "fontsize": 7}
    if handles:
        ax.legend(handles=handles, loc="lower right", facecolor="#1a2035",
                  edgecolor="#2d3748", labelcolor="#e2e8f0", fontsize=8)
    sat_str = agora.strftime("%d/%m/%Y %H:%M UTC") if agora else "sem satelite"
    ax.set_title(f"SIPREMO-WRF  |  Limiar: {limiar:.0%}  |  GOES-19: {sat_str}",
                 fontsize=9, color="#e2e8f0", pad=6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _card(label, valor, cor):
    return pn.pane.HTML(
        f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;'
        f'padding:12px 16px;text-align:center;min-width:130px">'
        f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:26px;font-weight:600;color:{cor}">{valor:.1f}'
        f'<span style="font-size:14px;color:#4b5563">%</span></div></div>',
        sizing_mode="fixed", width=155, height=82)

def iniciar(cfg, ds_ir=None, agora=None):
    pn.extension(sizing_mode="stretch_width")
    arquivos = _listar_wrfouts(cfg.wrf_path)
    _cache = {"arquivo": None, "ds": None}

    def _get_ds(arquivo):
        if _cache["arquivo"] != arquivo:
            _cache["ds"] = _ler_wrf(cfg, arquivo)
            _cache["arquivo"] = arquivo
        return _cache["ds"]

    w_arquivo = pnw.Select(
        name="Timestep WRF", options=arquivos, value=arquivos[-1], width=310)
    w_baixas = pnw.Toggle(name="Baixas", value=True, button_type="danger",  width=90)
    w_medias = pnw.Toggle(name="Medias", value=True, button_type="warning", width=90)
    w_altas  = pnw.Toggle(name="Altas",  value=True, button_type="primary", width=90)
    w_limiar = pnw.FloatSlider(
        name="Limiar", start=0.1, end=0.9, step=0.1, value=0.3, width=280)

    def _mapa(arquivo, baixas, medias, altas, limiar):
        ds = _get_ds(arquivo)
        ativas = {"baixas": baixas, "medias": medias, "altas": altas}
        png_bytes = _renderizar_mapa(ds, ds_ir, agora, ativas, limiar)
        return pn.pane.PNG(png_bytes, sizing_mode="stretch_width")

    def _metricas(arquivo):
        ds = _get_ds(arquivo)
        cld = ds["CLDFRA"]
        cards = []
        for nome, cfg_c in CAMADAS.items():
            ini, fim = cfg_c["slice"]
            pct = _cobertura(_max_camada(cld, ini, fim))
            cards.append(_card(cfg_c["label"], pct, cfg_c["cor"]))
        pct_total = _cobertura(cld.max(dim="bottom_top"))
        cards.append(_card("Cobertura total", pct_total, "#a78bfa"))
        return pn.Row(*cards, sizing_mode="stretch_width",
                      styles={"padding": "8px 0"})

    mapa_reativo = pn.bind(_mapa,
        arquivo=w_arquivo, baixas=w_baixas,
        medias=w_medias, altas=w_altas, limiar=w_limiar)
    met_reativo = pn.bind(_metricas, arquivo=w_arquivo)

    sidebar = pn.Column(
        pn.pane.HTML("<h3 style='color:#e2e8f0;margin:0 0 6px'>SIPREMO-WRF</h3>"),
        pn.pane.HTML("<p style='color:#6b7280;font-size:11px;margin:0 0 10px'>"
                     "Nebulosidade Estratificada</p>"),
        pn.layout.Divider(),
        pn.pane.HTML("<b style='color:#9ca3af;font-size:12px'>Timestep</b>"),
        w_arquivo,
        pn.layout.Divider(),
        pn.pane.HTML("<b style='color:#9ca3af;font-size:12px'>Camadas</b>"),
        pn.Row(w_baixas, w_medias, w_altas),
        pn.layout.Divider(),
        w_limiar,
        pn.layout.Divider(),
        pn.pane.HTML(
            "<div style='font-size:11px;color:#4b5563;line-height:1.8'>"
            "Vermelho = Baixas (0-2km)<br>"
            "Amarelo = Medias (2-6km)<br>"
            "Ciano = Altas (>6km)<br><br>"
            "Fundo: GOES-19 IR 10.3um</div>"),
        width=240,
        styles={"background":"#0d1117","padding":"14px",
                "border-right":"1px solid #1f2937"},
    )

    conteudo = pn.Column(
        pn.panel(met_reativo),
        pn.panel(mapa_reativo),
        styles={"background":"#0d1117","padding":"10px"},
        sizing_mode="stretch_width",
    )

    return pn.Row(sidebar, conteudo,
                  sizing_mode="stretch_width",
                  styles={"background":"#0d1117"}).servable()

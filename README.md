# SIPREMO-WRF-GOES

Dashboard interativo de nebulosidade estratificada integrando previsão numérica do tempo (WRF) com imagens de satélite em tempo real (GOES-19).

---

## Descrição

O SIPREMO-WRF-GOES é um sistema operacional de monitoramento meteorológico desenvolvido para o sudeste brasileiro, com foco no estado do Rio de Janeiro. O sistema combina saídas do modelo atmosférico WRF-ARW com imagens do satélite GOES-19 para validação e visualização interativa de camadas de nuvem por altitude.

---

## Funcionalidades

- Download automático de imagens do GOES-19 via S3 anônimo da NOAA (canal IR 10.3 µm)
- Leitura robusta de arquivos `wrfout` via memória RAM (compatível com Windows/WSL)
- Separação de nebulosidade por camada de altitude:
  - 🔴 **Nuvens Baixas** (0–2 km) — camadas 0 a 8 do modelo
  - 🟡 **Nuvens Médias** (2–6 km) — camadas 9 a 18 do modelo
  - 🔵 **Nuvens Altas** (> 6 km) — camadas 19 em diante
- Dashboard interativo com imagem GOES-19 como fundo georreferenciado
- Contornos WRF sobrepostos por camada com preenchimento e legenda
- Cards de cobertura percentual por camada
- Seletor de timestep entre todos os arquivos `wrfout` disponíveis
- Slider de limiar de fração de nuvem ajustável
- Logs estruturados com timestamp e diagnóstico de falhas
- Salvamento automático de mapas em PNG

---

## Estrutura

```
sipremo-wrf-goes/
├── sipremo_wrf.py      # Pipeline principal: leitura WRF, download GOES, mapas matplotlib
└── sipremo_panel.py    # Dashboard interativo Panel/Bokeh
```

---

## Requisitos

```bash
pip install xarray h5netcdf s3fs numpy matplotlib cartopy panel bokeh pillow
```

---

## Configuração

Edite a classe `Config` no início do `sipremo_wrf.py`:

```python
wrf_path = r"C:\caminho\para\seus\arquivos\wrfout"
wrf_file = "wrfout_d01_2026-04-29_000000"
```

---

## Uso no Jupyter

```python
import sys, importlib.util

pasta = r"C:\caminho\para\sipremo-wrf-goes"

def _load(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, arq)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod

wrf = _load("sipremo_wrf",   pasta + r"\sipremo_wrf.py")
pnl = _load("sipremo_panel", pasta + r"\sipremo_panel.py")

import panel as pn
pn.extension()

CFG = wrf.CFG
arquivos = pnl._listar_wrfouts(CFG.wrf_path)
CFG.wrf_file = arquivos[-1]

# Com GOES-19
ds_ir, agora = None, None
try:
    caminho, agora = wrf.buscar_arquivo_goes(CFG)
    ds_ir = wrf.ler_goes(caminho)
except Exception as e:
    print(f"GOES indisponível: {e}")

dashboard = pnl.iniciar(CFG, ds_ir=ds_ir, agora=agora)
pn.serve(dashboard, port=5006, show=True)
```

---

## Domínio

| Parâmetro | Valor |
|---|---|
| Modelo | WRF-ARW |
| Resolução | 3 km |
| Grade | 150 × 150 pontos |
| Longitude | -45.31° a -40.69° W |
| Latitude | -24.61° a -20.35° S |
| Projeção | Mercator |
| Satélite | GOES-19 Canal 13 (IR 10.3 µm) |

---

## Organização SIPREMO

Desenvolvido por **Dryele Miranda** para o sistema **SIPREMO** (Sistema de Previsão Meteorológica Operacional).

GitHub: [@dryelemiranda](https://github.com/dryelemiranda)

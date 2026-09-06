# Oraculo de contagem (Previsao)

Visao computacional que assiste a um video ao vivo, detecta e rastreia
objetos (carros, pessoas, etc.) e conta quantos cruzam uma linha virtual
durante uma rodada de tempo fixo. E a peca que decide o resultado de cada
mercado no estilo previsao.io.

```
fonte de video -> YOLO (deteccao) -> ByteTrack (rastreio)
   -> contagem por cruzamento de linha -> rodadas (timer) -> resultado
```

## Requisitos
- Python 3.12 (o 3.14 do sistema ainda nao tem wheels de PyTorch)
- FFmpeg no PATH (para ingerir HLS/RTSP)
- GPU NVIDIA opcional (acelera muito; funciona em CPU tambem)

## Instalacao (Windows + uv)

```bash
# cria o ambiente com Python 3.12
uv venv --python 3.12 .venv

# ativa (PowerShell)
.venv\Scripts\Activate.ps1

# 1) PyTorch com CUDA (GPU NVIDIA). Sem GPU, pule para o passo 2.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) resto das dependencias
uv pip install -r requirements.txt
```

## Uso

```bash
# live de rua no YouTube (feed publico), contando veiculos, janela ao vivo
python run_oracle.py --source "https://www.youtube.com/watch?v=SEU_LIVE" --line h --show

# feed HLS publico direto (.m3u8), gravando um mp4 anotado
python run_oracle.py --source "https://.../stream.m3u8" --record

# webcam local, contando pessoas
python run_oracle.py --source 0 --classes person --show

# rodada curta (30s) so para testar rapido
python run_oracle.py --source 0 --round-seconds 30 --show
```

Tecla `q` fecha a janela. Ajuste tudo em `config.yaml` ou pela linha de comando.

## Calibrar a linha de contagem
A linha depende do enquadramento da camera. Formas de definir em `--line`:
- `h` - horizontal no meio da imagem
- `v` - vertical no meio da imagem
- `x1,y1,x2,y2` - em pixels (ex.: `0,400,1280,400`) ou em fracoes da
  imagem entre 0 e 1 (ex.: `0,0.6,1,0.6`)

## Saidas (pasta `runs/`)
- `state.json` - estado ao vivo (~1x/seg): rodada, tempo restante, contagem.
  E o que o frontend/mercado le para mostrar o placar em tempo real.
- `results.jsonl` - 1 linha por rodada encerrada: contagem final e detalhe
  por classe. E o "valor de resolucao" que liquida o mercado.
- `annotated.mp4` - video com overlay (so quando `--record`).

## Estrutura
```
run_oracle.py        # CLI / entrypoint
config.yaml          # configuracao padrao
oracle/
  source.py          # resolve a fonte (webcam/arquivo/HLS/live do YouTube)
  app.py             # loop principal (detecta -> rastreia -> conta -> render)
  counting.py        # contagem por cruzamento de linha
  round.py           # timer e ciclo de vida da rodada
  overlay.py         # desenho do HUD, caixas e linha
```

## Proximas pecas (fora do oraculo)
Motor de mercado (order book em tempo real), carteira/PIX, KYC/AML e painel de
liquidacao. O oraculo ja expoe `state.json` e `results.jsonl` para ligar nessas
partes depois.

## App web (oraculo + mercado + video) — server.py

App unico que junta tudo e mostra no navegador:
- video anotado ao vivo (MJPEG) com contador, timer e caixas de deteccao
- livro de ofertas com moeda virtual (bots dao liquidez), comprar "Mais de X"
  / "Ate X", carteira, posicoes e liquidacao automatica ao fim de cada rodada
- calibracao da linha por clique no video e seletor veiculos/pessoas

```bash
python server.py --source "https://www.youtube.com/watch?v=SEU_LIVE" --round-seconds 60 --threshold 3 --port 8000
# abrir http://127.0.0.1:8000
```

Rotas: `/` (pagina), `/stream.mjpg` (video), `/api/state` (estado+mercado),
`POST /api/order` (comprar), `/api/line` (calibrar), `/api/classes`, `/api/reset`.

Nota: isto e um prototipo com MOEDA VIRTUAL. Dinheiro real no Brasil exige
autorizacao da SPA (Lei 14.790/2023) e nao roda em serverless (Vercel): o
oraculo precisa de um servidor/VM com GPU.

## Logica de rodada (fases) — modelo do jogo
Cada rodada tem 3 fases (tempos configuraveis):
  betting  -> apostas ABERTAS + contagem rodando        (real: ~2 min)
  running  -> apostas ENCERRADAS, contagem continua      (real: ate ~5 min)
  pause    -> rodada liquidada, mostrando o resultado    (real: ~5 min)

O alvo X da proxima rodada = contagem FINAL desta ("vai passar mais ou menos
carros/pessoas que a rodada anterior?"). Bots so cotam na janela de apostas;
depois o livro congela. Parametros:
  --betting-seconds 120 --round-seconds 300 --pause-seconds 300   # valores reais
  --betting-seconds 25  --round-seconds 70  --pause-seconds 15    # demo acelerada

"""
Rede neural (MLP em numpy) para o Bomberman.

Usada em dois lugares:
  - simulador.py / treinar.py  (treino sem Pygame)
  - ia_jogador1.py             (jogo real, main.py)

A rede só LÊ o estado do jogo (nunca modifica), então não dispara o
detector de "cheat" do main.py.
"""
import os
import numpy as np

# ---- Constantes iguais às do main.py ----
ROWS, COLS = 11, 13
TEMPO_EXPLOSAO = 4.0
MAX_BOMBAS = 5
TEMPO_PARTIDA = 180.0

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]
DIRS = {"cima": (0, -1), "baixo": (0, 1), "esquerda": (-1, 0), "direita": (1, 0)}

# ---- Arquitetura ----
RAIO = 3                      # janela 7x7 ao redor do jogador
JANELA = 2 * RAIO + 1
CANAIS = 6                    # parede, bloco, powerup, bomba, perigo, outro jogador
N_ESCALARES = 8
N_ENTRADAS = CANAIS * JANELA * JANELA + N_ESCALARES
N_OCULTA = 32
N_SAIDAS = len(ACOES)

CAMINHO_PESOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pesos.npy")


class Rede:
    """MLP: entradas -> 32 (tanh) -> 6 saídas (uma por ação)."""

    TAMANHO = N_ENTRADAS * N_OCULTA + N_OCULTA + N_OCULTA * N_SAIDAS + N_SAIDAS

    def __init__(self, vetor=None):
        if vetor is None:
            vetor = Rede.aleatoria()
        self.set_pesos(vetor)

    @staticmethod
    def aleatoria():
        v = np.empty(Rede.TAMANHO, dtype=np.float32)
        a = N_ENTRADAS * N_OCULTA
        b = a + N_OCULTA
        c = b + N_OCULTA * N_SAIDAS
        v[:a] = np.random.randn(a) * (1.0 / np.sqrt(N_ENTRADAS))
        v[a:b] = 0
        v[b:c] = np.random.randn(c - b) * (1.0 / np.sqrt(N_OCULTA))
        v[c:] = 0
        return v

    def set_pesos(self, v):
        v = np.asarray(v, dtype=np.float32)
        assert v.size == Rede.TAMANHO, "vetor de pesos com tamanho errado"
        self.vetor = v
        a = N_ENTRADAS * N_OCULTA
        b = a + N_OCULTA
        c = b + N_OCULTA * N_SAIDAS
        self.W1 = v[:a].reshape(N_ENTRADAS, N_OCULTA)
        self.b1 = v[a:b]
        self.W2 = v[b:c].reshape(N_OCULTA, N_SAIDAS)
        self.b2 = v[c:]

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def salvar(self, caminho=CAMINHO_PESOS):
        tmp = caminho + ".tmp.npy"
        np.save(tmp, self.vetor)
        for _ in range(10):
            try:
                os.replace(tmp, caminho)
                return
            except OSError:
                time.sleep(0.2)
                print("[AVISO] não consegui salvar pesos.npy desta vez; tentando na proxima geração")


def carregar_rede(caminho=CAMINHO_PESOS):
    if os.path.exists(caminho):
        return Rede(np.load(caminho))
    print(f"[AVISO] '{caminho}' não encontrado: usando rede com pesos aleatórios. "
          f"Rode treinar.py primeiro.")
    return Rede()


# ---------------------------------------------------------------------------
# Leitura do estado do jogo
# ---------------------------------------------------------------------------
def _tiles_da_explosao(mapa, b):
    """Tiles atingidos por uma bomba ainda não explodida (mesma regra do main.py)."""
    tiles = [(b.x, b.y)]
    alcance = 1 + (b.nivel - 1) * 2
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, alcance + 1):
            nx, ny = b.x + dx * i, b.y + dy * i
            if not (0 <= nx < COLS and 0 <= ny < ROWS) or mapa[ny][nx] == 2:
                break
            tiles.append((nx, ny))
            if mapa[ny][nx] == 1:
                break
    return tiles


def mapas_globais(mapa, bombas):
    """
    Canais globais (com borda de RAIO tiles marcada como parede):
      0 parede | 1 bloco quebrável | 2 powerup | 3 bomba | 4 perigo (0..1)
    Pode ser calculado uma vez por tick e reaproveitado por todos os jogadores.
    """
    H, W = ROWS + 2 * RAIO, COLS + 2 * RAIO
    G = np.zeros((5, H, W), dtype=np.float32)
    G[0] = 1.0
    m = np.array(mapa, dtype=np.int8)
    inner = (slice(RAIO, RAIO + ROWS), slice(RAIO, RAIO + COLS))
    G[0][inner] = (m == 2)
    G[1][inner] = (m == 1)
    G[2][inner] = (m == 3) | (m == 4)
    for b in bombas:
        if b.explodida:
            for fx, fy in b.fogo:
                G[4, fy + RAIO, fx + RAIO] = 1.0
        else:
            G[3, b.y + RAIO, b.x + RAIO] = 1.0
            urg = max(0.25, 1.0 - max(b.tempo_explosao, 0.0) / TEMPO_EXPLOSAO)
            for tx, ty in _tiles_da_explosao(mapa, b):
                if urg > G[4, ty + RAIO, tx + RAIO]:
                    G[4, ty + RAIO, tx + RAIO] = urg
    return G


def extrair_entradas(player, jogadores, tempo_restante, G):
    px, py = player.grid_x, player.grid_y
    janela = G[:, py:py + JANELA, px:px + JANELA]      # (5, 7, 7)
    outros_ch = np.zeros((1, JANELA, JANELA), dtype=np.float32)

    alvo, melhor = None, 1e9
    for j in jogadores:
        if j is player or not j.ativo:
            continue
        wx, wy = j.grid_x - px + RAIO, j.grid_y - py + RAIO
        if 0 <= wx < JANELA and 0 <= wy < JANELA:
            outros_ch[0, wy, wx] = 1.0
        d = abs(j.grid_x - px) + abs(j.grid_y - py)
        if d < melhor:
            melhor, alvo = d, j

    ocupado = G[3, py + RAIO, px + RAIO] > 0
    esc = np.array([
        G[4, py + RAIO, px + RAIO],
        1.0 if (len(player.bombas) < player.max_bombas and not ocupado) else 0.0,
        player.max_bombas / MAX_BOMBAS,
        (player.bomba_nivel - 1) / 3.0,
        len(player.bombas) / max(1, player.max_bombas),
        0.0,
        (alvo.grid_x - px) / COLS if alvo else 0.0,
        (alvo.grid_y - py) / ROWS if alvo else 0.0,
    ], dtype=np.float32)
    return np.concatenate([janela.ravel(), outros_ch.ravel(), esc])


def escolher_acao(rede, player, mapa, jogadores, bombas, tempo_restante, G=None, ruido=0.0):
    """Roda a rede e devolve a string da ação (já filtrando ações impossíveis)."""
    if G is None:
        G = mapas_globais(mapa, bombas)
    x = extrair_entradas(player, jogadores, tempo_restante, G)
    logits = rede.forward(x).copy()
    if ruido > 0:
        logits += np.random.randn(len(logits)) * ruido

    px, py = player.grid_x, player.grid_y
    for i, nome in enumerate(ACOES):
        if nome in DIRS:
            dx, dy = DIRS[nome]
            nx, ny = px + dx, py + dy
            if not (0 <= nx < COLS and 0 <= ny < ROWS) or mapa[ny][nx] not in (0, 3, 4) \
                    or G[3, ny + RAIO, nx + RAIO] > 0:
                logits[i] = -1e9
        elif nome == "bomba":
            if len(player.bombas) >= player.max_bombas or G[3, py + RAIO, px + RAIO] > 0:
                logits[i] = -1e9
    return ACOES[int(np.argmax(logits))]
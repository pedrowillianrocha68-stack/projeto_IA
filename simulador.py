"""
Simulador sem Pygame com as mesmas regras do main.py (1 passo = 0,1 s,
que é o tempo de movimento de um tile). Usado só para treinar a rede.
"""
import copy
import random
from rede_neural import (ROWS, COLS, TEMPO_EXPLOSAO, MAX_BOMBAS, TEMPO_PARTIDA,
                         ACOES, DIRS, RAIO,  mapas_globais, escolher_acao)
from rede_neural import _tiles_da_explosao 

DT = 0.1
TEMPO_FOGO = 0.5
PONTOS_BLOCO = 200
PONTOS_POWERUP_COLETADO = 400
PONTOS_POWERUP_DESTRUIDO = -50
PONTOS_MATAR = 1000
PONTOS_VITORIA = 10000
PROB_BOMBA, PROB_FOGO = 0.12, 0.10
BONUS_BOMBA = 30
TETO_BOMBAS = 20
BONUS_VIVO = 0.0
PENA_PERIGO = 0.5
BONUS_BOMBA_PERTO = 30
BONUS_ESCAPOU = 100
PENA_SUICIDIO = 500
BONUS_COMBO = 30
BONUS_EXPLORAR = 1
PENA_OCIOSO = 0.3

MAPA_BASE = [
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 2, 1, 2, 0, 2, 0, 2, 1, 2, 1, 2, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]
INICIOS = [(0, 0), (12, 0), (0, 10), (12, 10)]


class SimBomba:
    def __init__(self, x, y, nivel, dono):
        self.x, self.y = x, y
        self.tempo_explosao = TEMPO_EXPLOSAO
        self.explodida = False
        self.nivel = nivel
        self.tempo_fogo = 0.0
        self.fogo = []
        self.dono = dono


class SimPlayer:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.grid_x, self.grid_y = x, y
        self.max_bombas = 1
        self.bomba_nivel = 1
        self.bombas = []
        self.ativo = True


class Jogo:
    """redes: lista de 4 itens; cada um é uma Rede ou None (jogador aleatório)."""

    def __init__(self, redes, seed=None, tempo_partida=TEMPO_PARTIDA):
        self.rng = random.Random(seed)
        self.redes = redes
        self.mapa = copy.deepcopy(MAPA_BASE)
        self.jogadores = [SimPlayer(i, x, y) for i, (x, y) in enumerate(INICIOS)]
        self.bombas = []
        self.pontos = [0] * 4
        self.vivo_ticks = [0] * 4
        self.suicidou = [False] * 4
        self.ganho = [0] * 4          # pontos sem bônus de vitória e sem piso em 0
        self.bombas_postas = [0] * 4
        self.tempo_restante = tempo_partida
        self.fim = False
        self.ticks_perigo = [0] * 4 
        self.bombas_perto = [0] * 4
        self.ticks_ocioso = [0] * 4
        self.visitados = [set() for _ in range(4)]  # para contar quantos tiles cada jogador visitou
    # ------------------------------------------------------------------
    def _powerup(self):
        r = self.rng.random()
        if r < PROB_BOMBA:
            return 3
        if r < PROB_BOMBA + PROB_FOGO:
            return 4
        return 0

    def _bomba_em(self, x, y):
        return any(b.x == x and b.y == y and not b.explodida for b in self.bombas)

    def _executar(self, p, acao):
        if acao == "bomba":
            if len(p.bombas) < p.max_bombas and not self._bomba_em(p.grid_x, p.grid_y):
                b = SimBomba(p.grid_x, p.grid_y, p.bomba_nivel, p)
                from rede_neural import _tiles_da_explosao 
                alvo_tiles = set(_tiles_da_explosao(self.mapa, b))
                if any(o is not p and o.ativo and (o.grid_x, o.grid_y) in alvo_tiles
                       for o in self.jogadores):
                    self.bombas_perto[p.idx] += 1
                self.bombas.append(b)
                p.bombas.append(b)
                self.bombas_postas[p.idx] += 1
            return
        dx, dy = DIRS.get(acao, (0, 0))
        if not (dx or dy):
            return
        nx, ny = p.grid_x + dx, p.grid_y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS and self.mapa[ny][nx] in (0, 3, 4) \
                and not self._bomba_em(nx, ny):
            p.grid_x, p.grid_y = nx, ny
            if (nx, ny) not in self.visitados[p.idx]:
                self.visitados[p.idx].add((nx, ny))
                self.ganho[p.idx] += BONUS_EXPLORAR          
            t = self.mapa[ny][nx]
            if t == 3:
                p.max_bombas = min(MAX_BOMBAS, p.max_bombas + 1)
                self.mapa[ny][nx] = 0
                self.pontos[p.idx] += PONTOS_POWERUP_COLETADO
                self.ganho[p.idx] += PONTOS_POWERUP_COLETADO
            elif t == 4:
                p.bomba_nivel = min(4, p.bomba_nivel + 1)
                self.mapa[ny][nx] = 0
                self.pontos[p.idx] += PONTOS_POWERUP_COLETADO
                self.ganho[p.idx] += PONTOS_POWERUP_COLETADO

    def _explodir(self, b):
        b.explodida = True
        b.tempo_fogo = TEMPO_FOGO
        b.fogo.append((b.x, b.y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for i in range(1 + (b.nivel - 1) * 2):
                nx, ny = b.x + dx * (i + 1), b.y + dy * (i + 1)
                if not (0 <= nx < COLS and 0 <= ny < ROWS) or self.mapa[ny][nx] == 2:
                    break
                b.fogo.append((nx, ny))
                if self.mapa[ny][nx] == 1:
                    break
        for fx, fy in b.fogo:
            for o in self.bombas:
                if not o.explodida and (o.x, o.y) == (fx, fy):
                    self._explodir(o)

    # ------------------------------------------------------------------
    def passo(self):
        self.tempo_restante = max(0.0, round(self.tempo_restante - DT, 6))
        G = mapas_globais(self.mapa, self.bombas)

        for p in self.jogadores:
            if not p.ativo:
                continue
            self.vivo_ticks[p.idx] += 1
            if not p.bombas:
                self.ticks_ocioso[p.idx] += 1
            if G[4, p.grid_y + RAIO, p.grid_x + RAIO] > 0.5:
                self.ticks_perigo[p.idx] += 1
            rede = self.redes[p.idx]
            if rede is None:
                acao = self.rng.choice(ACOES)
            else:
                acao = escolher_acao(rede, p, self.mapa, self.jogadores, self.bombas,
                                     self.tempo_restante, G, ruido=0.15)
            self._executar(p, acao)

        for b in self.bombas[:]:
            if not b.explodida:
                b.tempo_explosao = round(b.tempo_explosao - DT, 6)
                if b.tempo_explosao <= 0:
                    self._explodir(b)
            else:
                b.tempo_fogo = round(b.tempo_fogo - DT, 6)
            if b.explodida and b.tempo_fogo <= 0:
                quebrados = 0
                for fx, fy in b.fogo:
                    if self.mapa[fy][fx] == 1:
                        self.mapa[fy][fx] = self._powerup()
                        self.pontos[b.dono.idx] += PONTOS_BLOCO
                        self.ganho[b.dono.idx] += PONTOS_BLOCO
                        quebrados += 1
                    elif self.mapa[fy][fx] in (3, 4):
                        self.mapa[fy][fx] = 0
                        self.pontos[b.dono.idx] += PONTOS_POWERUP_DESTRUIDO
                        self.ganho[b.dono.idx] += PONTOS_POWERUP_DESTRUIDO
                if quebrados >= 2:
                            self.ganho[b.dono.idx] += BONUS_COMBO * quebrados
                if b.dono.ativo and quebrados >=1:
                            self.ganho[b.dono.idx] += BONUS_ESCAPOU
                self.bombas.remove(b)
                if b in b.dono.bombas:
                    b.dono.bombas.remove(b)

        for b in self.bombas:
            if b.explodida and b.tempo_fogo > 0:
                for p in self.jogadores:
                    if p.ativo and (p.grid_x, p.grid_y) in b.fogo:
                        p.ativo = False
                        if b.dono is p:
                            self.suicidou[p.idx] = True
                            self.pontos[p.idx] = max(0, self.pontos[p.idx] - PONTOS_MATAR)
                            self.ganho[p.idx] -= PENA_SUICIDIO
                        else:
                            self.pontos[b.dono.idx] += PONTOS_MATAR
                            self.ganho[b.dono.idx] += PONTOS_MATAR

        vivos = [p for p in self.jogadores if p.ativo]
        if len(vivos) <= 1:
            if len(vivos) == 1:
                self.pontos[vivos[0].idx] += PONTOS_VITORIA
            self.fim = True
        if self.tempo_restante <= 0:
            self.fim = True

    def jogar(self):
        while not self.fim:
            self.passo()
        return self

    def fitness(self):
        """
        Pontos reais (blocos, powerups, kills, -1000 por suicídio) + pequeno
        incentivo por colocar bombas + mínimo bônus por sobreviver.
        Sem bônus de vitória: ele premiava ficar parado esperando os outros morrerem.
        """
        return [self.ganho[i] 
                + BONUS_BOMBA * min(self.bombas_postas[i], TETO_BOMBAS) 
                + BONUS_VIVO * self.vivo_ticks[i]
                - PENA_PERIGO * self.ticks_perigo[i]
                - PENA_OCIOSO * self.ticks_ocioso[i]
                + BONUS_BOMBA_PERTO * self.bombas_perto[i]
                for i in range(4)]
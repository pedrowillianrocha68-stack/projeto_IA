"""
Treino da rede por neuroevolução (algoritmo genético), sem Pygame.

Uso:
    python treinar.py                      # treino padrão
    python treinar.py --geracoes 300       # mais gerações
    python treinar.py --continuar          # retoma a partir do pesos.npy

Salva o melhor indivíduo em pesos.npy a cada geração.
"""
import argparse
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from rede_neural import Rede, CAMINHO_PESOS
from simulador import Jogo


def _jogar(args):
    vetores, seed, tempo_partida = args
    redes = [None if v is None else Rede(v) for v in vetores]
    return Jogo(redes, seed=seed, tempo_partida=tempo_partida).jogar().fitness()


def avaliar_geracao(pool, pop, partidas, tempo_partida, seed_base):
    """Cada indivíduo joga `partidas` partidas contra 3 outros da população."""
    n = len(pop)
    tarefas, donos = [], []
    for r in range(partidas):
        ordem = list(range(n))
        random.shuffle(ordem)
        for k in range(0, n - n % 4, 4):
            grupo = ordem[k:k + 4]
            tarefas.append(([pop[i] for i in grupo], seed_base + r * 1000 + k, tempo_partida))
            donos.append(grupo)
    resultados = pool.map(_jogar, tarefas)
    soma, cont = np.zeros(n), np.zeros(n)
    for grupo, fits in zip(donos, resultados):
        for i, f in zip(grupo, fits):
            soma[i] += f
            cont[i] += 1
    return soma / np.maximum(cont, 1)


def contra_aleatorios(pool, vetor, partidas=8, tempo_partida=180):
    """Métrica de acompanhamento: melhor rede vs 3 jogadores aleatórios."""
    tarefas = [([vetor, None, None, None], 10_000_000 + i, tempo_partida) for i in range(partidas)]
    fits = pool.map(_jogar, tarefas)
    return float(np.mean([f[0] for f in fits])), float(np.mean([f[0] > max(f[1:]) for f in fits]))


def mutar(pai, sigma, taxa):
    filho = pai.copy()
    m = np.random.rand(filho.size) < taxa
    filho[m] += np.random.randn(int(m.sum())).astype(np.float32) * sigma
    return filho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geracoes", type=int, default=150)
    ap.add_argument("--pop", type=int, default=40, help="múltiplo de 4")
    ap.add_argument("--partidas", type=int, default=3, help="partidas por indivíduo/geração")
    ap.add_argument("--tempo", type=int, default=90, help="duração das partidas de treino (s)")
    ap.add_argument("--continuar", action="store_true")
    a = ap.parse_args()

    pop_n = a.pop - a.pop % 4
    n_elite = max(2, pop_n // 5)

    if a.continuar:
        base = np.load(CAMINHO_PESOS)
        pop = [base.copy()] + [mutar(base, 0.1, 0.1) for _ in range(pop_n - 1)]
    else:
        pop = [Rede.aleatoria() for _ in range(pop_n)]

    t0 = time.time()
    with Pool(cpu_count()) as pool:
        for g in range(1, a.geracoes + 1):
            fit = avaliar_geracao(pool, pop, a.partidas, a.tempo, seed_base=g * 7919)
            ordem = np.argsort(fit)[::-1]
            melhor = pop[ordem[0]]
            Rede(melhor).salvar()

            msg = (f"Ger {g:3d}/{a.geracoes} | melhor {fit[ordem[0]]:8.1f} | "
                   f"média {fit.mean():8.1f} | {time.time() - t0:6.0f}s")
            if g % 10 == 0 or g == 1:
                pts, vit = contra_aleatorios(pool, melhor)
                msg += f" | vs aleatórios: {pts:.0f} pts, {vit * 100:.0f}% vitórias"
            print(msg, flush=True)

            sigma_ini = 0.08 if a.continuar else 0.25
            sigma = max(0.03, sigma_ini * (1-g / a.geracoes))
            elite = [pop[i] for i in ordem[:n_elite]]
            nova = list(elite)
            while len(nova) < pop_n:
                pai = elite[random.randrange(len(elite))]
                nova.append(mutar(pai, sigma, 0.05))
            pop = nova

    print(f"\nTreino concluído. Melhor rede salva em: {CAMINHO_PESOS}")


if __name__ == "__main__":
    main()
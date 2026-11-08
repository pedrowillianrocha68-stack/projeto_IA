from rede_neural import carregar_rede, escolher_acao

# Carrega os pesos treinados (pesos.npy) uma única vez ao importar o módulo
_REDE = carregar_rede()


def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    return escolher_acao(_REDE, player, mapa, jogadores, bombas, tempo_restante)
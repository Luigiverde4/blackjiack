# Diccionario con las jugagas recomendadas para cada posible combinación de cartas del jugador con la carta del dealer
estrategia = {
    'dura': {
        5:  dict.fromkeys(range(2, 12), 'P'),
        6:  dict.fromkeys(range(2, 12), 'P'),
        7:  dict.fromkeys(range(2, 12), 'P'),
        8:  dict.fromkeys(range(2, 12), 'P'),
        9:  {2: 'P', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'P', 11: 'P'},
        11: dict.fromkeys(range(2, 11), 'D') | {11: 'D*'},
        12: {2: 'P', 3: 'P', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        13: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        14: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        15: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        16: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        17: dict.fromkeys(range(2, 12), 'Q'),
        18: dict.fromkeys(range(2, 12), 'Q'),
        19: dict.fromkeys(range(2, 12), 'Q'),
        20: dict.fromkeys(range(2, 12), 'Q')
    },
    'blanda': {
        (1, 2): dict.fromkeys(range(2, 12), 'P'),
        (1, 3): dict.fromkeys(range(2, 12), 'P'),
        (1, 4): {2: 'P', 3: 'P', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (1, 5): {2: 'P', 3: 'P', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (1, 6): {2: 'P', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (1, 7): {2: 'Q', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'Q', 8: 'Q', 9: 'P', 10: 'P', 11: 'P'},
        (1, 8): dict.fromkeys(range(2, 12), 'Q'),
        (1, 9): dict.fromkeys(range(2, 12), 'Q'),
        (1, 10): dict.fromkeys(range(2, 12), 'Q')
    },
    'pares': {
        (2, 2): {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (3, 3): {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (4, 4): {2: 'P', 3: 'P', 4: 'P', 5: 'S', 6: 'S', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (5, 5): {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'P', 11: 'P'},
        (6, 6): {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (7, 7): dict.fromkeys(range(2, 9), 'S') | {9: 'P', 10: 'P', 11: 'P'},
        (8, 8): {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
        (9, 9): {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'Q', 8: 'S', 9: 'S', 10: 'Q', 11: 'Q'},
        (10, 10): dict.fromkeys(range(2, 12), 'Q'),
        (1, 1): dict.fromkeys(range(2, 12), 'S')  # As,As
    }
}

def es_blanda(c1, c2):
    ''' Indica si la jugada es blanda.
    Entrada:
        c1 (int): Carta 1 del jugador
        c2 (int): Carta 2 del jugador
    Salida:
        1 (int): Es blanda
        2 (int): No es blanda
    '''
    return 1 in (c1, c2) and (c1 + c2 <= 11)
# En Blackjack, entendemos por jugada blanda aquella en la que el jugador tiene un as que equivale a un 11 y luego puede convertirse en un uno si sobrepasa el 21.


def es_par(c1, c2):
    ''' Indica si el jugador tiene dos cartas iguales.
    Entrada:
        c1 (int): Carta 1 del jugador
        c2 (int): Carta 2 del jugador
    Salida:
        1 (int): Son pares
        2 (int): No lo son
    '''
    return c1 == c2


def tipo_mano(c1, c2):
    ''' Indica el tipo de mano según las cartas del juagador.
    Entrada:
        c1 (int): Carta 1 del jugador
        c2 (int): Carta 2 del jugador
    Salida:
        pares (string): El jugador tiene pares
        blanda (string): El jugador tiene jugada blanda
        dura (string): El jugador tiene jugada dura
    '''
    if es_par(c1, c2):
        return 'pares'
    elif es_blanda(c1, c2):
        return 'blanda'
    else:
        return 'dura'

def valor_carta(carta):
    ''' Indica el valor de cada carta en int.
    Entrada:
        carta (string): Carta
    Salida:
        10 (int): Si la carta es J,Q,K
        11 (int): Si la carta es A
        carta (int): Si la carta es 1-9
    '''
    # Jugamos con: A = 11, J/Q/K = 10
    if carta in ['J', 'Q', 'K']:
        return 10
    elif carta == 'A':
        return 11
    else:
        return int(carta)

def mejor_jugada(carta1, carta2, carta_dealer):
    ''' Indica la mejor jugada posible dadas las cartas del jugador y la del dealer.
    Entrada:
        carta1 (string): Carta 1 del jugador
        carta2 (string): Carta 2 del jugador
        carta_dealer (string): Carta del dealer
    Salida:
        P (string): Pedir
        Q (string): Quedarse
        S (string): Separar
    '''
    v1 = valor_carta(carta1)
    v2 = valor_carta(carta2)
    v_d = valor_carta(carta_dealer)

    tipo = tipo_mano(v1, v2)

    if tipo == 'pares':
        clave = (v1, v2)
    elif tipo == 'blanda':
        clave = (min(v1, v2), max(v1, v2))
    else:
        clave = v1 + v2

    jugada = estrategia.get(tipo, {}).get(clave, {}).get(v_d, 'P')
    return jugada

# Ejemplo
print(mejor_jugada('A', '9', '6')) 
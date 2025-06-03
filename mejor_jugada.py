# Diccionario con las jugagas recomendadas para cada posible combinación de cartas del jugador con la carta del dealer
import numpy as np
estrategia = {
    'dura': {
        5:  dict.fromkeys(range(2, 12), 'P'),
        6:  dict.fromkeys(range(2, 12), 'P'),
        7:  dict.fromkeys(range(2, 12), 'P'),
        8:  dict.fromkeys(range(2, 12), 'P'),
        9:  {2: 'P', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'P', 11: 'P'},
        11: dict.fromkeys(range(2, 11), 'D') | {11: 'D'},
        12: {2: 'P', 3: 'P', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        13: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        14: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        15: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        16: {2: 'Q', 3: 'Q', 4: 'Q', 5: 'Q', 6: 'Q', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        17: dict.fromkeys(range(2, 12), 'Q'),
        18: dict.fromkeys(range(2, 12), 'Q'),
        19: dict.fromkeys(range(2, 12), 'Q'),
        20: dict.fromkeys(range(2, 12), 'Q'),
        21: dict.fromkeys(range(2,12),'Q')
    },
    'blanda': {
        (11, 2): dict.fromkeys(range(2, 12), 'P'),
        (11, 3): dict.fromkeys(range(2, 12), 'P'),
        (11, 4): {2: 'P', 3: 'P', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (11, 5): {2: 'P', 3: 'P', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (11, 6): {2: 'P', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
        (11, 7): {2: 'Q', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'Q', 8: 'Q', 9: 'P', 10: 'P', 11: 'P'},
        (11, 8): dict.fromkeys(range(2, 12), 'Q'),
        (11, 9): dict.fromkeys(range(2, 12), 'Q'),
        (11, 10): dict.fromkeys(range(2, 12), 'Q')
    }
}


# En Blackjack, entendemos por jugada blanda aquella en la que el jugador tiene un as que equivale a un 11 y luego puede convertirse en un uno si sobrepasa el 21.


def valor_cartas(cartas):
    ''' Indica el valor de cada carta en int. Tiene en cuenta si la jugada es blanda o no
    Entrada:
        cartas (array de strings): Array con las cartas del jugador o dealer en strings (["four of spades","nine of clubs"])
    Salida:
        cartas_int (array de int): Array con las cartas del jugador o dealer en entero ([4 , 9])
    '''
    hay_as=False
    es_blanda = False
    dict_valores= {"one" : 1 ,"two": 2, "three": 3,
                    "four" : 4, "five": 5,
                    "six": 6, "seven" : 7, 
                    "eight" : 8, "nine" : 9, "ten" : 10,
                    "joker": 0}
    cartas_int=[]
    for v in cartas:
        num= v.split()[0]
        # Jugamos con: A = 11, J/Q/K = 10
        if num in ['jack', 'queen', 'king']:
            cartas_int.append(10)
        elif num == 'ace' and not hay_as:
            cartas_int.append(11)
            hay_as = True
        elif num == "ace" and hay_as:
            cartas_int.append(1)
        elif num in dict_valores:
            cartas_int.append(dict_valores[num])
        else:
            raise ValueError(f"Carta no reconocida: '{num}'")

    es_blanda = (np.sum(cartas_int) <= 21) and (11 in cartas_int)

    if not es_blanda and 11 in cartas_int:
        # Convertimos el as de 11 a 1 si ya no puede ser blanda
        ind_as = cartas_int.index(11)
        cartas_int[ind_as] = 1

    return cartas_int, es_blanda


def mejor_jugada(cartas_jugador, carta_dealer):
    ''' Indica la mejor jugada posible dadas las cartas del jugador y la del dealer.
    Entrada:
        cartas_jugador (array de strings): Array con todas las cartas del jugador
        carta_dealer (int): Carta del dealer
    Salida:
        P (string): Pedir
        Q (string): Quedarse
        D (string): Pedir y doblar apuesta
        F (string): Se acabó el juego (el jugador tiene +21)
    '''

    clave = 0
    cartas_int_jugador, es_blanda= valor_cartas(cartas_jugador)

    carta_int_dealer,es_blanda_dealer = valor_cartas(carta_dealer)
    if es_blanda:
        tipo = "blanda"
        clave = (11, np.sum(cartas_int_jugador)-11)
    else:
        tipo = "dura"
        clave = np.sum(cartas_int_jugador)
    print(clave)
    jugada = estrategia.get(tipo, {}).get(clave, {}).get(carta_int_dealer[0], 'P')
    if np.sum(cartas_int_jugador)>21:
        jugada = "F"
    return jugada, np.sum(cartas_int_jugador), np.sum(carta_int_dealer)

# Ejemplo
# print(mejor_jugada(["seven of hearts","four of hearts"],["four of spades"])) 
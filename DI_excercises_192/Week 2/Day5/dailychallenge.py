"""
Week 2 - Day 5 - Daily Challenge
Exercise 1: Quiz (OOP concepts)
Exercise 2: Deck of cards (Card + Deck classes)
"""

import random


# ============================================================
# Exercise 1: Quiz
# ============================================================

QUIZ_ANSWERS = {
    "What is a class?":
        "Un molde/plantilla que define atributos y métodos. Es el blueprint a "
        "partir del cual se crean los objetos.",

    "What is an instance?":
        "Un objeto concreto creado a partir de una clase. Tiene sus propios "
        "valores para los atributos definidos en la clase.",

    "What is encapsulation?":
        "Agrupar datos (atributos) y comportamiento (métodos) dentro de una "
        "clase, controlando el acceso desde afuera (ej. con _ o __ en Python).",

    "What is abstraction?":
        "Exponer solo lo esencial de un objeto y ocultar los detalles internos. "
        "El usuario interactúa con la interfaz sin importarle la implementación.",

    "What is inheritance?":
        "Mecanismo donde una clase (hija) hereda atributos y métodos de otra "
        "(padre), permitiendo reutilizar y extender comportamiento.",

    "What is multiple inheritance?":
        "Cuando una clase hereda de más de una clase padre simultáneamente. "
        "Python lo soporta nativamente (a diferencia de Java).",

    "What is polymorphism?":
        "Que distintas clases puedan responder al mismo método de formas "
        "diferentes. Ej: len() funciona en strings, listas y dicts.",

    "What is method resolution order or MRO?":
        "El orden en el que Python busca un método cuando hay herencia "
        "(especialmente múltiple). Se calcula con el algoritmo C3 y se "
        "puede consultar con ClassName.__mro__.",
}


def print_quiz():
    print("=== Exercise 1: OOP Quiz ===\n")
    for i, (q, a) in enumerate(QUIZ_ANSWERS.items(), 1):
        print(f"{i}. {q}")
        print(f"   -> {a}\n")


# ============================================================
# Exercise 2: Deck of cards
# ============================================================

class Card:
    SUITS = ("Hearts", "Diamonds", "Clubs", "Spades")
    VALUES = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")

    def __init__(self, suit, value):
        if suit not in self.SUITS:
            raise ValueError(f"Suit inválido: {suit}")
        if value not in self.VALUES:
            raise ValueError(f"Value inválido: {value}")
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.value} of {self.suit}"


class Deck:
    def __init__(self):
        self.cards = self._build_full_deck()

    @staticmethod
    def _build_full_deck():
        return [Card(suit, value) for suit in Card.SUITS for value in Card.VALUES]

    def shuffle(self):
        self.cards = self._build_full_deck()
        random.shuffle(self.cards)

    def deal(self):
        if not self.cards:
            return None
        return self.cards.pop()

    def __len__(self):
        return len(self.cards)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print_quiz()

    print("=== Exercise 2: Deck of cards ===\n")
    deck = Deck()
    print(f"Cartas iniciales en el mazo: {len(deck)}")

    deck.shuffle()
    print("Mazo barajado.\n")

    print("Repartiendo 5 cartas:")
    for _ in range(5):
        card = deck.deal()
        print(f"  - {card}")

    print(f"\nCartas restantes en el mazo: {len(deck)}")

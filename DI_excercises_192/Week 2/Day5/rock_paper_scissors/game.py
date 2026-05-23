"""
Rock Paper Scissors — game logic.
"""

import random


class Game:
    VALID_ITEMS = ("rock", "paper", "scissors")

    def get_user_item(self):
        while True:
            choice = input("Elige tu jugada (rock/paper/scissors): ").strip().lower()
            if choice in self.VALID_ITEMS:
                return choice
            print(f"Opción inválida. Usa: {', '.join(self.VALID_ITEMS)}.")

    def get_computer_item(self):
        return random.choice(self.VALID_ITEMS)

    def get_game_result(self, user_item, computer_item):
        if user_item == computer_item:
            return "draw"
        wins = {
            "rock": "scissors",
            "paper": "rock",
            "scissors": "paper",
        }
        if wins[user_item] == computer_item:
            return "win"
        return "loss"

    def play(self):
        user_item = self.get_user_item()
        computer_item = self.get_computer_item()
        result = self.get_game_result(user_item, computer_item)

        print(f"\nTú elegiste: {user_item}")
        print(f"La computadora eligió: {computer_item}")
        if result == "win":
            print("¡Ganaste!")
        elif result == "draw":
            print("Empate.")
        else:
            print("Perdiste.")

        return result

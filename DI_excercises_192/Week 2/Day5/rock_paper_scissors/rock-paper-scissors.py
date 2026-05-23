"""
Rock Paper Scissors — menu, score tracking, summary.
"""

from game import Game


def get_user_menu_choice():
    print("\n=== Rock Paper Scissors ===")
    print("(g) Play a new game")
    print("(s) Show scores")
    print("(q) Quit")

    while True:
        choice = input("Tu elección: ").strip().lower()
        if choice in ("g", "s", "q"):
            return choice
        print("Opción inválida. Usa g, s o q.")


def print_results(results):
    print("\n--- Resumen ---")
    print(f"Wins: {results['win']}, Losses: {results['loss']}, Draws: {results['draw']}")
    print("¡Gracias por jugar!")


def main():
    results = {"win": 0, "loss": 0, "draw": 0}

    while True:
        choice = get_user_menu_choice()

        if choice == "g":
            game = Game()
            outcome = game.play()
            results[outcome] += 1
        elif choice == "s":
            print(f"\nScores actuales — Wins: {results['win']}, Losses: {results['loss']}, Draws: {results['draw']}")
        elif choice == "q":
            print_results(results)
            break


if __name__ == "__main__":
    main()

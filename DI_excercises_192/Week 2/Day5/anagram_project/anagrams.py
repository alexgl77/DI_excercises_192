"""
Anagrams UI — menu-driven interface using AnagramChecker.
"""

from anagram_checker import AnagramChecker


def get_user_word():
    raw = input("Ingresa una palabra: ").strip()

    if not raw:
        return None, "Error: no ingresaste nada."
    if len(raw.split()) > 1:
        return None, "Error: solo se permite una palabra (sin espacios)."
    if not raw.isalpha():
        return None, "Error: solo se permiten caracteres alfabéticos (sin números ni símbolos)."

    return raw, None


def show_menu():
    print("\n=== Anagram Checker ===")
    print("1. Ingresar una palabra")
    print("2. Salir")


def main():
    checker = AnagramChecker()

    while True:
        show_menu()
        choice = input("Elige una opción (1/2): ").strip()

        if choice == "2":
            print("¡Hasta luego!")
            break
        if choice != "1":
            print("Opción inválida. Intenta de nuevo.")
            continue

        word, error = get_user_word()
        if error:
            print(error)
            continue

        valid = checker.is_valid_word(word)
        anagrams = checker.get_anagrams(word) if valid else []

        print("\n" + "-" * 40)
        print(f'YOUR WORD: "{word.upper()}"')
        if valid:
            print("This is a valid English word.")
            if anagrams:
                print(f"Anagrams for your word: {', '.join(anagrams)}.")
            else:
                print("Anagrams for your word: (ninguno encontrado).")
        else:
            print("This is NOT a valid English word.")
        print("-" * 40)


if __name__ == "__main__":
    main()

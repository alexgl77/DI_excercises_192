"""
Week 2 - Day 4 - Exercises XP
Exercise 1: Random Sentence Generator
Exercise 2: Working with JSON
"""

import json
import os
import random
import sys


# ----------------------------- Exercise 1 -----------------------------

WORDS_FILE = os.path.join(os.path.dirname(__file__), "words.txt")


def get_words_from_file(file_path):
    """Lee el archivo y devuelve una lista de palabras."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.split()


def get_random_sentence(length):
    """Genera una oración aleatoria de 'length' palabras en minúsculas."""
    words = get_words_from_file(WORDS_FILE)
    chosen = [random.choice(words) for _ in range(length)]
    sentence = " ".join(chosen).lower()
    return sentence


def main():
    print("Generador de oraciones aleatorias.")
    print("Te pediré un número de palabras (entre 2 y 20) y crearé una oración al azar.")

    user_input = input("¿Cuántas palabras quieres en la oración? ").strip()

    try:
        length = int(user_input)
    except ValueError:
        print("Error: debes ingresar un número entero.")
        sys.exit(1)

    if length < 2 or length > 20:
        print("Error: el número debe estar entre 2 y 20 (inclusive).")
        sys.exit(1)

    sentence = get_random_sentence(length)
    print(f"\nOración generada:\n{sentence}")


# ----------------------------- Exercise 2 -----------------------------

sampleJson = """{
   "company":{
      "employee":{
         "name":"emma",
         "payable":{
            "salary":7000,
            "bonus":800
         }
      }
   }
}"""


def exercise_json():
    data = json.loads(sampleJson)

    salary = data["company"]["employee"]["payable"]["salary"]
    print(f"Salary anidado: {salary}")

    data["company"]["employee"]["birth_date"] = "1995-08-14"

    output_path = os.path.join(os.path.dirname(__file__), "employee_modified.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"JSON modificado guardado en: {output_path}")


# ----------------------------- Run -----------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Random Sentence Generator ===")
    main()
    print("\n=== Exercise 2: Working with JSON ===")
    exercise_json()

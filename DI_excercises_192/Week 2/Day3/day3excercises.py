#Exercise1
import random
import json
import os


def get_words_from_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content.split()


def get_random_sentence(length):
    words_path = os.path.join(os.path.dirname(__file__), "words.txt")
    words = get_words_from_file(words_path)
    chosen = [random.choice(words) for _ in range(length)]
    return " ".join(chosen).lower()


def main():
    try:
        length = int(input("Enter sentence length (2-20): "))
    except ValueError:
        print("Invalid input! Please enter a number.")
        return

    if length < 2 or length > 20:
        print("Please enter a number between 2 and 20.")
        return

    sentence = get_random_sentence(length)
    print(f"Generated sentence: {sentence}")


main()


# Exercise 2: Working with JSON

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

# Step 1: Parse the JSON string
data = json.loads(sampleJson)

# Step 2: Print the salary
salary = data["company"]["employee"]["payable"]["salary"]
print(f"Salary: {salary}")

# Step 3: Add birth_date to employee
data["company"]["employee"]["birth_date"] = "1990-05-15"

# Step 4: Save to file
with open("employee.json", "w") as f:
    json.dump(data, f, indent=2)
print("Modified data saved to employee.json")

# Step 5: Read back and verify
with open("employee.json", "r") as f:
    verified = json.load(f)
print(f"Verified — birth_date: {verified['company']['employee']['birth_date']}")

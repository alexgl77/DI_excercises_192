# Exercise 1: List of tuples to dictionary (comprehension)
my_list = [("name", "Elie"), ("job", "Instructor")]
person_dict = {key: value for key, value in my_list}
print(person_dict)

# Exercise 2: Two lists to dictionary (comprehension)
states_ab = ["Ca", "NY", "RI"]
states_names = ["California", "New York", "Rhode Island"]
states_dict = {key: value for key, value in zip(states_ab, states_names)}
print(states_dict)

# Exercise 3: Vowels with value 0
vowel_map = {x: 0 for x in "aeiou"}
print(vowel_map)

# Exercise 4: Alphabet dictionary
alphabet_dict = {x: chr(x + 64) for x in range(1, 27)}
print(alphabet_dict)

# Exercise 5: Count vowels only
string = "awesome sauce"
vowel_count = {x: string.count(x) for x in "aeiou"}
print(vowel_count)

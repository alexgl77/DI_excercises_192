# Exercise 1: 
my_list = [("name", "Elie"), ("job", "Instructor")]
person_dict = {key: value for key, value in my_list}
print(person_dict)

# Exercise 2: 
states_ab = ["Ca", "NY", "RI"]
states_names = ["California", "New York", "Rhode Island"]
states_dict = {key: value for key, value in zip(states_ab, states_names)}
print(states_dict)

# Exercise 3: 
vowel_map = {x: 0 for x in "aeiou"}
print(vowel_map)

# Exercise 4: 
alphabet_dict = {x: chr(x + 64) for x in range(1, 27)}
print(alphabet_dict)

# Exercise 5: 
string = "awesome sauce"
vowel_count = {x: string.count(x) for x in "aeiou"}
print(vowel_count)

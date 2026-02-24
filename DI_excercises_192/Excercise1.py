#Excercise 1


first = "Hello World" #this is a comment 
print(first) 
message = "I AM A COMPUTER!" 
print(message) 
if 1 < 2 and 4 > 2:
    print("Math is fun!")
else:
    print("Math is not fun!")

nope = None
print(nope)

result = True and False
print(result)

length = len("What´s my lenght?")
print(length)

string = "i am shouting"
uppercase_string = string.upper()
print(uppercase_string)

# Convert the string "1000" to the number 1000
number = int("1000")
print(number)

# Combine the number 4 with the string "real" to produce "4real"
combined = str(4) + "real"
print(combined)

# Record the output of 3 * "cool"
cool_output = 3 * "cool"
print(cool_output)

# Record the output of 1 / 0 (causes error)
try:
    division_output = 1 / 0
except ZeroDivisionError:
    division_output = "Error: Cannot divide by zero"
print(division_output)

# Determine the type of []
list_type = type([])
print(list_type)

# Ask the user for their name
name = input("What is your name? ")
print("Hello, " + name)

# Ask the user for a number and check if negative, positive, or zero
user_number = int(input("Enter a number: "))
if user_number < 0:
    print("That number is less than 0!")
elif user_number > 0:
    print("That number is greater than 0!")
else:
    print("You picked 0!")

# Find the index of "l" in "apple"
l_index = "apple".index("l")
print(l_index)

# Check whether "y" is in "xylophone"
y_in_xylophone = "y" in "xylophone"
print(y_in_xylophone)

# Check whether my_string is all in lowercase
my_string = "hello world"
is_lowercase = my_string.islower()
print(is_lowercase)

#Excercise 2 

def calculate_years(human_years):
    if human_years == 1:
        cat_years = 15
        dog_years = 15
    elif human_years == 2:
        cat_years = 15 + 9
        dog_years = 15 + 9
    else:
        cat_years = 15 + 9 + (human_years - 2) * 4
        dog_years = 15 + 9 + (human_years - 2) * 5


    return [human_years, cat_years, dog_years]

print(calculate_years(10))
print(calculate_years(1))
print(calculate_years(2))

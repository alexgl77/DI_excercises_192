#ex 1
my_fav_numbers = {1,2,7,11,14,21,44}
my_fav_numbers.add(3)
my_fav_numbers.add(22)
my_fav_numbers.discard(22)
friend_fav_numbers = {2,3,4,5,6,7,8}
our_fav_numbers = my_fav_numbers.union(friend_fav_numbers)
print(our_fav_numbers)

#ex2
my_tuple = (1,2,3,4)
#my_tupple_append(11) #tuples are immutable

my_tuple = my_tuple + (11,29)
print(my_tuple)

#ex3

basket = ["Banana", "Apples", "Oranges", "Blueberries"]
basket.remove("Banana")
basket.remove("Blueberries")
basket.append("Kiwi")
basket.insert(0,"Apples")
basket.count("Apples")
basket.clear()
print(basket)

#Ex4

#float is a number with a decimal point while an integer is a whole number withou a decimal point 
my_sec2 = [i/2 for i in range(3, 11)]

#Ex5

for i in range(1,21):
    print(i)

for i in range(1,21):
    if i % 2 == 0:
        print(i)


#Ex6
while True:
    user_name = input("Enter your name: ")
    if user_name.isdigit() or len(user_name)<3:
        print("Give the correct name")
    else:
        print("Thank you")
        break

#Ex7
fav_fruits = input("Enter your favourite fruits (separated by commas): ")
chosen_fruit = input("Enter any fruit name: ")
if chosen_fruit in fav_fruits:
    print("You chose one of your favourite fruits! Enjoy!")
else:
    print("You chose a new fruit. I hope you enjoy it too!")

#Ex8
toppings = []
while True:
    topping = input("Enter a topping (or 'quit' to stop): ")
    if topping == "quit":
        break
    toppings.append(topping)
    print(f"Adding {topping} to your pizza.")

total = 10 + 2.50 * len(toppings)
print(f"Your toppings: {toppings}")
print(f"Total cost: ${total}")

#Ex9
num_people = int(input("How many people in your family? "))
total_cost = 0
for i in range(num_people):
    age = int(input(f"Enter the age of person {i+1}: "))
    if age < 3:
        total_cost += 0
    elif age <= 12:
        total_cost += 10
    else:
        total_cost += 15
print(f"Total ticket cost: ${total_cost}")

#Ex9 Bonus
attendees = []
while True:
    age = input("Enter age (or 'quit' to stop): ")
    if age == "quit":
        break
    age = int(age)
    if 16 <= age <= 21:
        attendees.append(age)
    else:
        print(f"Age {age} is not allowed to watch.")
print(f"Final list of attendees: {attendees}")

print( "Hello to python")

greeting = "hello class"
print(type(greeting))
print(greeting.upper())
print(greeting.lower())


test = "hello, world"
print(test.replace("world", "python"))

print(test.count("1"))


my_age = 27
temp = 25

price = 19.99
 
is_raining = False
is_sunny = True
print( temp, price, is_raining, is_sunny)

print(not True)
print(not False)

#string formatting
first = "John"
last = "Doe"

text1 = "Hello,"+ " " + first + " " + last
print(text1)

text3 = "Hello, {} {}".format(first, last)

text4 = f"Hello, {first} {last}"
print(text4)


price = 19.99
quantity = 3
total = f"Total: ${price * quantity}"
print(total)

pi = 3.14159
print(f"{pi: .3f}")

has_licesnse = True

if not has_licesnse:
    print("No")
else:
    print("Yes")

hobbies = "coding, gaming, running" 
if "coding" in hobbies:
    print("I love coding!")   

status = "adult" if my_age >= 18 else "minor"
print(status)
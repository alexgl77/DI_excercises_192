my_list=[("name","Elie"),("job", "Instructor")]
result = dict(my_list)
print(result)

states_ab = ["Ca", "NY", "RI"]
states_names = ["California", "New York", "Rhode Island"]
zipmethod = zip(states_ab, states_names)
print(dict(zipmethod))

vowels = ["a","e","i","o","u"]
values0 = [0,0,0,0,0]
result2 = {x: 0 for x in vowels}
print(result2)

result3 = {x : chr(x+64)for x in range(1,27)}
print(result3)

string = "awesome sauce"
result4 = {x: string.count(x) for x in string}
print(result4)
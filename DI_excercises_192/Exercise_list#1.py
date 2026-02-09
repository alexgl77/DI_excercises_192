my_list = [1,2,3,4]
print(my_list)
result = [x * 20 for x in my_list]
print(result)
list_2 = ["Elie", "Tim", "Matt"]
first_letters = [letter[0] for letter in list_2]
print(first_letters)
list_3 = [1,2,3,4,5,6]
even_numbers = [x for x in list_3 if x % 2 == 0]
print(even_numbers)
list_4 = [1,2,3,4]
List_5 = [3,4,5,6]
result1 = [x for x in list_4 if x in List_5]
print(result1)

names = ["Elie", "Tim", "Matt"]
result2 = [name[::-1].lower() for name in names]
print(result2)

string1 = "first"
string2 = "third"
result3 = [x for x in string1 if x in string2]
print(result3)

list6 = range(1,101)
result4 = [x for x in range(1,101) if x % 12 == 0]
print(result4)

string3 = "amazing"
result5 = [x for x in string3 if x not in "aeiou"]   
print(result5)

list7 = [0,1,2]
result6= [list7[:]for _ in range(3)]
print(result6)

result7= [list(range(10)) for _ in range(10)]
print(result7)
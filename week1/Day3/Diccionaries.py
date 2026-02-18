#ex1

keys = ['Ten', 'Twenty', 'Thirty']
values = [10, 20, 30]

my_dict= dict(zip(keys, values))
print(my_dict)

#ex2
family = {"rick": 43, 'beth': 13, 'morty': 5, 'summer': 8}
total_cost = 0
for key, value in family.items():
    if value < 3:
        print(f"{key} Does not pay")
        total_cost += 0
    elif value <= 12:
        print(f"{key} pays 10 dollars")
        total_cost += 10
    else:
        print(f"{key} pays 15 dollars")
        total_cost += 15
print(f"The total cost for the family is {total_cost} dollars")

#ex3

Brand = {
    "name": "Zara",
    "creation_date": 1975,
    "creator_name": "Amancio Ortega Gaona",
    "type_of_clothes": ["men", "women", "children", "home"],
    "international_competitors": ["Gap", "H&M", "Benetton"],
    "number_stores": 7000,
    "major_color": {
        "France": "blue",
        "Spain": "red",
        "US": ["pink", "green"] }
}

Brand["number_stores"] = 2
print(f"Zara clients can choose the following categories:{Brand['type_of_clothes']}")
Brand["country_creation"] = "Spain"
if "international_competitors" in Brand:
    Brand["international_competitors"].append("Desigual")
del Brand["creation_date"]
print(Brand["international_competitors"][-1])
print(Brand["major_color"]["US"])
print(len(Brand.keys()))
print(list(Brand.keys()))

#Ex4
users = ["Mickey", "Minnie", "Donald", "Ariel", "Pluto"]
dict0 = {}
for i, user in enumerate(users):
    dict0[user] = i
print(dict0)

dict1 = {user: i for i, user in enumerate(users)}
print(dict1)

dict2 = dict(zip(users, range(len(users))))
print(dict2)

dict3= {i : user for i,user in enumerate(users)}
print(dict3)

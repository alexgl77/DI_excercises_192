#ex1

class Cat:
    def __init__(self, cat_name, cat_age):
        self.name = cat_name
        self.age = cat_age

Cat1 = Cat("FLuffy", 3)
Cat2 = Cat("Jasper", 5)
Cat3 = Cat("Mittens", 2)
cats = [Cat1, Cat2, Cat3]

def oldest_cat(cats):
    return max(cats, key=lambda cat: cat.age)

oldest = oldest_cat(cats)
print(f"The oldest cat is {oldest.name} and is {oldest.age} years old.")

def find_oldest_cat(Cat1, Cat2, Cat3):
    max_age = max(Cat1.age, Cat2.age, Cat3.age)
    if max_age == Cat1.age:
        return Cat1
    elif max_age == Cat2.age:
        return Cat2
    else:
        return Cat3

print(f"The oldest cat is {find_oldest_cat(Cat1, Cat2, Cat3).name} and is {find_oldest_cat(Cat1, Cat2, Cat3).age} years old.")

#ex2

class Dog:
    def __init__(self, name, height):
        self.name = name
        self.height = height
    def bark(self):
        return f"{self.name} goes woof!"
    def jump(self):
        return f"{self.name} jumps {self.height * 2} cm high!"

davids_dog = Dog("Rex", 50)
sarahs_dog = Dog("Bella", 35)

print(f"Davids dog: {davids_dog.name}, {davids_dog.height} cm")
print(davids_dog.bark())
print(davids_dog.jump())
print(f"Sarahs dog: {sarahs_dog.name}, {sarahs_dog.height} cm")
print(sarahs_dog.bark())
print(sarahs_dog.jump())

if davids_dog.height > sarahs_dog.height:
    print(f"{davids_dog.name} is bigger")
elif sarahs_dog.height > davids_dog.height:
    print(f"{sarahs_dog.name} is bigger")
else:
    print("Both dogs are the same height")

#ex3

class Song:
    def __init__(self, lyrics):
        self.lyrics = lyrics
    def sing_me_a_song(self):
        for line in self.lyrics:
            print(line)

stairway = Song(["There's a lady who's sure", "all that glitters is gold", "and she's buying a stairway to heaven."])
stairway.sing_me_a_song()

#ex4

class Zoo:
  def __init__(self, zoo_name):
    self.name = zoo_name
    self.animals = [] # a default attribute

  def add_animal(self, new_animal):
    if new_animal not in self.animals:
      self.animals.append(new_animal)

  def get_animals(self):
    print(self.animals)

  def sell_animal(self, animal_sold):
    if animal_sold in self.animals:
      self.animals.remove(animal_sold)
      
  def sort_animals(self):
    sorted_animals = {}
    for animal in self.animals:
      first_letter = animal[0]
      if first_letter not in sorted_animals:
        sorted_animals[first_letter] = []
      sorted_animals[first_letter].append(animal)
    for key in sorted(sorted_animals.keys()):
      sorted_animals[key].sort()
    return sorted_animals
  def get_groups(self):
    groups = self.sort_animals()
    for key, value in groups.items():
      print(f"{key}: {value}")



brooklyn_safari = Zoo("Brooklyn Safari")
brooklyn_safari.add_animal("Giraffe")
brooklyn_safari.add_animal("Bear")
brooklyn_safari.add_animal("Lion")
brooklyn_safari.add_animal("Baboon")
brooklyn_safari.add_animal("Cat")
brooklyn_safari.add_animal("Cougar")
brooklyn_safari.add_animal("Zebra")
brooklyn_safari.get_groups()
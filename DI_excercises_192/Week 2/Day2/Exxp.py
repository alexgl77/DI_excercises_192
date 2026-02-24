import random

# Exercise 1: Pets

class Pets:
    def __init__(self, animals):
        self.animals = animals

    def walk(self):
        for animal in self.animals:
            print(animal.walk())


class Cat:
    is_lazy = True

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def walk(self):
        return f'{self.name} is just walking around'


class Bengal(Cat):
    def sing(self, sounds):
        return f'{sounds}'


class Chartreux(Cat):
    def sing(self, sounds):
        return f'{sounds}'


class Siamese(Cat):
    def sing(self, sounds):
        return f'{sounds}'


all_cats = [Bengal("Luna", 3), Chartreux("Milo", 5), Siamese("Cleo", 2)]
sara_pets = Pets(all_cats)
sara_pets.walk()


# Exercise 2: Dogs

class Dog:
    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def bark(self):
        return f'{self.name} is barking'

    def run_speed(self):
        return self.weight / self.age * 10

    def fight(self, other_dog):
        my_power = self.run_speed() * self.weight
        their_power = other_dog.run_speed() * other_dog.weight
        if my_power > their_power:
            return f'{self.name} won the fight against {other_dog.name}'
        elif my_power < their_power:
            return f'{other_dog.name} won the fight against {self.name}'
        else:
            return "It's a tie!"


dog1 = Dog("Rex", 3, 30)
dog2 = Dog("Buddy", 5, 20)
dog3 = Dog("Max", 2, 25)

print(dog1.bark())
print(dog2.run_speed())
print(dog1.fight(dog2))


# Exercise 3: Dogs Domesticated

class PetDog(Dog):
    def __init__(self, name, age, weight):
        super().__init__(name, age, weight)
        self.trained = False

    def train(self):
        print(self.bark())
        self.trained = True

    def play(self, *args):
        names = [self.name] + [dog.name for dog in args]
        print(f'{", ".join(names)} all play together')

    def do_a_trick(self):
        if self.trained:
            tricks = ["does a barrel roll", "stands on his back legs", "shakes your hand", "plays dead"]
            print(f'{self.name} {random.choice(tricks)}')


my_dog = PetDog("Fido", 2, 10)
my_dog.train()
my_dog.play(dog1, dog2)
my_dog.do_a_trick()


# Exercise 4: Family and Person Classes

class Person:
    def __init__(self, first_name, age):
        self.first_name = first_name
        self.age = age
        self.last_name = ""

    def is_18(self):
        return self.age >= 18


class Family:
    def __init__(self, last_name):
        self.last_name = last_name
        self.members = []

    def born(self, first_name, age):
        person = Person(first_name, age)
        person.last_name = self.last_name
        self.members.append(person)

    def check_majority(self, first_name):
        for member in self.members:
            if member.first_name == first_name:
                if member.is_18():
                    print("You are over 18, your parents Jane and John accept that you will go out with your friends")
                else:
                    print("Sorry, you are not allowed to go out with your friends.")
                return

    def family_presentation(self):
        print(f'Family: {self.last_name}')
        for member in self.members:
            print(f'{member.first_name}, {member.age} years old')


my_family = Family("Smith")
my_family.born("Alice", 20)
my_family.born("Bob", 15)
my_family.check_majority("Alice")
my_family.check_majority("Bob")
my_family.family_presentation()

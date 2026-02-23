#Ex2

class Dog:
    def __init__(self, name, age, weight, breed):
        self.name = name
        self.age = age
        self.weight = weight
        self.breed = breed
    def run_speed(self):
        return (self.weight / self.age) * 10
    def fight(self, other_dog: "Dog"):
        my_speed = self.run_speed()
        their_speed = other_dog.run_speed()
        if my_speed > their_speed:
            print(f"{self.name} wins the fight against {other_dog.name}")
        elif my_speed < their_speed:
            print(f"{other_dog.name} wins the fight against {self.name}")
        else:
            print("It's a tie!")

class Dogs:
    def __init__(self):
        self.pack = []

    def add_dog(self, dog: Dog):
        self.pack.append(dog)

    def fight_all(self):
        for i in range(len(self.pack)):
            for j in range(i + 1, len(self.pack)):
                self.pack[i].fight(self.pack[j])

dog1 = Dog("Rex", 5, 20, "Labrador")
dog2 = Dog("Bella", 3, 15, "Beagle")
dog3 = Dog("Max", 4, 25, "German Shepherd")
pack = Dogs()
pack.add_dog(dog1)
pack.add_dog(dog2)
pack.add_dog(dog3)
pack.fight_all()


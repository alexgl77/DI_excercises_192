class Pet:
    is_lazy = False

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def description(self):
        print(f"{self.name} is {self.age} years old.")

    def make_sound(self):
        print("...")

class Cat(Pet):
    is_lazy = True
    def __init__(self, name: str, age: int, indoor: bool):
        super().__init__(name, age)
        self.indoor = indoor
    def make_sound(self):
        print(f"{self.name} says: Meow!")

class Dog(Pet):
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)
        self.breed = breed
    def make_sound(self):
        print(f"{self.name} says: Woof!")
    def fetch(self, item: str):
        print(f"{self.name} goes after the {item}!")         

#tests

cat1 = Cat("Whiskers", 3, True)
cat1.description()      
cat1.make_sound()
dog1 = Dog("Rex", 5, "Labrador")
dog1.description()         
dog1.make_sound()
dog1.fetch("ball")

print(Cat.is_lazy)
print(Dog.is_lazy)






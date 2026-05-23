"""
Daily Challenge - Week 2 Day 3
Circle class with OOP and dunder methods.
"""

import math


class Circle:
    def __init__(self, radius=None, diameter=None):
        if radius is None and diameter is None:
            raise ValueError("Debes especificar radius o diameter.")
        if radius is not None and diameter is not None:
            raise ValueError("Especifica solo uno: radius o diameter.")
        if radius is not None:
            self.radius = radius
        else:
            self.diameter = diameter

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("El radio no puede ser negativo.")
        self._radius = value

    @property
    def diameter(self):
        return self._radius * 2

    @diameter.setter
    def diameter(self, value):
        if value < 0:
            raise ValueError("El diámetro no puede ser negativo.")
        self._radius = value / 2

    @property
    def area(self):
        return math.pi * (self._radius ** 2)

    def __str__(self):
        return f"Circle(radius={self._radius:.2f}, diameter={self.diameter:.2f}, area={self.area:.2f})"

    def __repr__(self):
        return f"Circle(radius={self._radius})"

    def __add__(self, other):
        if not isinstance(other, Circle):
            return NotImplemented
        return Circle(radius=self._radius + other._radius)

    def __gt__(self, other):
        if not isinstance(other, Circle):
            return NotImplemented
        return self._radius > other._radius

    def __lt__(self, other):
        if not isinstance(other, Circle):
            return NotImplemented
        return self._radius < other._radius

    def __eq__(self, other):
        if not isinstance(other, Circle):
            return NotImplemented
        return self._radius == other._radius

    def __hash__(self):
        return hash(self._radius)


if __name__ == "__main__":
    c1 = Circle(radius=5)
    c2 = Circle(diameter=8)
    c3 = Circle(radius=2)

    print("Atributos de cada círculo:")
    print(c1)
    print(c2)
    print(c3)

    print("\nSuma de c1 + c2:")
    c4 = c1 + c2
    print(c4)

    print("\nComparaciones:")
    print(f"c1 > c2: {c1 > c2}")
    print(f"c1 == c3: {c1 == c3}")
    print(f"c2 < c1: {c2 < c1}")

    print("\nLista de círculos sin ordenar:")
    circles = [c1, c2, c3, c4]
    for c in circles:
        print(f"  {c}")

    print("\nLista ordenada de menor a mayor radio:")
    circles_sorted = sorted(circles)
    for c in circles_sorted:
        print(f"  {c}")

    # Bonus: dibujar los círculos ordenados con turtle
    try:
        import turtle

        screen = turtle.Screen()
        screen.title("Círculos ordenados")
        t = turtle.Turtle()
        t.speed(0)
        t.penup()
        start_x = -250
        for c in circles_sorted:
            t.goto(start_x, -c.radius)
            t.pendown()
            t.circle(c.radius)
            t.penup()
            start_x += c.diameter + 20
        t.hideturtle()
        screen.mainloop()
    except Exception as e:
        print(f"\n(Bonus turtle omitido: {e})")

import numpy as np


# Exercise 1: Array Creation and Manipulation
# 1D array con numeros de 0 a 9
ex1 = np.arange(10)
print("Exercise 1:")
print(ex1)
print()


# Exercise 2: Type Conversion and Array Operations
# Convertir lista a NumPy array y cambiar dtype a int
ex2 = np.array([3.14, 2.17, 0, 1, 2]).astype(int)
print("Exercise 2:")
print(ex2)
print()


# Exercise 3: Working with Multi-Dimensional Arrays
# Array 3x3 con valores de 1 a 9
ex3 = np.arange(1, 10).reshape(3, 3)
print("Exercise 3:")
print(ex3)
print()


# Exercise 4: Creating Multi-Dimensional Array with Random Numbers
# Array 2D shape (4, 5) con numeros aleatorios
ex4 = np.random.rand(4, 5)
print("Exercise 4:")
print(np.round(ex4, 2))
print()


# Exercise 5: Indexing Arrays
# Seleccionar la segunda fila
array_ex5 = np.array([[21, 22, 23, 22, 22],
                      [20, 21, 22, 23, 24],
                      [21, 22, 23, 22, 22]])
ex5 = array_ex5[1]
print("Exercise 5:")
print(ex5)
print()


# Exercise 6: Reversing elements
# Invertir orden de un array 1D
ex6 = np.arange(10)[::-1]
print("Exercise 6:")
print(ex6)
print()


# Exercise 7: Identity Matrix
# Matriz identidad 4x4
ex7 = np.eye(4)
print("Exercise 7:")
print(ex7)
print()


# Exercise 8: Simple Aggregate Funcs
# Suma y promedio de un array 1D
array_ex8 = np.arange(10)
suma = array_ex8.sum()
promedio = array_ex8.mean()
print("Exercise 8:")
print(f"Sum: {suma}, Average: {promedio}")
print()


# Exercise 9: Create Array and Change its Structure
# Array de 1 a 20 reshape a 4x5
ex9 = np.arange(1, 21).reshape(4, 5)
print("Exercise 9:")
print(ex9)
print()


# Exercise 10: Conditional Selection of Values
# Extraer numeros impares
array_ex10 = np.arange(1, 11)
ex10 = array_ex10[array_ex10 % 2 == 1]
print("Exercise 10:")
print(ex10)

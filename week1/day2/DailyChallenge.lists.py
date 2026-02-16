#challenge1
number = int(input("Enter a number: "))
length = int(input("Enter the length: "))
result= [number * i for i in range(1,length +1)]
print(result)

result_1= []
for i in range(1, length+1):
    result_1.append(number*i)
print(result_1)

result_2 = []
i = 1
while i <= length:
    result_2.append(number * i)
    i += 1
print(result_2)    

#challenge2
user_input = input("Enter a string: ")
result_3 = user_input[0]
for i in range(1, len((user_input))):
    if user_input[i] != user_input[i-1]:
        result_3 += user_input[i]
print(result_3)
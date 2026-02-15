user_input = input("Enter a string of exactly 10 characters long: ")
if len(user_input) < 10:
    print("The string is too short.")
elif len(user_input) > 10:
    print("The string is too long.")
else:
    print("Perfect string")
    print(f"First Character: {user_input[0]}")
    print(f"Last Character: {user_input[-1]}")
    for i in range(1, len(user_input)+1):
        print(user_input[:i])


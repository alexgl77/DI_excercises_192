import random 
def number_guessing_game():
    random_number = random.randint(1,100)
    max_attempts = 7
    
    for attempt in range (1, max_attempts + 1):
        guess = int(input("Enter your guess: "))
        if guess == random_number:
            print("You won!")
            break
        elif guess < random_number:
            print("Too low")
        else:
            print("Too high")
    else:
        print(f"You lost! The number was {random_number}")
number_guessing_game()




        

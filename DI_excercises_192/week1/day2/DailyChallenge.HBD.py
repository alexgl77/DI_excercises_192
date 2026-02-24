from datetime import date

your_bd = input("Enter your birth date (dd/mm/yyyy): ")
day = int(your_bd[0:2])
month = int(your_bd[3:5])
year = int(your_bd[6:10])

today = date.today()
age = today.year - year
if month > today.month or (month == today.month and day > today.day):
    age -= 1

candles = int(str(age)[-1])
candles_str = "i" * candles
print(f"      ___{candles_str}___")
print("      |:H:a:p:p:y:|")
print("    __|___________|__")
print("   |^^^^^^^^^^^^^^^^^|")
print("   |:B:i:r:t:h:d:a:y:|")
print("   |                 |")
print("   ~~~~~~~~~~~~~~~~~~~")
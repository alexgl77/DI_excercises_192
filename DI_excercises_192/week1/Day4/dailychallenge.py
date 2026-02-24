menu = {
    "espresso": 7.0,
    "latte": 12.0,
    "cappuccino": 10.0
}

def show_menu(menu_dict):
    if len(menu_dict) == 0:
        print("The menu is empty.")
    else:
        print("Current menu:")
        for drink, price in menu_dict.items():
            print(f"{drink} - {price}₪")

def add_item(menu_dict):
    name = input("Enter new drink name: ").lower()
    if name in menu_dict:
        print("Item already exists!")
        return
    price = float(input("Enter price: "))
    menu_dict[name] = price
    print(f'"{name}" added!')

def update_price(menu_dict):
    name = input("Which drink do you want to update? ").lower()
    if name not in menu_dict:
        print("Item not found.")
        return
    new_price = float(input("Enter the new price: "))
    menu_dict[name] = new_price
    print("Price updated!")

def delete_item(menu_dict):
    name = input("Which drink do you want to delete? ").lower()
    if name not in menu_dict:
        print("Item not found.")
        return
    del menu_dict[name]
    print("Item deleted.")

def show_options():
    print("\nWhat would you like to do?")
    print("1. Show menu")
    print("2. Add item")
    print("3. Update price")
    print("4. Delete item")
    print("5. Exit")

def run_coffee_shop():
    while True:
        show_options()
        choice = input("> ")

        if choice == "1":
            show_menu(menu)
        elif choice == "2":
            add_item(menu)
        elif choice == "3":
            update_price(menu)
        elif choice == "4":
            delete_item(menu)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

run_coffee_shop()

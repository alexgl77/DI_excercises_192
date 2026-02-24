#Challenge1
user_word = input(str("Enter a word: "))
result = {}
for i, char in enumerate(user_word):
    if char in result:
        result[char].append(i)
    else:
        result[char] = [i]
print(result)

#Challenge2
items_purchase = {"Water": "$1", "Bread": "$3", "TV": "$1,000", "Fertilizer": "$20"}
wallet = "$300"
wallet_clean = int(wallet.replace("$", "").replace(",", ""))

basket = []

for item, price in items_purchase.items():
    price = int(price.replace("$", "").replace(",", ""))
    if wallet_clean >= price:
        basket.append(item)
        wallet_clean -= price

if basket:
    print(sorted(basket))
else:
    print("Nothing in the basket")    
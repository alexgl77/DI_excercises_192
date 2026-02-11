# 1. What is a module?
# A Python file that contains functions and code you can reuse in other files.
# For example, random and math are modules.

# 2. Three ways to import a module:
import random                    # import the whole module
from random import choice        # import only one function
from random import choice as c   # import with an alias

# 3. What is the purpose of importing?
# To reuse code that already exists without having to write it from scratch.
# For example, you don't need to create your own function to generate random numbers,
# you just import random.

# 4. Three examples using random module:
# - Generate a random number for a dice game: random.randint(1, 6)
# - Pick a random winner from a list: random.choice(["Ana", "Luis", "Pedro"])
# - Create a random password: random.randrange(1000, 9999)

# 5. What is an ImportError?
# An error that occurs when you try to import something that doesn't exist:
# from random import blah  -> ImportError: no "blah" in random

# 6. When would using an OrderedDict be useful?
# When you need a dictionary that remembers the order elements were added.
# if you need to compare the order of two dictionaries.

# 7. When would using a defaultdict be useful?
# When you want a dictionary with an automatic default value for new keys,
# without having to check if the key exists:
from collections import defaultdict
d = defaultdict(int)
d["cats"] += 1   # no error even though "cats" didn't exist, starts at 0
print(d)

# 8. Purpose of if __name__ == '__main__':
# It runs the code only when you execute the file directly.
# If another file imports your functions, that block won't run.
# It separates functions from test code.

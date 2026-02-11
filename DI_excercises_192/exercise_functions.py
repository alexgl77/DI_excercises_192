import math
import re
from collections import Counter

#difference
def difference(a, b):
    """Returns the difference between two numbers."""
    return a - b

#print_day
def print_day(number, days=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]):
    """Returns the day of the week based on a number (1-7)."""
    if 1 <= number <= 7:
        return days[number - 1]
    else:
        return None

#last_element
def last_element(lst):
    """Returns the last element of a list, or None if empty."""
    if lst:
        return lst[-1]
    else:
        return None

#number_compare
def number_compare(a, b):
    """Compares two numbers and returns which is greater."""
    if a > b:
        return "First is greater"
    elif a < b:
        return "Second is greater"
    else:
        return "numbers are equal"

#single_letter_count
def single_letter_count(word, letter):
    """Counts occurrences of a letter in a word, case insensitive."""
    return word.lower().count(letter.lower())

#multiple_letter_count
def multiple_letter_count(s):
    """Returns a dictionary with the count of each letter in a string."""
    return {letter: s.count(letter) for letter in s}

#list_manipulation
def list_manipulation(lst, command, location, value=None):
    """Manipulates a list by adding/removing elements at beginning/end."""
    if command == "remove" and location == "end":
        return lst.pop()
    elif command == "remove" and location == "beginning":
        return lst.pop(0)
    elif command == "add" and location == "beginning":
        lst.insert(0, value)
        return lst
    elif command == "add" and location == "end":
        lst.append(value)
        return lst

#is_palindrome
def is_palindrome(s):
    """Returns True if the string is a palindrome, case and space insensitive."""
    s = s.replace(" ", "").lower()
    return s == s[::-1]

#frequency
def frequency(lst, search_term):
    """Returns the number of times search_term appears in the list."""
    return lst.count(search_term)

#flip_case
def flip_case(s, letter):
    """Flips the case of all occurrences of a letter in a string."""
    result = ""
    for char in s:
        if char.lower() == letter.lower():
            result += char.swapcase()
        else:
            result += char
    return result

#multiply_even_numbers
def multiply_even_numbers(lst):
    """Returns the product of all even numbers in the list."""
    result = 1
    for num in lst:
        if num % 2 == 0:
            result *= num
    return result

#mode
def mode(lst):
    """Returns the most frequent number in the list."""
    counts = Counter(lst)
    return counts.most_common(1)[0][0]

#capitalize
def capitalize(s):
    """Returns the string with the first letter capitalized."""
    return s[0].upper() + s[1:]

#compact
def compact(lst):
    """Returns a list with only truthy values."""
    return [x for x in lst if x]

#partition
def is_even(num):
    return num % 2 == 0

def partition(lst, callback):
    """Splits a list into two lists based on a callback function."""
    trues = []
    falses = []
    for x in lst:
        if callback(x):
            trues.append(x)
        else:
            falses.append(x)
    return [trues, falses]

#intersection
def intersection(lst1, lst2):
    """Returns a list of values present in both lists."""
    return [x for x in lst1 if x in lst2]

#once
def once(fn):
    """Returns a function that can only be invoked once."""
    def inner(*args):
        if not inner.has_run:
            inner.has_run = True
            return fn(*args)
        return None
    inner.has_run = False
    return inner

def add(a, b):
    return a + b

#super bonus - decorator
def run_once(fn):
    """Decorator that limits a function to a single invocation."""
    def inner(*args):
        if not inner.has_run:
            inner.has_run = True
            return fn(*args)
        return None
    inner.has_run = False
    return inner

@run_once
def add_once(a, b):
    return a + b

#Reversed strings
def reverse_string(s):
    """Returns the reversed string."""
    return s[::-1]

#looking for a benefactor
def new_avg(donations, new_average):
    """Returns the donation needed to reach a new average."""
    new_donation = new_average * (len(donations) + 1) - sum(donations)
    if new_donation <= 0:
        raise ValueError("Expected donation is not positive")
    return math.ceil(new_donation)

#sum of a sequence
def sequence_sum(begin, end, step):
    """Returns the sum of a sequence from begin to end with a given step."""
    if begin > end:
        return 0
    return sum(range(begin, end + 1, step))

#difference between largest and smallest
def max_diff(lst):
    """Returns the difference between the largest and smallest values."""
    if len(lst) <= 1:
        return 0
    return max(lst) - min(lst)

#count smileys
def count_smileys(arr):
    """Returns the count of valid smiley faces in a list."""
    if not arr:
        return 0
    return len([face for face in arr if re.match(r'^[:;][-~]?[)D]$', face)])

#count sentences in paragraph
def count_sentences(paragraph):
    """Returns the number of sentences based on '.', '?' and '!'."""
    return paragraph.count('.') + paragraph.count('?') + paragraph.count('!')

#tortoise race
def race(v1, v2, g):
    """Returns [hours, minutes, seconds] for tortoise B to catch A."""
    if v1 >= v2:
        return None
    total_seconds = int(g / (v2 - v1) * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return [hours, minutes, seconds]

#shifted string rotation
def shifted_diff(first, second):
    """Returns the number of characters shifted, or -1 if not a rotation."""
    if len(first) != len(second):
        return -1
    if first == second:
        return 0
    doubled = first + first
    if second in doubled:
        return doubled.index(second)
    return -1

if __name__ == "__main__":
    print(print_day(4))
    print(print_day(41))
    print(last_element([1, 100, 99]))
    print(number_compare(100, 1000))
    print(single_letter_count("Amazing", "a"))
    print(multiple_letter_count("awesome sauce"))
    print(list_manipulation([1, 100, 4], "add", "end", 50))
    print(is_palindrome("hannah"))
    print(frequency([1, 1, 1, 3], 0))
    print(flip_case("Hardy har har", "h"))
    print(multiply_even_numbers([200, 1, 2, 5, 1]))
    print(mode([2, 4, 1, 2, 4, 4, 4]))
    print(capitalize("alex"))
    print(partition([1, 2, 3, 4], is_even))
    print(intersection([1, 2, 3], [2, 3, 4]))
    one_addition = once(add)
    print(one_addition(2, 2))
    print(one_addition(2, 2))
    print(one_addition(12, 200))
    print(add_once(2, 2))
    print(add_once(2, 20))
    print(add_once(12, 20))
    print(reverse_string("world"))
    print(reverse_string("word"))
    print(new_avg([14, 30, 5, 7, 9, 11, 15], 92))
    print(sequence_sum(2, 2, 2))
    print(sequence_sum(2, 6, 2))
    print(sequence_sum(1, 5, 1))
    print(sequence_sum(1, 5, 3))
    print(max_diff([1, 2, 3, 4]))
    print(max_diff([1, 2, 3, -4]))
    print(max_diff([]))
    print(count_smileys([':)', ';(', ';}', ':-D']))
    print(count_smileys([';D', ':-(', ':-)', ';~)']))
    print(count_smileys([';]', ':[', ';*', ':$', ';-D']))
    print(count_sentences("Hello. How are you? I'm fine!"))
    print(race(720, 850, 70))
    print(race(80, 91, 37))
    print(shifted_diff("coffee", "eecoff"))
    print(shifted_diff("eecoff", "coffee"))
    print(shifted_diff("moose", "Moose"))
    print(shifted_diff("isn't", "'tisn"))
    print(shifted_diff("Esham", "Esham"))
    print(shifted_diff("dog", "god"))

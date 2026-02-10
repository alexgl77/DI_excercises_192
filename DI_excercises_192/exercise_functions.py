#difference
def difference(a,b):
    return a - b

#print_day
def print_day(number, days=["Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]):
    if 1 <=number <=7:
        return days[number-1]
    else:
        return None
print(print_day(4))
print(print_day(41))

#last_element
def last_element(list):
    if list:
        return list[-1]
    else:
        return None
print(last_element([1,100,99]))

#number_compare
def number_compare(a,b):
    if a > b:
        return "First is greater"
    elif a < b:
        return "Second is greater"
    else: 
        return "numbers are equal"
print(number_compare(100,1000))

#single_letter_count
def single_letter_count(word, letter):
    return word.lower().count(letter.lower())    
print(single_letter_count("Amazing", "a"))
    
#multiple_letter_count
def multiple_letter_count(string):
   return {letter: string.count(letter)for letter in string}
print(multiple_letter_count("awesome sauce"))

#list_manipulation
def list_manipulations(lst,command,location, value = None):
    if command == "remove" and location == "end":
        return lst.pop()
    elif command == "remove" and location == "beginning":
        return lst.pop(0)
    elif command == "add" and location == "beginning":
        lst.insert(0,value)
        return lst
    elif command == "add" and location == "end":
        lst.append(value)
        return lst 
print(list_manipulations([1,100,4], "add","end", 50))

#is_palicindrome
def is_palicindrome(string):
    string = string.replace(" ", "").lower()
    return string == string[::-1]
print(is_palicindrome("hannah"))

#frecuency
def frequency(lst,search_term):
    return lst.count(search_term)
print(frequency([1,1,1,3],0))

#flip_case
def flip_case(string,letter):
    result = ""
    for char in string:
        if char.lower() == letter.lower():
         result += char.swapcase()
    else:
        result += char
    return result     

#multiply_even_numbers
def multiply_even_numbers(lst):
    result = 1
    for num in lst:
        if num % 2 == 0:
            result *= num 
    return result 
print(multiply_even_numbers([200,1,2,5,1]))

#mode
def mode(lst):
    result = lst[0]
    for num in lst:
        if lst.count(num) > lst.count(result):
            result = num
    return result
print(mode([2,4,1,2,4,4,4]))        

#capitalize

def capitalize(string):
    return string[0].upper() + string[1:]
print(capitalize("alex"))

#compact
def compact(lst):
    return [x for x in lst if x]

#partition
def is_even(num):
    return num % 2 == 0

def partition(lst,callback):
    trues = []
    falses = []
    for x in lst:
        if callback(x):
            trues.append(x)
        else:
            falses.append(x)
    return [trues, falses]
print(partition([1,2,3,4], is_even))

#intersection
def intersection(lst1, lst2):
    return [x for x in lst1 if x in lst2]
print(intersection([1,2,3], [2,3,4]))

#once
def once(fn):
    def inner(*args):
        if not inner.has_run:
            inner.has_run = True
            return fn(*args)
        return None
    inner.has_run = False
    return inner

def add(a, b):
    return a + b

one_addition = once(add)
print(one_addition(2,2))
print(one_addition(2,2))
print(one_addition(12,200))

#super bonus - decorator
def run_once(fn):
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

print(add_once(2,2))
print(add_once(2,20))
print(add_once(12,20))

#Exercise 2

#Reversed strings
def reverse_string(string):
    return string[::-1]
print(reverse_string("world"))  
print(reverse_string("word"))

#looking for a benefactor
import math
def new_avg(donations,new_average):
    new_donation = new_average * (len(donations)+1) - sum(donations)
    if new_donation <= 0:
        raise ValueError("Expected donation is not positive")
    return math.ceil(new_donation)
print(new_avg([14, 30, 5, 7, 9, 11, 15], 92))

#sum of a sequence
def sequence_sum(begin, end, step):
    if begin > end:
        return 0
    return sum(range(begin, end + 1, step))

print(sequence_sum(2, 2, 2))   
print(sequence_sum(2, 6, 2))   
print(sequence_sum(1, 5, 1))   
print(sequence_sum(1, 5, 3))   

#difference between largest and smallest
def max_diff(lst):
    if len(lst) <= 1:
        return 0
    return max(lst) - min(lst)

print(max_diff([1, 2, 3, 4]))   
print(max_diff([1, 2, 3, -4]))  
print(max_diff([]))              

#count smileys
import re

def count_smileys(arr):
    if not arr:
        return 0
    return len([face for face in arr if re.match(r'^[:;][-~]?[)D]$', face)])

print(count_smileys([':)', ';(', ';}', ':-D']))       
print(count_smileys([';D', ':-(', ':-)', ';~)']))      
print(count_smileys([';]', ':[', ';*', ':$', ';-D']))  

#count sentences in paragraph
def count_sentences(paragraph):
    return paragraph.count('.') + paragraph.count('?') + paragraph.count('!')

print(count_sentences("Hello. How are you? I'm fine!")) 

#tortoise race
def race(v1, v2, g):
    if v1 >= v2:
        return None
    total_seconds = int(g / (v2 - v1) * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return [hours, minutes, seconds]

print(race(720, 850, 70))  
print(race(80, 91, 37))    

#shifted string rotation
def shifted_diff(first, second):
    if len(first) != len(second):
        return -1
    if first == second:
        return 0
    doubled = first + first
    if second in doubled:
        return doubled.index(second)
    return -1

print(shifted_diff("coffee", "eecoff"))  
print(shifted_diff("eecoff", "coffee"))  
print(shifted_diff("moose", "Moose"))   
print(shifted_diff("isn't", "'tisn"))    
print(shifted_diff("Esham", "Esham"))     
print(shifted_diff("dog", "god"))         

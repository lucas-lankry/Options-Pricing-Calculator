


# R-1.1 Write a short Python function, is multiple(n, m), that takes two integer
# values and returns True if n is a multiple of m, that is, n = mi for some
# integer i, and False otherwise.


def is_multiple(n,m):
    if n % m == 0:
        return True
    else:
        return False
    
# print(is_multiple(20,9))

# R-1.2 Write a short Python function, is even(k), that takes an integer value and
# returns True if k is even, and False otherwise. However, your function
# cannot use the multiplication, modulo, or division operators.

def is_even(k):
    return (1&k) == 0 

# print(is_even(8))

# R-1.3 Write a short Python function, minmax(data), that takes a sequence of
# one or more numbers, and returns the smallest and largest numbers, in the
# form of a tuple of length two. Do not use the built-in functions min or
# max in implementing your solution.

def minmax(data):
    smallest = data[0]
    largest = data[0]
    for i in data: 
        if i < smallest : 
            smallest = i
        elif i > largest:
            largest = i
    return (smallest, largest)
    

# print(minmax([0,1,2,3,4,5,6,7,8,9,-29,19,45]))

# R-1.4 Write a short Python function that takes a positive integer n and returns
# the sum of the squares of all the positive integers smaller than n.

def sum_sqre(n):
    total = 0
    for nb in range(1,n):
        total += nb * nb 
    return total

# print(sum_sqre(5))


# R-1.5 Give a single command that computes the sum from Exercise R-1.4, relying on Python’s
# comprehension syntax and the built-in sum function.

def sum_square(n):
    return sum(i * i for i in range(1,n))

# print(sum_square(5))

# R-1.6 Write a short Python function that takes a positive integer n and returns
# the sum of the squares of all the odd positive integers smaller than n.


def sum_square_odd(n):
    total= 0
    for i in range(1,n):
        if (1 & i == 0) is False:
            total += i*i
        else:
            total += 0
    return total

# print(sum_square_odd(5))


# R-1.7 Give a single command that computes the sum from 
# Exercise R-1.6, relying on Python’s comprehension syntax and the built-in sum function.

def sum_square_odd_function(n):
    return sum(i * i for i in range(1,n) if i % 2 != 0)

# print(sum_square_odd_function(5))

# R-1.8 Python allows negative integers to be used as indices into a sequence,
# such as a string. If string s has length n, and expression s[k] is used for 
# index −n ≤ k < 0, what is the equivalent index j ≥ 0 such that s[j] references
# the same element?
# j = n + k

# R-1.9 What parameters should be sent to the range constructor, to produce a
# range with values 50, 60, 70, 80?

# print(list(range(50, 90, 10)))

# R-1.10 What parameters should be sent to the range constructor, to produce a
# range with values 8, 6, 4, 2, 0, −2, −4, −6, −8?

# print(list(range(8, -10, -2)))

# R-1.11 Demonstrate how to use Python’s list comprehension syntax to produce
# the list [1, 2, 4, 8, 16, 32, 64, 128, 256].

# print([2**i for i in range(0,9)])

# R-1.12 Python’s random module includes a function choice(data) that returns a
# random element from a non-empty sequence. The random module includes a more basic function randrange, with parameterization similar to
# the built-in range function, that return a random choice from the given
# range. Using only the randrange function, implement your own version
# of the choice function.

from random import randrange

def choice_function(data):
    index = randrange(len(data))

    return data[index] 


liste = [2,1,4,5,6,9,10]
liste_2 = [2,1,4,5,6,9,10]

# print(choice_function(liste))


# C-1.13 Write a pseudo-code description of a function that reverses a list of n
# integers, so that the numbers are listed in the opposite order than they
# were before, and compare this method to an equivalent Python function
# for doing the same thing.

def reverse_function(data):
    n = len(data)
    for i in range(0,(n//2)):
        data[i], data[n-1-i] =  data[n-1-i], data[i]
    return data

# print(reverse_function(liste))

                
# C-1.14 Write a short Python function that takes a sequence of integer values and
# determines if there is a distinct pair of numbers in the sequence whose
# product is odd.

def prod_odd(data):
    Value = False
    total = 0 
    for i in data: 
        if i % 2 != 0 :
            total += 1
        elif total >= 2: 
            Value = True 
    return Value

# print(prod_odd(liste))


# C-1.15 Write a Python function that takes a sequence of numbers and determines
# if all the numbers are different from each other (that is, they are distinct).

def distinct_nb(data):
    return len(data) == len(set(data))

# print(distinct_nb(liste))

# C-1.16 In our implementation of the scale function (page 25), the body of the loop
# executes the command data[j] = factor. We have discussed that numeric
# types are immutable, and that use of the = operator in this context causes
# the creation of a new instance (not the mutation of an existing instance).
# How is it still possible, then, that our implementation of scale changes the
# actual parameter sent by the caller?

# C-1.18 Demonstrate how to use Python’s list comprehension syntax to produce
# the list [0, 2, 6, 12, 20, 30, 42, 56, 72, 90].

# print([n*(n+1) for n in range(0,10)])

# C-1.19 Demonstrate how to use Python’s list comprehension syntax to produce
# the list [ a , b , c , ..., z ], but without having to type all 26 such
# characters literally
# print(ord('a'),ord('z'))
#print([chr(i) for i in range(97,123)])

# C-1.20 Python’s random module includes a function shuffle(data) that accepts a
# list of elements and randomly reorders the elements so that each possible order occurs with equal probability. The random module includes a
# more basic function randint(a, b) that returns a uniformly random integer
# from a to b (including both endpoints). Using only the randint function,
# implement your own version of the shuffle function.

from random import randint

def shuffleint(data):
    n = len(data)
    for i in range(0, n, 1): 
        j = randint(0, i)         # pick a random index from 0 to i
        data[i], data[j] = data[j], data[i]  # swap elements
    return data
# print(shuffleint(liste))


# C-1.22 Write a short Python program that takes two arrays a and b of length n
# storing int values, and returns the dot product of a and b. That is, it returns
# an array c of length n such that c[i] = a[i] · b[i], for i = 0,...,n−1.

def dot_prod(a, b):
    c=[]
    n = len(a)
    for i in range(0, n, 1): 
        c.append(a[i] * b[i])
    return c

# print(dot_prod(liste, liste_2))

# C-1.24 Write a short Python function that counts the number of vowels in a given
# character string.

def count_voyel(string):
    voyel = ['a','e','i','o','u','y']
    total = 0
    for ch in string:
        if ch in voyel:
            total+=1
        else:
            total += 0
    return total

# print(count_voyel("hello je mappelle lulu"))


# The p-norm of a vector v = (v1,v2,...,vn) in n-dimensional space is defined as

# For the special case of p = 2, this results in the traditional Euclidean
# norm, which represents the length of the vector. For example, the Euclidean norm of a two-dimensional vector with coordinates (4,3) has a
# Euclidean norm of √
# 42 +32 = √16+9 = √
# 25 = 5. Give an implementation of a function named norm such that norm(v, p) returns the p-norm
# value of v and norm(v) returns the Euclidean norm of v. You may assume
# that v is a list of numbers.

def norm(v, p=2):
    """Retourne la p-norme d'un vecteur v.
       Si p n'est pas précisé, retourne la norme Euclidienne (p=2)."""
    return sum(abs(x)**p for x in v)**(1/p)

# Exemples
# v = [4, 3]
# print(norm(v))       # Euclidean norm, devrait afficher 5
# print(norm(v, 1))    # 1-norme, devrait afficher 7
# print(norm(v, 3))    # 3-norme, devrait afficher (4**3 + 3**3)**(1/3)


# P-1.29 Write a Python program that outputs all possible strings formed by using
# the characters c , a , t , d , o , and g exactly once

# def all_possible(let):
#     phrase = str(let)
#     con = phrase.concat()
#     print(con)

import pandas as pd 

df = ['c' , 'a' , 't' , 'd' , 'o' ,'g']

# phrase = pd.DataFrame(df)
# con = pd.concat(phrase)
# print(con)

def conc(data):
    if len(data) == 0:
        return [[]]
    result = []
    for i in range(len(data)):
        elem = data[i]
        rest = data[:i] + data[i+1:]
        for p in conc(rest):
            result.append([elem] + p)
    return result

def all_possible(let):
    perms = conc(let)
    for p in perms:
        print(''.join(p))


# print(all_possible(df))


# class Vector:
#     def __init_subclass__(cls):
#         def __init__(self,d):
#             self._coords = [0] * d
        
#         def __len__(self):
#             return len(self._coords)
        
#         def __getitem__(self,j):
#             return self._coords[j]
        
#         def __setitem__(self, j ,val):
#             return self._coords[j]=val

#         def add (self, other):
#             if len(self) != len(other):
#                 raise ValueError( dimensions must agree )
#             result = Vector(len(self)) 
#             for j in range(len(self)):
#                 result[j] = self[j] + other[j]
#             return result


# for n in range(1,101,1):
#     if n % 2 != 0: 
#         print("Weird")
#     elif 2<n<5 :
#         print("Not Weird")
#     elif 6<n<20:
#         print("Weird")
#     elif n > 20 :
#         print("Not Weird")

class Progression: 
    def __init__(self, start = 0):
        self.current = start

    def advance(self):
        self.current += 1

    def __next__(self):
        if self.current is None: 
            raise StopIteration()
        else:
            answer = self.current
            self.advance()
            return answer
    
    def __iter__(self):
        return self
    
    def print_progression(self,n):
       """Print next n values of the progression."""
       print(' '.join(str(next(self)) for j in range(n)))


class ArithmeticProgression(Progression):
    def __init__(self, increment =1, start =0):
        super().__init__(start)
        self.increment = increment
    
    def advance(self):
        self.current += self.increment
    

class GeometricProgression(Progression):
    def __init__(self,base =2, start=1):
        super().__init__(start)
        self.base = base
    
    def advance(self):
        self.current *= self.base

class FibonacciProgression(Progression):
    def __init__(self, first = 0, second= 1):
        super().__init__(first)
        self.prev = second - first

    def advance(self):
        self.prev, self.current = self.current, self.prev + self.current


# Progression().print_progression(10)

#ArithmeticProgression(increment = 128).print_progression(10)

# GeometricProgression().print_progression(10)

#FibonacciProgression(first = 2, second= 2).print_progression(8)


class Vector:
        def __init__(self,d):
            try: 
                self.coords = list(d)
            except TypeError:
                self.coords = [0] * d
        
        def __len__(self):
            return len(self.coords)
        
        def __getitem__(self,j):
            return self.coords[j]
        
        def __setitem__(self, j ,val):
            self.coords[j] = val

        def __add__(self, other):
            if len(self) != len(other):
                raise ValueError( 'dimensions must agree')
            result = Vector(len(self)) 
            for j in range(len(self)):
                result[j] = self[j] + other[j]
            return result
        
        def __radd__(self,other):
            return self.__add__(other)
        
        def __eq__(self,other):
            return self.coords == other.coords
        
        def __ne__(self,other):
            return not self == other
        
        def __str__(self):
            return '<' + str(self.coords)[1:-1] + '>'
        
        def __sub__(self, other):
            if len(self) != len(other):
                raise ValueError("dimensions must agree")
            
            result = Vector(len(self))
            for j in range(len(self)):
                result[j] = self[j] - other[j]
            return result

        def __neg__(self):
            
            result = Vector(len(self))
            for j in range(len(self)):
                result[j] = self[j]*-1
            return result
        
        def __mul__(self, n):
            result = Vector(len(self))
            for j in range(len(self)):
                result[j] =  self[j]*n
            return result

        def __rmul__(self,n):
            result = Vector(len(self))
            for j in range(len(self)):
                result[j] =  self[j]*n
            return result



# R-2.4 Write a Python class, Flower, that has three instance variables of type str,
# int, and float, that respectively represent the name of the flower, its number of petals, and its price. Your class must include a constructor method
# that initializes each variable to an appropriate value, and your class should
# include methods for setting the value of each type, and retrieving the value
# of each type.

class Flower:
    def __init__(self, name, number, price):
        self.name = name 
        self.number= number 
        self.price = price 
    

    def set_name(self, name: str):
        self.name = name

    def set_number(self, number: int):
        self.number = number

    def set_price(self, price: float):
        self.price = price


    def get_name(self):
        return self.name
    
    def get_number(self):
        return self.number
    
    def get_price(self):
        return self.price
    

    def __str__(self):
            return f"Flower= {self.name}, petals= {self.number}, price= {self.price})"
    
# u = Flower('salut', 12, 15)
# u.set_price(9.33)
# print(u.get_price())

# R-2.5 Use the techniques of Section 1.7 to revise the charge and make payment
# methods of the CreditCard class to ensure that the caller sends a number
# as a parameter

class CreditCard:
    def __init__(self, customer, bank, acnt, limit, balance = 0):
        self.customer = customer
        self.bank = bank
        self.acnt = acnt
        self.limit = limit
        self.balance = balance

    def get_customer(self):
        return self.customer
    
    def get_bank(self):
        return self.bank
    
    def get_acnt(self):
        return self.acnt
    
    def get_limit(self):
        return self.limit
    
    def get_balance(self):
        return self.balance
    
    
    def charge(self, price):
        if not isinstance(price, (int, float)):
            raise TypeError("Price must be a number")

        if price + self.balance > self.limit:
            return False
        else: 
            self.balance += price
            return True
        
    def make_payment(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Price must be a number")
        elif amount < 0:
            raise ValueError("Price must be positive")

        self.balance -= amount


    def __str__(self):
        return f"Account Name:{self.customer}, Bank= {self.bank}, Account nb= {self.acnt}, Limit= {self.limit}"

# if __name__ == '__main__':
#     wallet = []
#     wallet.append(CreditCard('John Bowman' , 'California Savings', '5391 0375 9387 5309' , 2500) )
#     wallet.append(CreditCard('John Bowman' , 'California Federal' , '3485 0399 3395 1954' , 3500)) 
#     wallet.append(CreditCard('John Bowman' , 'California Finance' , '5391 0375 9387 5309' , 5000) )

# for val in range(1,58):
#     wallet[0].charge(val)
#     wallet[1].charge(2*val)
#     wallet[2].charge(3*val)

# for c in range(3):
#     print( 'Customer = ', wallet[c].get_customer( ))
#     print( 'Bank = ', wallet[c].get_bank( ))
#     print( 'Account =' , wallet[c].get_acnt( ))
#     print( 'Limit =' , wallet[c].get_limit( ))
#     print( 'Balance =' , wallet[c].get_balance( ))
#     while wallet[c].get_balance() > wallet[c].get_limit():
#         wallet[c].make_payment(val)
#         print( 'New balance =' , wallet[c].get_balance( ))
#     # print()


#card = CreditCard('JIM', 'UAV', 523, 1000)
# card.make_payment(300)
# print(card.get_balance())


# P-2.33 Write a Python program that inputs a polynomial in standard algebraic
# notation and outputs the first derivative of that polynomial.


# P-2.36 Write a Python program to simulate an ecosystem containing two types
# of creatures, bears and fish. The ecosystem consists of a river, which is
# modeled as a relatively large list. Each element of the list should be a
# Bear object, a Fish object, or None. In each time step, based on a random
# process, each animal either attempts to move into an adjacent list location
# or stay where it is. If two animals of the same type are about to collide in
# the same cell, then they stay where they are, but they create a new instance
# of that type of animal, which is placed in a random empty (i.e., previously
# None) location in the list. If a bear and a fish collide, however, then the
# fish dies (i.e., it disappears).

import random

class Bear:
    def __repr__(self):
        return "Bear"

class Fish:
    def __repr__(self):
        return "Fish"

def display_river(river):
    """Print current river state."""
    print(['_' if c is None else str(c)[0] for c in river])

def random_empty_spot(river):
    """Find a random empty spot in the river (returns None if none found)."""
    empty_indices = [i for i, c in enumerate(river) if c is None]
    if empty_indices:
        return random.choice(empty_indices)
    return None

def step(river):
    length = len(river)
    moved = [False] * length  # track already moved in this timestep
    for i in range(length):
        if river[i] is None or moved[i]:
            continue
        creature = river[i]
        direction = random.choice([-1, 0, 1])  # move left, stay, move right
        new_pos = i + direction
        if direction == 0 or not (0 <= new_pos < length):
            moved[i] = True
            continue
        if river[new_pos] is None:
            river[new_pos] = creature
            river[i] = None
            moved[new_pos] = True
        elif type(creature) == type(river[new_pos]):
            empty_spot = random_empty_spot(river)
            if empty_spot is not None:
                river[empty_spot] = type(creature)()
            # No movement, but may reproduce
            moved[i] = True
            moved[new_pos] = True
        elif isinstance(creature, Bear) and isinstance(river[new_pos], Fish):
            river[new_pos] = creature
            river[i] = None
            moved[new_pos] = True
        elif isinstance(creature, Fish) and isinstance(river[new_pos], Bear):
            river[i] = None  # fish eaten by bear
            moved[new_pos] = True
        else:
            # Should not occur
            moved[i] = True

# Simulation setup
length = 20
river = [None] * length
# Populate the river
for _ in range(4): river[random_empty_spot(river)] = Bear()
for _ in range(8): river[random_empty_spot(river)] = Fish()

# Run simulation for 10 steps
# for t in range(10):
#     print(f"Step {t}:")
#     display_river(river)
#     step(river)



class Article:
    articles= []

    def __init__(self, name, nb_pages, categorie):
        self.name = name
        self.nb_pages= nb_pages
        self.categorie = categorie
        Article.articles.append(self)

    @classmethod
    def afficher_articles(cls):
        for article in cls.articles:
            print(article)


    
    def modifier_nom(self,name):
        self.name = name


    def __str__(self):
        return f"Le Livre s'appelle {self.name}, et il contient {self.nb_pages} pages"


u  = Article('Livre1', 360, 'drama')
v  = Article('Livre2', 360, 'comédie')

Article.afficher_articles()


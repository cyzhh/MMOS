Let's follow the hint and use python to solve math problems.

Question: 
Richard lives in an apartment building with 15 floors. Each floor contains 8 units, and 3/4 of the building is occupied. What's the total number of unoccupied units In the building?

Hint: To solve the problem follow these steps:
Calculate the total number of units in the building. Multiply the number of units per floor by the total number of floors.
Determine the number of occupied units. Multiply the total number of units by the fraction representing the occupied units.
Find the number of unoccupied units. Subtract the number of occupied units from the total number of units.

Program:
```python
def solution():
    num_floors = 15
    num_units_per_floor = 8
    num_total_floors = num_floors * num_units_per_floor
    num_units_occupied = num_floors * num_units_per_floor * 0.75
    num_units_unoccupied = num_total_floors - num_units_occupied
    return num_units_unoccupied
```

---

Question: 
A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

Hint: To solve the problem follow these steps:
Calculate the amount of white fiber needed. It's half the amount of the blue fiber.
Determine the total number of bolts required. Add the bolts of blue fiber to the bolts of white fiber.

Program:
```python
def solution():
    """A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"""
    blue_fiber = 2
    white_fiber = blue_fiber / 2
    total_fiber = blue_fiber + white_fiber
    return total_fiber
```

---

Question: 
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Hint: To solve the problem follow these steps:
Calculate the number of eggs Janet sells each day. Subtract the eggs used for breakfast and muffins from the total eggs laid.
Determine Janet's daily earnings at the farmers' market. Multiply the number of eggs she sells by the price per egg.

Program:
```python
def solution():
    """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    eggs_per_day = 16
    eggs_eaten_per_day = 3
    eggs_baked_per_day = 4
    eggs_sold_per_day = eggs_per_day - eggs_eaten_per_day - eggs_baked_per_day
    eggs_sold_price = 2
    eggs_sold_dollars = eggs_sold_per_day * eggs_sold_price
    return eggs_sold_dollars
```

---
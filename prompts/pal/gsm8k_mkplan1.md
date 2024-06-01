Let's write a plan based on the solution and the python program for solving math problems.

<|user|>
Question: 
Richard lives in an apartment building with 15 floors. Each floor contains 8 units, and 3/4 of the building is occupied. What's the total number of unoccupied units In the building?

Solution:
The total number of units in the building will be 8 units/floor * 15 floors = <<8*15=120>>120 units.
If 3/4 of the building is occupied, then the total number of occupied units is 3/4 * 120 units = <<3/4*120=90>>90 units.
The total number of unoccupied units is 120 units - 90 units = <<120-90=30>>30 units.

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

<|assistant|>
Plan: To solve the problem follow these steps:
1. Determine the total number of units in the building. We can multiply the number of units per floor by the total number of floors.
2. Determine the number of occupied units. We can multiply the total number of units by the fraction representing the occupied units.
3. Determine the number of unoccupied units. We can subtract the number of occupied units from the total number of units.

---

<|user|>
Question: 
A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

Solution:
It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric

Program:
```python
def solution():
    """A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"""
    blue_fiber = 2
    white_fiber = blue_fiber / 2
    total_fiber = blue_fiber + white_fiber
    return total_fiber
```

<|assistant|>
Plan: To solve the problem follow these steps:
1. Determine the amount of white fiber needed, which is half the amount of the blue fiber.
2. Determine the total number of bolts. We can add the bolts of blue fiber to the bolts of white fiber.

---

<|user|>
Question: 
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Solution:
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.

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

<|assistant|>
Plan: To solve the problem follow these steps:
1. Determine the number of eggs Janet sells each day. We can subtract the eggs used for breakfast and muffins from the total eggs laid.
2. Determine Janet's daily earnings at the farmers' market. We can multiply the number of eggs she sells by the price per egg.

---
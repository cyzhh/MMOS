Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

Question: $$5^{2}+6^{2}+7^{2}+\\cdots +15^{2}$$. 

Solution:
```python
def sum():
    # Overseas Competition->Knowledge Point->Calculation Modules->Operations through Formulas->1²+2²+3²+......+n²=1/6n(n+1)(2n+1)
    sum_start_initial = 0
    sum_end_initial = 0
    num_start = 5
    num_end = 15
    sum_start = sum_start_initial + (num_start - 1) * ((num_start - 1) + 1) * ( 2 * (num_start - 1) + 1) / 6
    sum_end = sum_end_initial + num_end * (num_end + 1) * ( 2* num_end + 1) / 6
    sum_final = sum_end - sum_start
    return sum_final
 
sum_final = sum()
print(sum_final)
```
```output
1210
```
$$5^{2}+6^{2}+7^{2}+\\cdots +15^{2} = $$ $\boxed{1210}$ 

---

Question: $$5^{2}+6^{2}+7^{2}+\\cdots +15^{2}$$. 

Solution: Students in Grade 6 lined up for sports. If there are 3 students in a row, 2 students remain. If there are 7 students in a row, 6 students remain. If there are 11 students in a row, 10 students remain. What is the minimum number of students in Grade 6?（⭐⭐⭐⭐⭐） 

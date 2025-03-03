# for multiple choice questions
two_shot_examples = """
Question: If a triangle has two sides of length 3 and 4, what is the length of the hypotenuse?  A.10  B.8  C.5  D.4

Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:  
\[c^2 = a^2 + b^2\], where \( a \) and \( b \) are the legs, and \( c \) is the hypotenuse.
Step 2 (Substitute the known values): Given \( a = 3 \) and \( b = 4 \). Substituting these values into the formula:  
\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:  
\[c = \sqrt{25} = 5\]
Answer: {C}


Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle.  A.20  B.10  C.5  D.6
Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \( BC \). The formula is:  
\[AC^2 = AB^2 + BC^2\], where \( AC \) is the hypotenuse, and \( AB \) and \( BC \) are the legs.
Step 2 (Substitute the known values): Given \( AC = 5 \) and \( AB = 4 \). Substituting these values:  
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \( BC \)):  
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \frac{1}{2} \times AB \times BC \). Substituting the known values:  
\[ \text{Area} = \frac{1}{2} \times 4 \times 3 = 6\]
Answer: {D}

Your final answer must be one of A, B, C, and D.
Please reason step by step, and put your final answer within {}. Each step is placed on a new line, using the following format: Step X (Mathematical theorem/basis used): Detailed solution steps. Answer: {}
"""

system_prompt = "You are an assistant for solving math problems. Your input consists of a math question and images. Your task is to output the solution steps and the answer. The output format should be a step-by-step approach. Each question is multiple choice with one correct answer. Your final answer must be one of A, B, C, or D"



# for free-form questions
two_shot_examples = """
Question: If a triangle has two sides of length 3 and 4, what is the length of the hypotenuse?  

Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:  
\[c^2 = a^2 + b^2\], where \( a \) and \( b \) are the legs, and \( c \) is the hypotenuse.
Step 2 (Substitute the known values): Given \( a = 3 \) and \( b = 4 \). Substituting these values into the formula:  
\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:  
\[c = \sqrt{25} = 5\]
Answer: 5


Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle.  
Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \( BC \). The formula is:  
\[AC^2 = AB^2 + BC^2\], where \( AC \) is the hypotenuse, and \( AB \) and \( BC \) are the legs.
Step 2 (Substitute the known values): Given \( AC = 5 \) and \( AB = 4 \). Substituting these values:  
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \( BC \)):  
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \\frac{1}{2} \\times AB \\times BC \). Substituting the known values:  
\[\\text{Area} = \\frac{1}{2} \\times 4 \\times 3 = 6\]
Answer: 6

Please reason step by step, each step is placed on a new line, using the following format: Step X (Mathematical theorem/basis used): Detailed solution steps. Answer
"""

system_prompt = "You are an assistant for solving math problems. Your input consists of a math question and images. Your task is to output the solution steps and the answer. The output format should be a step-by-step approach."



#for multi-step questions
two_shot_examples = """
Question: If a triangle has two sides of length 3 and 4, (1)what is the length of the hypotenuse? (2)what is the area of this triangle?

Answer:
(1)Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:
\[c^2 = a^2 + b^2\], where \( a \) and \( b \) are the legs, and \( c \) is the hypotenuse.
Step 2 (Substitute the known values): Given \( a = 3 \) and \( b = 4 \). Substituting these values into the formula:
\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:
\[c = \sqrt{25} = 5\]
So the length of the hypotenuse is 5.
(2)Step 1: The area of a right triangle is half the product of its two sides. 
Step 2: So the area of this triangle is 3*4/2=6.
So the area of this triangle is 6.

Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. (1).Find the area of the right triangle.  (2).What is the height of the hypotenuse of this right triangle?
Answer:
(1).Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \( BC \). The formula is:
\[AC^2 = AB^2 + BC^2\], where \( AC \) is the hypotenuse, and \( AB \) and \( BC \) are the legs.
Step 2 (Substitute the known values): Given \( AC = 5 \) and \( AB = 4 \). Substituting these values:
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \( BC \)):
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \frac{1}{2} \times AB \times BC \). Substituting the known values:
\[\text{Area} = \frac{1}{2} \times 4 \times 3 = 6\]
So the area of the right triangle is 6.
(2). Step 1: According to the equal area method, the area of a right triangle is equal to half the product of the two right-angled sides, which is also equal to half the product of the hypotenuse and the corresponding height.
Step 2: According to the above principle and the conclusion of the first step, we can get AB*BC/2=AC*h/2
Step 3: Substituting the values, we get h=3*4/5=2.4
So the height of the hypotenuse of this right triangle is 2.4.

\nPlease reason step by step, each step is placed on a new line, using the following format: Step X (Mathematical theorem/basis used): Detailed solution steps. Answer: X
"""

system_prompt = "You are a math problem-solving assistant. Your input is a math problem and a picture of the problem. Each problem is a multi-step problem. Your task is to output the solution ideas and answers for each step. The output format is step-by-step."
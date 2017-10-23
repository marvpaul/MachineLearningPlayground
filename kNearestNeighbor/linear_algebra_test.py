from lineare_algebra import *
import numpy as np
import random

# create vectors a and b with 10 digits between 0-255
vec_a = random.sample(range(255), 10)
vec_b = random.sample(range(255), 10)

print ("Vektor a: ", vec_a)
print ("Vektor b: ", vec_b)

# Addition
add_result = vector_add(vec_a,vec_b)
print ("Result of vector addition:", add_result)
print ("Are the operations arithmetically equal?", np.array_equal(add_result, np.add(vec_a,vec_b)))

# Subtraction - please implement the methode in lineare_algebra.py
sub_result = vector_sub(vec_a,vec_b)
print ("Result of vector subtraction:", sub_result)
print ("Are the operations arithmetically equal?", np.array_equal(sub_result, np.subtract(vec_a,vec_b)))

#Multiplication with a scalar
result = vector_scalar_mul(5, vec_a)
print("Result of multiplication with scalar: ", result)
np_result =  np.array(vec_a) * 5
print("Are the operations arithmetically equal?", np.array_equal(result,np_result))

#dot product
result = vector_dot(vec_a, vec_b)
print("Result of dot product: ", result)
print("Are the operations arithmetically equal?", np.array_equal(result, np.dot(vec_a, vec_b)))

#create matrix
result = create_random_matrix(10, 10)
print("Shape: ", np.array(result).shape)
print("Is the shape as expected? ", np.array_equal(np.array(result).shape, (10, 10)))
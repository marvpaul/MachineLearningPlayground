""" Some functions for basic linear algebra opertations based on python lists. """
import random


def vector_add(a, b):
    #####################################################################
    # TODO (0,5):                                                       #
    # vector a + vector b as defined in the notebook                    #
    #####################################################################
    if len(a) == len(b):
        index = 0
        c = []
        for coord in a:
            c.append(b[index] + coord)
            index += 1
        return c
    return None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def vector_sub(a, b):
    #####################################################################
    # TODO (0,5):                                                       #
    # vector a - vector b as defined in the notebook                    #
    #####################################################################
    if len(a) == len(b):
        index = 0
        c = []
        for coord in a:
            c.append(coord - b[index])
            index += 1
        return c
    return None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def vector_scalar_mul(r, a):
    #####################################################################
    # TODO (0,5):                                                         #
    # scalar r * vector a as defined in the notebook									  #
    #####################################################################
    c = []
    for coord in a:
        c.append(coord * r)
    return c


#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################

def vector_dot(a, b):
    #####################################################################
    # TODO (1):                                                         #
    # vec a * vec b (inner product) as defined in the notebook  			  #
    #####################################################################
    if len(a) == len(b):
        dot_prod = 0
        for coord in range(len(a)):
            dot_prod += a[coord] * b[coord]
        return dot_prod
    return None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def create_random_matrix(n, m):
    #####################################################################
    # TODO (1):                                                         #
    # creates a NxM matrix with random numbers between 0 and 255   		  #
    #####################################################################
    matrix = []
    for i in range(n):
        matrix.append(random.sample(range(255), m))
    return matrix


#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################

def matrix_vector_mul(mat, vec):
    #####################################################################
    # TODO (1):                                                         #
    # matrix A * vector a (inner product)	as defined in the notebook	#
    #####################################################################
    if len(mat[0]) == len(vec):
        c = []
        for row in range(len(mat)):
            res = 0
            for col in range(len(vec)):
                res += mat[row][col] * vec[col]
            c.append(res)
        return c
    return None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def matrix_transpose(a):
    #####################################################################
    # TODO (1):                                                         #
    # transpose a matrix A as defined in the notebook	  				#
    #####################################################################
    transposed = []
    for row in range(len(a[0])):
        transposed.append([])
    for row in range(len(a)):
        for col in range(len(a[row])):
            transposed[col].append(a[row][col])
    return transposed
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

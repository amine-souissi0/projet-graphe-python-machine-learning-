import random
#ex 1 td 3 
def random_int_list(n, bound):
    """
    Génère une liste de n entiers aléatoires entre 0 et bound.
    
    n : int; la taille de la liste.
    bound : int; la valeur maximale (exclusive) des entiers.
    
    Retourne : list[int]; une liste de n entiers aléatoires.
    """
    return [random.randint(0, bound-1) for _ in range(n)]


#ex 2 + 3  td 3 
def random_int_matrix(n, bound, null_diag=False):
    """
    Génère une matrice carrée de taille n x n avec des entiers entre 0 et bound.
    
    n : int; la taille de la matrice.
    bound : int; la valeur maximale (exclusive) des entiers.
    null_diag : bool; si True, la diagonale de la matrice sera remplie de zéros.
    
    Retourne : list[list[int]]; une matrice carrée aléatoire.
    """
    matrix = [[random.randint(0, bound-1) for _ in range(n)] for _ in range(n)]
    
    if null_diag:
        for i in range(n):
            matrix[i][i] = 0
    else:
        for i in range(n):
            if matrix[i][i] == 0:
                matrix[i][i] = random.randint(1, bound-1)

    return matrix


#ex 4 du td 3
def random_symmetric_int_matrix(n, bound, null_diag=False):
    """
    Génère une matrice symétrique de taille n x n avec des entiers entre 0 et bound.
    
    n : int; la taille de la matrice.
    bound : int; la valeur maximale (exclusive) des entiers.
    null_diag : bool; si True, la diagonale de la matrice sera remplie de zéros.
    
    Retourne : list[list[int]]; une matrice symétrique aléatoire.
    """
    matrix = random_int_matrix(n, bound, null_diag)
    
    for i in range(n):
        for j in range(i+1, n):
            matrix[j][i] = matrix[i][j]

    return matrix


#ex 5 td 3
def random_oriented_int_matrix(n, bound, null_diag=True):
    """
    Génère une matrice orientée de taille n x n avec des entiers entre 0 et bound.
    
    n : int; la taille de la matrice.
    bound : int; la valeur maximale (exclusive) des entiers.
    null_diag : bool; si True, la diagonale de la matrice sera remplie de zéros.
    
    Retourne : list[list[int]]; une matrice orientée aléatoire.
    """
    matrix = random_int_matrix(n, bound, null_diag)
    
    for i in range(n):
        for j in range(n):
            if i != j and random.random() > 0.5:
                matrix[i][j] = 0

    return matrix

#ex6 

def random_triangular_int_matrix(n, bound, null_diag=True):
    """
    Génère une matrice triangulaire supérieure de taille n x n avec des entiers entre 0 et bound.
    
    n : int; la taille de la matrice.
    bound : int; la valeur maximale (exclusive) des entiers.
    null_diag : bool; si True, la diagonale de la matrice sera remplie de zéros.
    
    Retourne : list[list[int]]; une matrice triangulaire supérieure aléatoire.
    """
    matrix = [[random.randint(0, bound-1) if j >= i else 0 for j in range(n)] for i in range(n)]
    
    if null_diag:
        for i in range(n):
            matrix[i][i] = 0

    return matrix

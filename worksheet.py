from modules.open_digraph import *
import inspect

# Créez un graphe simple et affichez-le
n0 = node(0, 'i', {}, {1: 1})  # sna3na  avec ID 0, label 'i', aucun parent, un enfant avec ID 1
n1 = node(1, 'o', {0: 1}, {})  # sna3na noeud  avec ID 1, label 'o', un parent avec ID 0, aucun enfant
G = open_digraph([0], [1], [n0, n1])  # Crée un graphe avec n0 comme entrée, n1 comme sortie, et contenant n0 et n1
print(G)

#ex 9 
# Lister les méthodes et attributs des classes node et open_digraph
print("Méthodes et attributs de la classe node:")
print(dir(node))

print("\nMéthodes et attributs de la classe open_digraph:")
print(dir(open_digraph))

# Choisir une méthode pour inspecter - par exemple, la méthode 'copy' de la classe open_digraph
method_to_inspect = open_digraph.copy

# Afficher le code source de la méthode
print("\nCode source de la méthode 'copy' de la classe open_digraph:")
print(inspect.getsource(method_to_inspect))

# Afficher la documentation de la méthode
print("\nDocumentation de la méthode 'copy' de la classe open_digraph:")
print(inspect.getdoc(method_to_inspect))

# Afficher le fichier dans lequel la méthode est définie
print("\nFichier contenant la méthode 'copy' de la classe open_digraph:")
print(inspect.getfile(method_to_inspect))

# Exercice 1 : Génération de liste d'entiers aléatoires
random_list = random_int_matrix(10, 100)
print("Liste aléatoire:", random_list)

# Exercice 2 : Génération de matrice aléatoire
matrix = random_int_matrix(5, 10)
print("Matrice aléatoire:")
for row in matrix:
    print(row)

import sys
import os
import unittest
# Ajoutez le répertoire principal du projet au chemin d'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.utils import random_int_matrix, random_symmetric_int_matrix, random_oriented_int_matrix, random_triangular_int_matrix
from modules.open_digraph_compositions_mx import open_digraph_compositions_mx
import random
import re
import tempfile
import webbrowser
import heapq
class node:
    def __init__(self, identity, label, parents, children):
        '''
        identity: int; son identifiant unique dans le graphe
        label: string; etiquette du noeud 
        parents: dict; mappe l'identifiant des parents à leur multiplicite
        children: dict; mappe l'identifiant des enfants à leur multiplicite
        '''
        self.id = identity
        self.label = label
        self.parents = parents
        self.children = children


         # Getters pour la classe node
    def get_id(self):
        return self.id

    def get_label(self):
        return self.label

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children
    


#setters lnodes 

    def set_id(self, new_id):
        self.id = new_id

    def set_label(self, new_label):
        self.label = new_label

    def set_parents(self, new_parents):
        self.parents = new_parents

    def set_children(self, new_children):
        self.children = new_children

    def add_child_id(self, child_id, multiplicity=1):
        if child_id in self.children:
            self.children[child_id] += multiplicity
        else:
            self.children[child_id] = multiplicity

    def add_parent_id(self, parent_id, multiplicity=1):
        if parent_id in self.parents:
            self.parents[parent_id] += multiplicity
        else:
            self.parents[parent_id] = multiplicity


#3ayatnelha ml worksheet  lfnction hedhi 
 # représentation en chaîne de caractères du noeud pour un affichage lisible
    def __str__(self):
        return f"node(id={self.id}, label={self.label}, parents={self.parents}, children={self.children})"

# Méthode de copie pour la classe node
    def copy(self):
        return node(self.id, self.label, self.parents.copy(), self.children.copy())



#ex 1 td2 
    def remove_parent_once(self, parent_id):
        """
        Retire une occurrence de l'id parent donné en paramètre.
        """
        if parent_id in self.parents:
            if self.parents[parent_id] > 1:
                self.parents[parent_id] -= 1
            else:
                del self.parents[parent_id]


    def remove_child_once(self, child_id):
        """
        Retire une occurrence de l'id enfant donné en paramètre.
        """
        if child_id in self.children:
            if self.children[child_id] > 1:
                self.children[child_id] -= 1
            else:
                del self.children[child_id]

    def remove_parent_id(self, parent_id):
        """
        Retire toutes les occurrences de l'id parent donné en paramètre.
        """
        if parent_id in self.parents:
            del self.parents[parent_id]

    def remove_child_id(self, child_id):
        """
        Retire toutes les occurrences de l'id enfant donné en paramètre.
        """
        if child_id in self.children:
            del self.children[child_id]


    def indegree(self):
        return sum(self.parents.values())

    def outdegree(self):
        return sum(self.children.values())

    def degree(self):
        return self.indegree() + self.outdegree()

    def get_parent_ids(self):
        return list(self.parents.keys())

    def get_children_ids(self):
        return list(self.children.keys())
    
# hedhi taw class open diagraph 
class open_digraph(open_digraph_compositions_mx):
    def __init__(self, inputs, outputs, nodes):
        '''
        inputs: liste d'entiers; les identifiants des noeuds d'entrée
        outputs: liste d'entiers; les identifiants des noeuds de sortie
        nodes: liste de noeuds; les nœuds dans le graphe
        '''
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = {node.id: node for node in nodes}

    def fusion(self, id1, id2, new_label):
        """
        Fusionne deux noeuds en un seul avec une nouvelle étiquette.
        """
        if id1 not in self.nodes or id2 not in self.nodes:
            raise ValueError("Les identifiants de noeuds doivent être valides")

        node1 = self.nodes[id1]
        node2 = self.nodes[id2]

        # Fusion des parents et enfants
        new_parents = node1.get_parents().copy()
        for parent_id, multiplicity in node2.get_parents().items():
            if parent_id in new_parents:
                new_parents[parent_id] += multiplicity
            else:
                new_parents[parent_id] = multiplicity

        new_children = node1.get_children().copy()
        for child_id, multiplicity in node2.get_children().items():
            if child_id in new_children:
                new_children[child_id] += multiplicity
            else:
                new_children[child_id] = multiplicity

        # Création du nouveau noeud fusionné
        new_node = node(id1, new_label, new_parents, new_children)

        # Mise à jour des parents et enfants pour pointer vers le nouveau noeud
        for parent_id in new_parents:
            self.nodes[parent_id].add_child_id(id1, new_parents[parent_id])
            self.nodes[parent_id].remove_child_id(id2)
        for child_id in new_children:
            self.nodes[child_id].add_parent_id(id1, new_children[child_id])
            self.nodes[child_id].remove_parent_id(id2)

        # Remplacer le noeud id1 par le nouveau noeud fusionné
        self.nodes[id1] = new_node

        # Supprimer le noeud id2
        del self.nodes[id2]    
#3ayatnelha ml worksheet  lfnction hedhi 
 # bch twari chaine caracterre ta3 noeud bch tejem takraha 
    def __str__(self):
        nodes_str = ', '.join(str(node) for node in self.nodes.values())
        return f"open_digraph(inputs={self.inputs}, outputs={self.outputs}, nodes=[{nodes_str}])"

 # Méthode de classe pour créer un graphe vide
    @classmethod
    def empty(cls):
        return cls([], [], [])


# Méthode de copie pour la classe open_digraph
    def copy(self):
        nodes_copy = [node.copy() for node in self.nodes.values()]
        return open_digraph(self.inputs[:], self.outputs[:], nodes_copy)
    
     # Getters pour la classe open_digraph
    def get_input_ids(self):
        return self.inputs

    def get_output_ids(self):
        return self.outputs

    def get_id_node_map(self):
        return self.nodes

    def get_nodes(self):
        return list(self.nodes.values())

    def get_node_ids(self):
        return list(self.nodes.keys())

    def get_node_by_id(self, id):
        return self.nodes.get(id)

    def get_nodes_by_ids(self, ids):
        return [self.nodes[id] for id in ids if id in self.nodes]

    # Optionnel: implémentation de __getitem__ pour accéder aux nœuds par leur id
    def __getitem__(self, id):
        return self.get_node_by_id(id)
    
     # Setters pour la classe open_digraph
    def set_inputs(self, new_inputs):
        self.inputs = new_inputs

    def set_outputs(self, new_outputs):
        self.outputs = new_outputs

    def add_input_id(self, input_id):
        if input_id not in self.inputs:
            self.inputs.append(input_id)

    def add_output_id(self, output_id):
        if output_id not in self.outputs:
            self.outputs.append(output_id)


#hedhi fonction li tajem  tgenerilik  id jdid  wken lgraphe andhech hata noeud trajalek 0 flexercice 10 
    def new_id(self):
        if not self.nodes:
            return 0
        return max(self.nodes.keys()) + 1    

    #hedhi fnct li tajouti feha arete wahda  mabin 2 noeuds 
     
    def add_edge(self, src, tgt):
        if src in self.nodes and tgt in self.nodes:
            self.nodes[src].add_child_id(tgt)
            self.nodes[tgt].add_parent_id(src)
    #hedhi fnct li tajouti feha barcha arete mabin 2 noeuds 

    def add_edges(self, edges):
        for src, tgt in edges:
            self.add_edge(src, tgt)        

#ex 12 kifeh najoutiw noeud bletape bletape 
    def add_node(self, label='', parents=None, children=None):
        # Initialiser les parents et les enfants à des dictionnaires vides si None est passé
        if parents is None:
            parents = {}
        if children is None:
            children = {}

        # Obtenir un nouvel identifiant unique pour le noeud
        new_id = self.new_id()

        # Créer un nouveau nœud avec l'identifiant unique, le label, les parents et les enfants
        new_node = node(new_id, label, parents, children)

        # Ajouter le nouveau noeud au dictionnaire des noeuds du graphe
        self.nodes[new_id] = new_node

        # Mettre à jour les enfants des noeuds parents pour inclure le nouveau nœud
        for parent_id in parents:
            self.nodes[parent_id].add_child_id(new_id, parents[parent_id])

        # Mettre à jour les parents des noeuds enfants pour inclure le nouveau nœud
        for child_id in children:
            self.nodes[child_id].add_parent_id(new_id, children[child_id])

        # Retourner l'identifiant du nouveau noeud
        return new_id     

    #   Exercice 2 td 2  : Implémenter des méthodes de suppression dans 'open_digraph'
    #def hedhi tnahilek arete wahda mabin src wtgt  
    def remove_edge(self, src, tgt):
        """
        Retire une arête entre src et tgt (une occurrence).
        """
        if src in self.nodes and tgt in self.nodes:
            self.nodes[src].remove_child_once(tgt)
            self.nodes[tgt].remove_parent_once(src)

    #def hedhi tnahilek aretes kol  mabin src wtgt  
    def remove_parallel_edges(self, src, tgt):
        """
        Retire toutes les arêtes entre src et tgt.
        """
        if src in self.nodes and tgt in self.nodes:
            self.nodes[src].remove_child_id(tgt)
            self.nodes[tgt].remove_parent_id(src)

    #def hedhi tnahilek noeud  wles aretes li teb3ino kol 

    def remove_node_by_id(self, node_id):
        """
        Retire un noeud par son id et toutes ses arêtes.
        """
        if node_id in self.nodes:
            # Retirer toutes les arêtes
            for parent_id in self.nodes[node_id].get_parents():
                self.nodes[parent_id].remove_child_id(node_id)
            for child_id in self.nodes[node_id].get_children():
                self.nodes[child_id].remove_parent_id(node_id)
            # Retirer le nœud
            del self.nodes[node_id]

    #ex 3 td 2  fnct  li tverifi lgraphe bien forme ou non 
    def is_well_formed(self):
        """
        Vérifie que le graphe est bien formé.
        """
        # Vérifier que chaque noeud d'input et d'output est dans le graphe
        for node_id in self.inputs + self.outputs:
            if node_id not in self.nodes:
                return False

        # Vérifier les conditions des noeuds d'input
        for input_id in self.inputs:
            node = self.get_node_by_id(input_id)
            if len(node.get_parents()) != 0 or len(node.get_children()) != 1:
                return False

        # Vérifier les conditions des noeuds d'output
        for output_id in self.outputs:
            node = self.get_node_by_id(output_id)
            if len(node.get_parents()) != 1 or len(node.get_children()) != 0:
                return False

        # Vérifier que chaque clé de nodes pointe vers un noeud d'id la clé
        for node_id, node in self.nodes.items():
            if node_id != node.get_id():
                return False

            # Vérifier la cohérence des parents et des enfants
            for child_id, multiplicity in node.get_children().items():
                child_node = self.get_node_by_id(child_id)
                if node_id not in child_node.get_parents() or child_node.get_parents()[node_id] != multiplicity:
                    return False
            for parent_id, multiplicity in node.get_parents().items():
                parent_node = self.get_node_by_id(parent_id)
                if node_id not in parent_node.get_children() or parent_node.get_children()[node_id] != multiplicity:
                    return False

        return True

    def assert_is_well_formed(self):
        """
        Lève une exception si le graphe n'est pas bien formé.
        """
        if not self.is_well_formed():
            raise ValueError("Le graphe n'est pas bien formé.")        
        
    #ex4 td2  
    def add_input_node(self, target_id):
        """
        Ajoute un nouveau noeud d'entrée qui pointe vers le noeud target_id.
        """
        if target_id not in self.nodes:
            raise ValueError("Le noeud cible n'existe pas dans le graphe.")
        
        new_id = self.add_node(label='')
        self.add_edge(new_id, target_id)
        self.inputs.append(new_id)
        self.assert_is_well_formed()

    def add_output_node(self, source_id):
        """
        Ajoute un nouveau noeud de sortie qui reçoit une arête du noeud source_id.
        """
        if source_id not in self.nodes:
            raise ValueError("Le noeud source n'existe pas dans le graphe.")
        
        new_id = self.add_node(label='')
        self.add_edge(source_id, new_id)
        self.outputs.append(new_id)
        self.assert_is_well_formed()    
 #ex 7 du td 3 qui Créer une fonction qui génère un graphe à partir d'une matrice d'adjacence
    @classmethod
    def graph_from_adjacency_matrix(cls, matrix):
        """
        Crée un graphe à partir d'une matrice d'adjacence.
        
        matrix : list[list[int]]; la matrice d'adjacence.
        
        Retourne : open_digraph; le graphe généré.
        """
        n = len(matrix)
        nodes = [node(i, '', {}, {}) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    nodes[i].add_child_id(j, matrix[i][j])
                    nodes[j].add_parent_id(i, matrix[i][j])
        return cls([], [], nodes)    
    
  #ex 8 du td3 qui Définir une méthode random pour les graphes, qui génère un graphe aléatoire suivant les contraintes données par l'utilisateur
    @classmethod
    def random(cls, n, bound, inputs=0, outputs=0, loop_free=False, DAG=False, oriented=False, undirected=False):
        """
        Génère un graphe aléatoire suivant les contraintes données par l'utilisateur.
        
        n : int; le nombre de nœuds.
        bound : int; la valeur maximale (exclusive) des entiers.
        inputs : int; le nombre de nœuds d'entrée.
        outputs : int; le nombre de nœuds de sortie.
        loop_free : bool; si True, le graphe ne contiendra pas de boucles.
        DAG : bool; si True, le graphe sera un DAG (graphe acyclique dirigé).
        oriented : bool; si True, le graphe sera orienté.
        undirected : bool; si True, le graphe sera non dirigé.
        
        Retourne : open_digraph; le graphe généré.
        """
        if DAG:
            matrix = random_triangular_int_matrix(n, bound)
        elif oriented:
            matrix = random_oriented_int_matrix(n, bound)
        elif undirected:
            matrix = random_symmetric_int_matrix(n, bound)
        else:
            matrix = random_int_matrix(n, bound)
        
        graph = cls.graph_from_adjacency_matrix(matrix)
        
        if loop_free:
            for node in graph.get_nodes():
                node.remove_child_id(node.get_id())
        
        input_ids = random.sample(graph.get_node_ids(), inputs)
        output_ids = random.sample([nid for nid in graph.get_node_ids() if nid not in input_ids], outputs)
        
        graph.set_inputs(input_ids)
        graph.set_outputs(output_ids)
        
        return graph
    
 #ex 9 td 3 qui     Définir une méthode qui, lorsqu'appliquée à un graphe à n nœuds, renvoie un dictionnaire associant à chaque id de nœud un unique entier 0 ≤ i < n

    

    def get_id_index_map(self):
        """
        Renvoie un dictionnaire associant à chaque id de nœud un unique entier 0 ≤ i < n.
        
        Retourne : dict[int, int]; le dictionnaire des indices.
        """
        return {node_id: index for index, node_id in enumerate(self.get_node_ids())}
    
    #ex 10 du td 3 qui Définir une méthode adjacency_matrix qui donne une matrice d'adjacence du graphe
    def adjacency_matrix(self):
        """
        Donne une matrice d'adjacence du graphe.
        
        Retourne : list[list[int]]; la matrice d'adjacence.
        """
        n = len(self.nodes)
        matrix = [[0] * n for _ in range(n)]
        id_index_map = self.get_id_index_map()
        
        for node_id, node in self.nodes.items():
            for child_id, multiplicity in node.get_children().items():
                matrix[id_index_map[node_id]][id_index_map[child_id]] = multiplicity
        
        return matrix
    #ex 11 td 3 qui 
    

    #ex 1 td 4 : Pour implémenter la méthode save_as_dot_file
    def save_as_dot_file(self, path, verbose=False):
        with open(path, 'w') as file:
            file.write('digraph G {\n')
            
            # Écriture des nœuds
            for node in self.get_nodes():
                label = f'label="{node.get_label()}"' if verbose else ''
                file.write(f'  {node.get_id()} [{label}];\n')
            
            # Écriture des arêtes
            for node in self.get_nodes():
                for child_id, multiplicity in node.get_children().items():
                    for _ in range(multiplicity):
                        file.write(f'  {node.get_id()} -> {child_id};\n')
            
            file.write('}\n')
          


    #exercic 2 :une méthode from_dot_file pour lire un fichier .dot et créer un graphe
    @classmethod
    def from_dot_file(cls, path):
        with open(path, 'r') as file:
            content = file.read()

        nodes = {}
        inputs = []
        outputs = []

        node_pattern = re.compile(r'(\d+) \[label="(.*?)"\];')
        edge_pattern = re.compile(r'(\d+) -> (\d+);')

        for match in node_pattern.finditer(content):
            node_id = int(match.group(1))
            label = match.group(2)
            nodes[node_id] = node(node_id, label, {}, {})

        for match in edge_pattern.finditer(content):
            src = int(match.group(1))
            tgt = int(match.group(2))
            nodes[src].add_child_id(tgt)
            nodes[tgt].add_parent_id(src)

        for node_id, node in nodes.items():
            if len(node.get_parents()) == 0:
                inputs.append(node_id)
            if len(node.get_children()) == 0:
                outputs.append(node_id)

        return cls(inputs, outputs, list(nodes.values()))
    
  #exercice 3 td 4   : méthode display pour afficher le graphe
    def save_as_dot_file(self, path, verbose=False):
        with open(path, 'w') as file:
            file.write('digraph G {\n')
            for node_id, node in self.nodes.items():
                label = f' [label="{node.get_label()}"]' if verbose else ''
                file.write(f'  {node_id}{label};\n')
            for node_id, node in self.nodes.items():
                for child_id in node.get_children():
                    file.write(f'  {node_id} -> {child_id};\n')
            file.write('}\n')
    
    def display(self, verbose=False):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dot') as temp_file:
            self.save_as_dot_file(temp_file.name, verbose=verbose)
            temp_dot_file = temp_file.name
        
        output_file = temp_dot_file.replace('.dot', '.pdf')
        os.system(f'dot -Tpdf {temp_dot_file} -o {output_file}')
        webbrowser.open(output_file)
      #ex 3 td4 : Ajouter la méthode is_cyclic pour open_digraph
    def is_cyclic(self):
        visited = set()
        stack = set()

        def visit(node_id):
            if node_id in stack:
                return True
            if node_id in visited:
                return False
            stack.add(node_id)
            visited.add(node_id)
            for child_id in self.get_node_by_id(node_id).get_children():
                if visit(child_id):
                    return True
            stack.remove(node_id)
            return False

        for node_id in self.get_node_ids():
            if visit(node_id):
                return True
        return False
#ex 6 et 7 td 5 
    def min_id(self):
        """Renvoie l'indice minimum des noeuds du graphe."""
        if not self.nodes:
            return None
        return min(self.nodes.keys())

    def max_id(self):
        """Renvoie l'indice maximum des noeuds du graphe."""
        if not self.nodes:
            return None
        return max(self.nodes.keys())

    def shift_indices(self, n):
        """Ajoute n à tous les indices du graphe."""
        new_nodes = {}
        for node_id, node_obj in self.nodes.items():
            new_node_id = node_id + n
            new_parents = {k + n: v for k, v in node_obj.get_parents().items()}
            new_children = {k + n: v for k, v in node_obj.get_children().items()}
            new_node = node(new_node_id, node_obj.get_label(), new_parents, new_children)
            new_nodes[new_node_id] = new_node

        self.nodes = new_nodes
        self.inputs = [i + n for i in self.inputs]
        self.outputs = [o + n for o in self.outputs]



    def min_id(self):
        """
        Retourne l'id minimum parmi les noeuds du graphe.
        """
        return min(self.get_node_ids())

    def max_id(self):
        """
        Retourne l'id maximum parmi les noeuds du graphe.
        """
        return max(self.get_node_ids())

    def shift_indices(self, n):
        """
        Ajoute n à tous les indices des noeuds du graphe.
        """
        new_nodes = {}
        for node_id in self.get_node_ids():
            new_node_id = node_id + n
            node = self.get_node_by_id(node_id)
            new_parents = {pid + n: multiplicity for pid, multiplicity in node.get_parents().items()}
            new_children = {cid + n: multiplicity for cid, multiplicity in node.get_children().items()}
            new_node = node.__class__(new_node_id, node.get_label(), new_parents, new_children)
            new_nodes[new_node_id] = new_node
        self.nodes = new_nodes

        # Mettre à jour les inputs et outputs
        self.inputs = [i + n for i in self.inputs]
        self.outputs = [o + n for o in self.outputs]  

        

    @classmethod
    def identity(cls, n):
        """
        Crée un open_digraph représentant l'identité sur n fils.
        """
        nodes = [node(i, '', {}, {i + n: 1}) for i in range(n)]
        nodes += [node(i + n, '', {i: 1}, {}) for i in range(n)]
        inputs = list(range(n))
        outputs = list(range(n, 2 * n))
        return cls(inputs, outputs, nodes)
# exercices 4 et 5, nous devons ajouter les méthodes connected_components et extract_components à la classe open_digraph. Ces méthodes permettent de déterminer les composantes connexes du graphe et d'extraire chaque composante sous forme de nouveau graphe

    def connected_components(self):
        """
        Renvoie le nombre de composantes connexes et un dictionnaire associant chaque id de nœud à un numéro de composante connexe.
        """
        def dfs(node_id, component_id):
            stack = [node_id]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    components[current] = component_id
                    stack.extend(self.nodes[current].get_parents().keys())
                    stack.extend(self.nodes[current].get_children().keys())

        visited = set()
        components = {}
        component_id = 0

        for node_id in self.get_node_ids():
            if node_id not in visited:
                dfs(node_id, component_id)
                component_id += 1

        return component_id, components

    def extract_components(self):
        """
        Renvoie une liste de graphes, chacun correspondant à une composante connexe du graphe.
        """
        num_components, components = self.connected_components()
        subgraphs = {i: [] for i in range(num_components)}

        for node_id, component_id in components.items():
            subgraphs[component_id].append(self.nodes[node_id])

        result = []
        for nodes in subgraphs.values():
            inputs = [node.get_id() for node in nodes if node.get_id() in self.inputs]
            outputs = [node.get_id() for node in nodes if node.get_id() in self.outputs]
            result.append(open_digraph(inputs, outputs, nodes))

        return result
 #ex 1 td 7 
    def dijkstra(self, src, direction=1):
        """
        Implements Dijkstra's algorithm for shortest paths in the graph.
        src : int - the starting node id
        direction : int - 1 for forward direction (children), -1 for reverse direction (parents)
        Returns a dictionary of shortest distances from src to each node.
        """
        dist = {node_id: float('inf') for node_id in self.get_node_ids()}
        dist[src] = 0
        prev = {}
        Q = [(0, src)]
        
        while Q:
            current_distance, u = heapq.heappop(Q)
            
            if current_distance > dist[u]:
                continue
            
            neighbours = self.nodes[u].get_children() if direction == 1 else self.nodes[u].get_parents()
            
            for v, weight in neighbours.items():
                distance = current_distance + weight
                if distance < dist[v]:
                    dist[v] = distance
                    prev[v] = u
                    heapq.heappush(Q, (distance, v))
        
        return dist, prev
    
#ex 2 td 7 :Modification for Target Node
    def dijkstra(self, src, tgt=None, direction=1):
        dist = {node_id: float('inf') for node_id in self.get_node_ids()}
        dist[src] = 0
        prev = {}
        Q = [(0, src)]
        
        while Q:
            current_distance, u = heapq.heappop(Q)
            
            if u == tgt:
                break
            
            if current_distance > dist[u]:
                continue
            
            neighbours = self.nodes[u].get_children() if direction == 1 else self.nodes[u].get_parents()
            
            for v, weight in neighbours.items():
                distance = current_distance + weight
                if distance < dist[v]:
                    dist[v] = distance
                    prev[v] = u
                    heapq.heappush(Q, (distance, v))
        
        return dist, prev

    def common_ancestors(self, node1, node2):
        dist1, _ = self.dijkstra(node1, direction=-1)
        dist2, _ = self.dijkstra(node2, direction=-1)

        common_ancestors = {}
        for node in dist1:
            if node in dist2:
                common_ancestors[node] = (dist1[node], dist2[node])
        
        return common_ancestors

    
    #td 8:
    #Cette méthode réalise un tri topologique du graphe en utilisant l'algorithme de Kahn.


    def topological_sort(self):
        """
        Effectue un tri topologique du graphe.
        Renvoie une liste des identifiants des nœuds dans l'ordre topologique.
        Lève une erreur si le graphe contient un cycle.
        """
        in_degree = {node_id: 0 for node_id in self.get_node_ids()}
        for node in self.get_nodes():
            for child_id in node.get_children():
                if child_id in in_degree:
                    in_degree[child_id] += 1
                else:
                    in_degree[child_id] = 1

        zero_in_degree_queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        topological_order = []

        while zero_in_degree_queue:
            u = zero_in_degree_queue.pop(0)
            topological_order.append(u)

            for v in self.get_node_by_id(u).get_children():
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    zero_in_degree_queue.append(v)

        if len(topological_order) != len(self.get_node_ids()):
            raise ValueError("Le graphe contient un cycle, le tri topologique n'est pas possible.")

        return topological_order

    #Cette méthode réalise un tri topologique compressé, c'est-à-dire qu'elle regroupe les nœuds au même niveau dans des ensembles
    def compressed_topological_sort(self):
        """
        Effectue un tri topologique compressé du graphe.
        Renvoie une liste d'ensembles, chaque ensemble contenant les nœuds au même niveau.
        """
        topological_order = self.topological_sort()
        levels = {}
        for node_id in topological_order:
            max_level = -1
            for parent_id in self.get_node_by_id(node_id).get_parents():
                max_level = max(max_level, levels[parent_id])
            levels[node_id] = max_level + 1

        level_sets = {}
        for node_id, level in levels.items():
            if level not in level_sets:
                level_sets[level] = set()
            level_sets[level].add(node_id)

        sorted_levels = [level_sets[level] for level in sorted(level_sets)]
        return sorted_levels
    
    #Cette méthode vérifie si le graphe est acyclique (ne contient pas de cycles) en utilisant le tri topologique
    def is_acyclic(self):
        """
        Vérifie si le graphe est acyclique.
        Renvoie True si le graphe est acyclique, False sinon.
        """
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    #Exercice 2: Implémenter une méthode qui retourne la profondeur d’un nœud donné dans un graphe
    def depth(self, node_id):
        """
        Calcule la profondeur d'un nœud donné dans un graphe acyclique.
        node_id : int - l'identifiant du nœud
        Retourne un entier représentant la profondeur du nœud.
        """
        topological_order = self.topological_sort()
        depth_map = {node_id: -1 for node_id in topological_order}

        for node in topological_order:
            if node in self.get_input_ids():
                depth_map[node] = 0
            else:
                parents = self.get_node_by_id(node).get_parents().keys()
                if parents:
                    depth_map[node] = max([depth_map[parent] for parent in parents]) + 1
                else:
                    depth_map[node] = 0  # Si le nœud n'a pas de parents, sa profondeur est 0

        return depth_map[node_id]

    def graph_depth(self):
        """
        Calcule la profondeur du graphe entier.
        Retourne un entier représentant la profondeur maximale de tous les nœuds.
        """
        topological_order = self.topological_sort()
        depth_map = {node_id: -1 for node_id in topological_order}

        for node in topological_order:
            if node in self.get_input_ids():
                depth_map[node] = 0
            else:
                parents = self.get_node_by_id(node).get_parents().keys()
                if parents:
                    depth_map[node] = max([depth_map[parent] for parent in parents]) + 1
                else:
                    depth_map[node] = 0  # Si le nœud n'a pas de parents, sa profondeur est 0

        return max(depth_map.values())


    #Exercice 3: Implémenter une méthode qui calcule le chemin et la distance maximum d’un nœud u à un nœud v
    def longest_path(self, src, tgt):
        """
        Calcule le plus long chemin d'un nœud src à un nœud tgt dans un graphe acyclique.
        src : int - l'identifiant du nœud source
        tgt : int - l'identifiant du nœud cible
        Retourne la distance maximale et le chemin sous forme de liste de nœuds.
        """
        topological_order = self.topological_sort()
        dist = {node_id: float('-inf') for node_id in self.get_node_ids()}
        prev = {node_id: None for node_id in self.get_node_ids()}

        dist[src] = 0

        for node in topological_order:
            if node == tgt:
                break
            for child, weight in self.get_node_by_id(node).get_children().items():
                if dist[child] < dist[node] + weight:
                    dist[child] = dist[node] + weight
                    prev[child] = node

        # Reconstruire le chemin
        path = []
        node = tgt
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        return dist[tgt], path

    #ex 2 td 9 
    def fusion(self, id1, id2, new_label):
        """
        Fusionne deux noeuds en un seul avec une nouvelle étiquette.
        """
        if id1 not in self.nodes or id2 not in self.nodes:
            raise ValueError("Les identifiants de noeuds doivent être valides")

        node1 = self.nodes[id1]
        node2 = self.nodes[id2]

        # Fusion des parents et enfants
        new_parents = node1.get_parents()
        for parent_id, multiplicity in node2.get_parents().items():
            if parent_id in new_parents:
                new_parents[parent_id] += multiplicity
            else:
                new_parents[parent_id] = multiplicity

        new_children = node1.get_children()
        for child_id, multiplicity in node2.get_children().items():
            if child_id in new_children:
                new_children[child_id] += multiplicity
            else:
                new_children[child_id] = multiplicity

        # Création du nouveau noeud fusionné
        new_node = node(id1, new_label, new_parents, new_children)

        # Mise à jour des parents et enfants pour pointer vers le nouveau noeud
        for parent_id in new_parents:
            self.nodes[parent_id].add_child_id(id1, new_parents[parent_id])
            self.nodes[parent_id].remove_child_id(id2)
        for child_id in new_children:
            self.nodes[child_id].add_parent_id(id1, new_children[child_id])
            self.nodes[child_id].remove_parent_id(id2)

        # Remplacer le noeud id1 par le nouveau noeud fusionné
        self.nodes[id1] = new_node

        # Supprimer le noeud id2
        del self.nodes[id2]


    @classmethod
    def from_multiple_formulas(cls, *args):
        combined_g = cls.empty()
        input_vars = []
        output_nodes = []

        for formula in args:
            g, vars = cls.from_formula(formula)
           # shift_value = combined_g.max_id() + 1

            g.shift_indices(shift_value)
            combined_g.nodes.update(g.get_id_node_map())
            input_vars.extend(vars)
            output_nodes.append(g.get_output_ids()[0])

        combined_g.set_inputs([node.get_id() for node in combined_g.get_nodes() if node.get_label().startswith('x')])
        combined_g.set_outputs(output_nodes)
        return combined_g

    #ex 2 td 12:es règles de réécriture dans le contexte des circuits booléens ou des graphes dirigés sont utilisées pour simplifier et optimiser ces circuits ou graphes
    def apply_xor_associativity(self):
        changes = False
        for node in list(self.get_nodes()):  # Utilisez une copie de la liste des nœuds
            if node.get_label() == '^':
                children_ids = node.get_children_ids()
                if len(children_ids) == 2:
                    left_child, right_child = children_ids
                    left_node = self.get_node_by_id(left_child)
                    
                    if left_node.get_label() == '^':
                        grand_children_ids = left_node.get_children_ids()
                        for grand_child_id in grand_children_ids:
                            self.add_edge(node.get_id(), grand_child_id)
                        self.remove_edge(left_child, node.get_id())
                        self.remove_node(left_child)
                        changes = True
        return changes

    def apply_copy_associativity(self):
        changes = False
        for node in list(self.get_nodes()):  # Utilisez une copie de la liste des nœuds
            if node.get_label() == 'copy':
                children_ids = node.get_children_ids()
                if len(children_ids) == 2:
                    left_child, right_child = children_ids
                    left_node = self.get_node_by_id(left_child)
                    
                    if left_node.get_label() == 'copy':
                        grand_children_ids = left_node.get_children_ids()
                        for grand_child_id in grand_children_ids:
                            self.add_edge(node.get_id(), grand_child_id)
                        self.remove_edge(left_child, node.get_id())
                        self.remove_node(left_child)
                        changes = True
        return changes

    def apply_not_through_xor(self):
        changes = False
        for node in list(self.get_nodes()):  # Utilisez une copie de la liste des nœuds
            if node.get_label() == '~':
                children_ids = node.get_children_ids()
                if len(children_ids) == 1:
                    child_id = children_ids[0]
                    child_node = self.get_node_by_id(child_id)
                    
                    if child_node.get_label() == '^':
                        grand_children_ids = child_node.get_children_ids()
                        for grand_child_id in grand_children_ids:
                            not_node_id = self.add_node(label='~')
                            self.add_edge(not_node_id, grand_child_id)
                            self.add_edge(node.get_id(), not_node_id)
                        self.remove_edge(node.get_id(), child_id)
                        self.remove_node(child_id)
                        changes = True
        return changes

    def apply_not_through_copy(self):
        changes = False
        for node in list(self.get_nodes()):  # Utilisez une copie de la liste des nœuds
            if node.get_label() == '~':
                children_ids = node.get_children_ids()
                if len(children_ids) == 1:
                    child_id = children_ids[0]
                    child_node = self.get_node_by_id(child_id)
                    
                    if child_node.get_label() == 'copy':
                        grand_children_ids = child_node.get_children_ids()
                        for grand_child_id in grand_children_ids:
                            not_node_id = self.add_node(label='~')
                            self.add_edge(not_node_id, grand_child_id)
                            self.add_edge(node.get_id(), not_node_id)
                        self.remove_edge(node.get_id(), child_id)
                        self.remove_node(child_id)
                        changes = True
        return changes

    def apply_rewriting_rules(self):
        changes = True
        while changes:
            changes = False
            changes |= self.apply_xor_associativity()
            changes |= self.apply_copy_associativity()
            changes |= self.apply_not_through_xor()
            changes |= self.apply_not_through_copy()

    
    #ex 4 td 12 :Tests Unitaires pour les Règles de Réécriture 
    def apply_rewriting_rules(self):
        changes = False
        for node in self.get_nodes():
            if node.get_label() == '&':
                parents = node.get_parent_ids()
                if len(parents) == 2 and parents[0] == parents[1]:
                    node.set_label(self.get_node_by_id(parents[0]).get_label())
                    changes = True
            elif node.get_label() == '|':
                parents = node.get_parent_ids()
                if len(parents) == 2 and parents[0] == parents[1]:
                    node.set_label(self.get_node_by_id(parents[0]).get_label())
                    changes = True
            elif node.get_label() == '&' and '~' in [self.get_node_by_id(p).get_label() for p in node.get_parent_ids()]:
                node.set_label('0')
                changes = True
            elif node.get_label() == '|' and '~' in [self.get_node_by_id(p).get_label() for p in node.get_parent_ids()]:
                node.set_label('1')
                changes = True
        return changes

    def apply_all_rewriting_rules(self):
        while self.apply_rewriting_rules():
            pass

class bool_circ(open_digraph):
    valid_labels = {'&', '|', '~', '', '0', '1', '^', 'a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3', 'c0', 'c1', 'c2', 'c3', 'c4', 'i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6','d0','d1','d2','d3'}

    def __init__(self, inputs, outputs, nodes):
        super().__init__(inputs, outputs, nodes)
        for node in self.get_nodes():
            label = node.get_label()
            if label not in self.valid_labels:
                raise ValueError(f"Étiquette non valide: {label}")

    


    def is_bool_circ(self):
        if not self.is_well_formed():
            return False
        for node in self.get_nodes():
            label = node.get_label()
            if label not in self.valid_labels:
                return False
        return True

    def is_well_formed(self):
        for node in self.get_nodes():
            label = node.get_label()
            if label == 'copy':
                if node.indegree() != 1:
                    return False
            elif label in ['&', '|']:
                if node.outdegree() != 1:
                    return False
            elif label == '~':
                if node.indegree() != 1 or node.outdegree() != 1:
                    return False
            elif label in ['0', '1', '^']:
                if node.degree() != 0:
                    return False
        return not self.is_cyclic()
    
    @classmethod
    def from_integer(cls, n, bit_length=8):
        g = cls.empty()
        binary_representation = bin(n)[2:].zfill(bit_length)
        input_node_ids = []
        for bit in binary_representation:
            label = '1' if bit == '1' else '0'
            node_id = g.add_node(label)
            input_node_ids.append(node_id)
        g.set_inputs(input_node_ids)
        return cls(g.get_input_ids(), [], g.get_nodes())

    @classmethod
    def parse_parentheses(cls, s):
        g = cls.empty()
        current_node = None
        stack = []
        s2 = ''
        
        for char in s:
            if char == '(':
                stack.append((current_node, s2))
                s2 = ''
            elif char == ')':
                new_node = g.add_node(label=s2)
                if current_node is not None:
                    g.add_edge(current_node, new_node)
                current_node, s2 = stack.pop()
                current_node = new_node
            else:
                s2 += char

        return g

    @classmethod
    def from_formula(cls, s):
        g = cls.parse_parentheses(s)
        variables = set(re.findall(r'x\d+', s))
        variable_nodes = {var: g.add_node(label=var) for var in variables}
        
        for node in g.get_nodes():
            if node.get_label() in variable_nodes:
                var_node = variable_nodes[node.get_label()]
                node.set_label('')
                g.add_edge(var_node, node)

        g.set_inputs([node for node in variable_nodes.values()])
        return g, list(variable_nodes.keys())

    @classmethod
    def parse_optimized_parentheses(cls, s):
        s = re.sub(r'\(([^()]+)\)', r'\1', s)  # Remove redundant parentheses
        return cls.parse_parentheses(s)

    @classmethod
    def from_optimized_formula(cls, s):
        s = re.sub(r'\(([^()]+)\)', r'\1', s)  # Remove redundant parentheses
        return cls.from_formula(s)

    @classmethod
    def random_bool_circ(cls, n, inputs_count, outputs_count):
        g = cls.random(n, 2)
        
        input_nodes = [node for node in g.get_nodes() if node.indegree() == 0]
        output_nodes = [node for node in g.get_nodes() if node.outdegree() == 0]

        # Assurer suffisamment d'entrées
        while len(input_nodes) < inputs_count:
            new_input = g.add_node(label='', children={random.choice(list(g.nodes.keys())): 1})
            input_nodes.append(g.get_node_by_id(new_input))

        # Réduire le nombre d'entrées si nécessaire
        while len(input_nodes) > inputs_count:
            node_to_convert = random.choice(input_nodes)
            new_input = g.add_node(label='', children={node_to_convert.get_id(): 1})
            input_nodes.remove(node_to_convert)
            input_nodes.append(g.get_node_by_id(new_input))
        
        # Assurer suffisamment de sorties
        while len(output_nodes) < outputs_count:
            new_output = g.add_node(label='', parents={random.choice(list(g.nodes.keys())): 1})
            output_nodes.append(g.get_node_by_id(new_output))

        # Réduire le nombre de sorties si nécessaire
        while len(output_nodes) > outputs_count:
            node_to_convert = random.choice(output_nodes)
            new_output = g.add_node(label='', parents={node_to_convert.get_id(): 1})
            output_nodes.remove(node_to_convert)
            output_nodes.append(g.get_node_by_id(new_output))

        # Étiquetage des nœuds
        for node in g.get_nodes():
            if node.indegree() == 1 and node.outdegree() == 1:
                node.set_label('~')
            elif node.indegree() > 1 and node.outdegree() == 1:
                node.set_label(random.choice(['&', '|', '^']))
            elif node.indegree() == 1 and node.outdegree() > 1:
                node.set_label('copy')
            elif node.indegree() == 0 or node.outdegree() == 0:
                node.set_label(random.choice(['0', '1']))

        # Définition des entrées et sorties
        g.set_inputs([node.get_id() for node in input_nodes])
        g.set_outputs([node.get_id() for node in output_nodes])

        return cls(g.get_input_ids(), g.get_output_ids(), g.get_nodes())
#ex3 td 10 : 
#Un Adder_n est un circuit qui prend en entrée deux registres de taille 𝑛n (a et b) et un bit de retenue (carry). Il produit en sortie un registre de taille 𝑛n contenant la somme des deux nombres donnés en entrée, modulo 2𝑛2 n , et un bit de retenue indiquant si le calcul a dépassé la taille du registre

    @classmethod
    def adder_n(cls, n):
        """
        Construit un circuit Adder_n de taille n.
        """
        g = cls.empty()
        
        # Créer les noeuds d'entrée
        a_inputs = [g.add_node('&') for i in range(n)]
        b_inputs = [g.add_node('&') for i in range(n)]
        c_input = g.add_node('&')
        
        # Initialiser les listes de sorties
        s_outputs = []
        c_outputs = [c_input]  # Commence avec la retenue initiale

        # Construire les Adders
        for i in range(n):
            a = a_inputs[i]
            b = b_inputs[i]
            c = c_outputs[-1]  # Retenue précédente

            # Noeuds intermédiaires pour la somme et la nouvelle retenue
            s = g.add_node('^')
            carry_and = g.add_node('&')
            or_node = g.add_node('|')

            # Ajouter les arêtes
            g.add_edge(a, s)
            g.add_edge(b, s)
            g.add_edge(c, s)

            g.add_edge(a, carry_and)
            g.add_edge(b, carry_and)

            g.add_edge(carry_and, or_node)
            g.add_edge(c, or_node)

            # Ajouter les noeuds de sortie
            s_outputs.append(s)
            c_outputs.append(or_node)
        
        return cls(a_inputs + b_inputs + [c_input], s_outputs + [c_outputs[-1]], g.get_nodes())
#Un Half_Adder_n est un circuit similaire à Adder_n, mais avec la retenue initiale fixée à 0.
    @classmethod
    def half_adder_n(cls, n):
        """
        Construit un circuit Half_Adder_n de taille n.
        """
        g = cls.adder_n(n)  # Utilise adder_n pour construire le half_adder
        zero_node = g.add_node('0')  # Ajoute le bit de retenue initialisé à 0
        g.add_edge(zero_node, g.get_node_by_id(g.get_input_ids()[-1]))  # Connecte le bit de retenue initialisé à 0

        return g
#ex 4 td 10 :La méthode carry_lookahead_adder_4 créera un circuit carry-lookahead pour des registres de taille 4

    @classmethod
    def carry_lookahead_adder_4(cls):
        g = cls.empty()

        # Création des inputs
        a_inputs = [g.add_node('a' + str(i)) for i in range(4)]
        b_inputs = [g.add_node('b' + str(i)) for i in range(4)]
        c_input = g.add_node('c0')

        # Création des gates Gi et Pi
        g_nodes = [g.add_node('&') for _ in range(4)]
        p_nodes = [g.add_node('|') for _ in range(4)]

        for i in range(4):
            g.add_edge(a_inputs[i], g_nodes[i])
            g.add_edge(b_inputs[i], g_nodes[i])
            g.add_edge(a_inputs[i], p_nodes[i])
            g.add_edge(b_inputs[i], p_nodes[i])

        # Création des outputs pour les Gi et Pi
        carry_outputs = []
        for i in range(1, 5):
            carry = g.add_node('c' + str(i))
            carry_outputs.append(carry)

        # Création des XOR pour les sum
        s_outputs = [g.add_node('^') for _ in range(4)]
        for i in range(4):
            g.add_edge(p_nodes[i], s_outputs[i])
            g.add_edge(g_nodes[i], s_outputs[i])

        # Connexion des nodes Carry
        g.add_edge(c_input, carry_outputs[0])
        for i in range(1, 4):
            g.add_edge(carry_outputs[i-1], carry_outputs[i])

        return cls(a_inputs + b_inputs + [c_input], s_outputs + [carry_outputs[-1]], g.get_nodes())


    @classmethod
    def carry_lookahead_adder_4n(cls, n):
        g = cls.empty()

        # Pour chaque bloc de taille 4
        for i in range(n):
            cla4 = cls.carry_lookahead_adder_4()
            g.nodes.update(cla4.get_id_node_map())
            if i > 0:
                prev_carry_output = g.get_node_by_id(carry_output)
                current_carry_input = g.get_node_by_id(cla4.get_input_ids()[-1])
                g.add_edge(prev_carry_output.get_id(), current_carry_input.get_id())

            # Update carry_output
            carry_output = cla4.get_output_ids()[-1]

        return cls(g.get_input_ids(), g.get_output_ids(), g.get_nodes())

    @classmethod
    def estimate_depth_and_gates(cls, n):
        """
        Estime la profondeur et le nombre de portes du circuit carry-lookahead adder pour des registres de taille 4n.
        """
        # Estimation de la profondeur
        depth_per_block = 5  # Une profondeur constante pour chaque bloc carry-lookahead de taille 4
        total_depth = depth_per_block

        # Estimation du nombre de portes
        gates_per_block = 25  # Par exemple, si chaque bloc utilise 25 portes
        total_gates = n * gates_per_block

        return total_depth, total_gates

    
    #ex3 td 11 : Ces transformations simplifient le circuit booléen en appliquant des règles logiques pour évaluer les sorties basées sur les entrées et les opérateurs logiques. Chaque transformation repose sur des règles de la logique booléenne pour simplifier ou évaluer les sous-expressions du circuit
    def apply_transformations(self):
        for node in self.get_nodes():
            label = node.get_label()
            if label in {'0', '1'}:
                self.transform_constant(node)
            elif label == '~':
                self.transform_not(node)
            elif label == '&':
                self.transform_and(node)
            elif label == '|':
                self.transform_or(node)
            elif label == '^':
                self.transform_xor(node)

    def transform_constant(self, node):
        for child_id in node.get_children_ids():
            self.remove_edge(node.get_id(), child_id)
            child = self.get_node_by_id(child_id)
            if node.get_label() == '0':
                child.set_label('0')
            elif node.get_label() == '1':
                child.set_label('1')

    def transform_not(self, node):
        for child_id in node.get_children_ids():
            self.remove_edge(node.get_id(), child_id)
            child = self.get_node_by_id(child_id)
            child.set_label('1' if node.get_label() == '0' else '0')

    def transform_and(self, node):
        parent_labels = [self.get_node_by_id(parent_id).get_label() for parent_id in node.get_parent_ids()]
        if '0' in parent_labels:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('0')
        elif '1' in parent_labels:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('1')

    def transform_or(self, node):
        parent_labels = [self.get_node_by_id(parent_id).get_label() for parent_id in node.get_parent_ids()]
        if '1' in parent_labels:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('1')
        elif '0' in parent_labels:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('0')

    def transform_xor(self, node):
        parent_labels = [self.get_node_by_id(parent_id).get_label() for parent_id in node.get_parent_ids()]
        if parent_labels.count('1') % 2 == 0:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('0')
        else:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('1')

#exercice 4:la méthode evaluate(self) dans la classe bool_circ pour appliquer les transformations jusqu'à ce qu'il n'y ait plus de transformations à appliquer. Ensuite, nous allons tester cette méthode sur l'additionneur
    def evaluate(self):
        changes = True
        iterations = 0
        max_iterations = 100
        while changes and iterations < max_iterations:
            changes = False
            iterations += 1
            print(f"Iteration {iterations}")
            for node in self.get_nodes():
                label = node.get_label()
                print(f"Evaluating node {node.get_id()} with label {label}")
                if label in {'0', '1'}:
                    changes = self.transform_constant(node) or changes
                elif label == '~':
                    changes = self.transform_not(node) or changes
                elif label == '&':
                    changes = self.transform_and(node) or changes
                elif label == '|':
                    changes = self.transform_or(node) or changes
                elif label == '^':
                    changes = self.transform_xor(node) or changes
                
        if iterations >= max_iterations:
            print("Reached maximum iterations")

    def transform_constant(self, node):
        transformed = False
        for child_id in node.get_children_ids():
            self.remove_edge(node.get_id(), child_id)
            child = self.get_node_by_id(child_id)
            if node.get_label() == '0':
                child.set_label('0')
            elif node.get_label() == '1':
                child.set_label('1')
            transformed = True
        if transformed:
            node.set_label('')  # Clear the label after transformation
        return transformed

    def transform_not(self, node):
        transformed = False
        for child_id in node.get_children_ids():
            self.remove_edge(node.get_id(), child_id)
            child = self.get_node_by_id(child_id)
            if node.get_label() == '0':
                child.set_label('1')
            elif node.get_label() == '1':
                child.set_label('0')
            transformed = True
        if transformed:
            node.set_label('')  # Clear the label after transformation
        return transformed

    def transform_and(self, node):
        parent_labels = [self.get_node_by_id(parent_id).get_label() for parent_id in node.get_parent_ids()]
        transformed = False
        if '0' in parent_labels:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('0')
            transformed = True
        elif all(label == '1' for label in parent_labels):
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('1')
            transformed = True
        if transformed:
            node.set_label('')  # Clear the label after transformation
        return transformed

    def transform_or(self, node):
        parent_labels = [self.get_node_by_id(parent_id).get_label() for parent_id in node.get_parent_ids()]
        transformed = False
        if '1' in parent_labels:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('1')
            transformed = True
        elif all(label == '0' for label in parent_labels):
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('0')
            transformed = True
        if transformed:
            node.set_label('')  # Clear the label after transformation
        return transformed

    def transform_xor(self, node):
        parent_labels = [self.get_node_by_id(parent_id).get_label() for parent_id in node.get_parent_ids()]
        transformed = False
        if parent_labels.count('1') % 2 == 0:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('0')
            transformed = True
        else:
            for child_id in node.get_children_ids():
                self.remove_edge(node.get_id(), child_id)
                self.get_node_by_id(child_id).set_label('1')
            transformed = True
        if transformed:
            node.set_label('')  # Clear the label after transformation
        return transformed
    
    
    #ex 1 td 12 :
    @classmethod
    def encoder_hamming_7_4(cls):
        """
        Construit un circuit booléen pour l'encodeur Hamming (7,4).
        """
        g = cls.empty()
        
        # Création des noeuds d'entrée
        input_ids = [g.add_node(label=f'd{i}') for i in range(4)]
        
        # Création des noeuds pour les bits de parité
        p1_id = g.add_node(label='^')
        p2_id = g.add_node(label='^')
        p3_id = g.add_node(label='^')
        
        # Connections pour les bits de parité
        g.add_edge(input_ids[0], p1_id)
        g.add_edge(input_ids[1], p1_id)
        g.add_edge(input_ids[3], p1_id)
        
        g.add_edge(input_ids[0], p2_id)
        g.add_edge(input_ids[2], p2_id)
        g.add_edge(input_ids[3], p2_id)
        
        g.add_edge(input_ids[1], p3_id)
        g.add_edge(input_ids[2], p3_id)
        g.add_edge(input_ids[3], p3_id)
        
        # Ajout des bits de parité aux sorties
        output_ids = [p1_id, p2_id, input_ids[0], p3_id, input_ids[1], input_ids[2], input_ids[3]]
        
        # Définition des entrées et des sorties du circuit
        g.set_inputs(input_ids)
        g.set_outputs(output_ids)
        
        return cls(g.get_input_ids(), g.get_output_ids(), g.get_nodes())

    @classmethod
    def decoder_hamming_7_4(cls):
        """
        Construit un circuit booléen pour le décodeur Hamming (7,4).
        """
        g = cls.empty()
        
        # Création des noeuds d'entrée
        input_ids = [g.add_node(label=f'i{i}') for i in range(7)]
        
        # Création des portes logiques pour la détection d'erreurs
        s1_id = g.add_node(label='^')
        s2_id = g.add_node(label='^')
        s3_id = g.add_node(label='^')
        
        g.add_edge(input_ids[0], s1_id)
        g.add_edge(input_ids[2], s1_id)
        g.add_edge(input_ids[4], s1_id)
        g.add_edge(input_ids[6], s1_id)
        
        g.add_edge(input_ids[1], s2_id)
        g.add_edge(input_ids[2], s2_id)
        g.add_edge(input_ids[5], s2_id)
        g.add_edge(input_ids[6], s2_id)
        
        g.add_edge(input_ids[3], s3_id)
        g.add_edge(input_ids[4], s3_id)
        g.add_edge(input_ids[5], s3_id)
        g.add_edge(input_ids[6], s3_id)
        
        # Noeud de correction (co-feuille de 0)
        correction_node_id = g.add_node(label='0')
        
        # Création des portes logiques pour les bits d'information corrigés
        d_ids = [g.add_node(label='^') for _ in range(4)]
        
        g.add_edge(input_ids[2], d_ids[0])
        g.add_edge(correction_node_id, d_ids[0])
        
        g.add_edge(input_ids[4], d_ids[1])
        g.add_edge(correction_node_id, d_ids[1])
        
        g.add_edge(input_ids[5], d_ids[2])
        g.add_edge(correction_node_id, d_ids[2])
        
        g.add_edge(input_ids[6], d_ids[3])
        g.add_edge(correction_node_id, d_ids[3])
        
        # Définition des sorties du circuit (bits d'information corrigés)
        output_ids = [d_id for d_id in d_ids]
        
        # Définition des entrées et des sorties du circuit
        g.set_inputs(input_ids)
        g.set_outputs(output_ids)
        
        return cls(g.get_input_ids(), g.get_output_ids(), g.get_nodes())
    

    
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()   
    
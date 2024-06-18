import sys
import os
import unittest
import re
import tempfile

# Ajout du répertoire racine du projet au chemin de recherche des modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from modules.utils import random_int_list, random_int_matrix, random_symmetric_int_matrix, random_oriented_int_matrix
from modules.open_digraph import node, open_digraph, bool_circ

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Création des graphes nécessaires pour les tests
        n0 = node(0, 'a', {}, {1: 1, 2: 1})
        n1 = node(1, 'b', {0: 1}, {3: 1})
        n2 = node(2, 'c', {0: 1}, {3: 1})
        n3 = node(3, 'd', {1: 1, 2: 1}, {})
        self.g1 = open_digraph([0], [3], [n0, n1, n2, n3])

        m0 = node(0, 'x', {}, {1: 1})
        m1 = node(1, 'y', {0: 1}, {})
        self.g2 = open_digraph([0], [1], [m0, m1])

        # Création d'un autre graphe pour les tests de profondeur et de chemin
        n0 = node(0, '', {}, {1: 1})
        n1 = node(1, '', {0: 1}, {2: 1, 3: 1})
        n2 = node(2, '', {1: 1}, {4: 1})
        n3 = node(3, '', {1: 1}, {4: 1})
        n4 = node(4, '', {2: 1, 3: 1}, {})
        self.g3 = open_digraph([], [], [n0, n1, n2, n3, n4])

    def test_random_int_list(self):
        lst = random_int_list(10, 100)
        self.assertEqual(len(lst), 10)
        for item in lst:
            self.assertTrue(0 <= item < 100)

    def test_random_int_matrix_without_null_diag(self):
        n = 5
        bound = 10
        null_diag = False
        matrix = random_int_matrix(n, bound, null_diag)
        self.assertEqual(len(matrix), n)
        for row in matrix:
            self.assertEqual(len(row), n)
        for row in matrix:
            for item in row:
                self.assertTrue(0 <= item < bound)
        for i in range(n):
            self.assertNotEqual(matrix[i][i], 0)

    def test_random_int_matrix_with_null_diag(self):
        n = 5
        bound = 10
        null_diag = True
        matrix = random_int_matrix(n, bound, null_diag)
        self.assertEqual(len(matrix), n)
        for row in matrix:
            self.assertEqual(len(row), n)
        for row in matrix:
            for item in row:
                self.assertTrue(0 <= item < bound)
        for i in range(n):
            self.assertEqual(matrix[i][i], 0)

    def test_random_symmetric_int_matrix(self):
        n = 5
        bound = 10
        null_diag = True
        matrix = random_symmetric_int_matrix(n, bound, null_diag)
        self.assertEqual(len(matrix), n)
        for row in matrix:
            self.assertEqual(len(row), n)
        for i in range(n):
            self.assertEqual(matrix[i][i], 0)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(matrix[i][j], matrix[j][i])

    def test_random_oriented_int_matrix(self):
        n = 5
        bound = 10
        null_diag = True
        matrix = random_oriented_int_matrix(n, bound, null_diag)
        self.assertEqual(len(matrix), n)
        for row in matrix:
            self.assertEqual(len(row), n)
        for i in range(n):
            self.assertEqual(matrix[i][i], 0)

    def test_save_as_dot_file(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        G = open_digraph([0], [1], [n0, n1])
        G.save_as_dot_file('test_graph.dot', verbose=True)
        with open('test_graph.dot', 'r') as file:
            content = file.read()
        expected_content = (
            'digraph G {\n'
            '  0 [label="i"];\n'
            '  1 [label="o"];\n'
            '  0 -> 1;\n'
            '}\n'
        )
        self.assertEqual(content, expected_content)
        os.remove('test_graph.dot')

    def test_display(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        G = open_digraph([0], [1], [n0, n1])
        G.display(verbose=True)

    def test_bool_circ_initialization(self):
        n0 = node(0, '&', {}, {1: 1})
        n1 = node(1, '|', {0: 1}, {2: 1})
        n2 = node(2, '~', {1: 1}, {3: 1})
        n3 = node(3, '0', {2: 1}, {})
        g = open_digraph([0], [3], [n0, n1, n2, n3])
        bc = bool_circ(g.get_input_ids(), g.get_output_ids(), g.get_nodes())
        self.assertEqual(bc.get_input_ids(), [0])
        self.assertEqual(bc.get_output_ids(), [3])
        self.assertEqual(len(bc.get_nodes()), 4)

    def test_bool_circ_invalid_label(self):
        n0 = node(0, 'invalid', {}, {})
        g = open_digraph([0], [], [n0])
        with self.assertRaises(ValueError):
            bc = bool_circ(g.get_input_ids(), g.get_output_ids(), g.get_nodes())

    def test_node_degrees(self):
        n = node(0, 'test', {1: 1, 2: 1}, {3: 1, 4: 1})
        self.assertEqual(n.indegree(), 2)
        self.assertEqual(n.outdegree(), 2)
        self.assertEqual(n.degree(), 4)

    def test_is_cyclic(self):
        n0 = node(0, 'copy', {}, {1: 1})
        n1 = node(1, '&', {0: 1}, {2: 1})
        n2 = node(2, '|', {1: 1}, {0: 1})  # cycle here
        g = open_digraph([0], [2], [n0, n1, n2])
        self.assertTrue(g.is_cyclic())
        n2 = node(2, '|', {1: 1}, {})  # no cycle
        g = open_digraph([0], [2], [n0, n1, n2])
        self.assertFalse(g.is_cyclic())

    def test_is_bool_circ(self):
        n0 = node(0, '&', {}, {1: 1})
        n1 = node(1, '|', {0: 1}, {2: 1})
        n2 = node(2, '~', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])
        bc = bool_circ(g.get_input_ids(), g.get_output_ids(), g.get_nodes())
       # self.assertTrue(bc.is_bool_circ())

        n2 = node(2, 'invalid', {}, {})
        g = open_digraph([0], [2], [n0, n1, n2])
        with self.assertRaises(ValueError):
            bool_circ(g.get_input_ids(), g.get_output_ids(), g.get_nodes())

    # Tests for Exercice 6 and Exercice 7
    def test_min_id(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        n2 = node(2, 'i', {}, {3: 1})
        n3 = node(3, 'o', {2: 1}, {})
        G = open_digraph([0], [1], [n0, n1, n2, n3])
        self.assertEqual(G.min_id(), 0)

    def test_max_id(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        n2 = node(2, 'i', {}, {3: 1})
        n3 = node(3, 'o', {2: 1}, {})
        G = open_digraph([0], [1], [n0, n1, n2, n3])
        self.assertEqual(G.max_id(), 3)

    def test_shift_indices(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        G = open_digraph([0], [1], [n0, n1])
        G.shift_indices(2)
        self.assertEqual(G.get_node_by_id(2).get_children(), {3: 1})
        self.assertEqual(G.get_node_by_id(3).get_parents(), {2: 1})
        self.assertEqual(G.get_input_ids(), [2])
        self.assertEqual(G.get_output_ids(), [3])
        G.shift_indices(-2)  # Shift back
        self.assertEqual(G.get_node_by_id(0).get_children(), {1: 1})
        self.assertEqual(G.get_node_by_id(1).get_parents(), {0: 1})
        self.assertEqual(G.get_input_ids(), [0])
        self.assertEqual(G.get_output_ids(), [1])

    def test_iparallel(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        g1 = open_digraph([0], [1], [n0, n1])

        n2 = node(0, 'i', {}, {1: 1})
        n3 = node(1, 'o', {0: 1}, {})
        g2 = open_digraph([0], [1], [n2, n3])

        g1.iparallel(g2)

        self.assertEqual(g1.get_node_by_id(0).get_children(), {1: 1})
        self.assertEqual(g1.get_node_by_id(1).get_parents(), {0: 1})
        self.assertEqual(g1.get_node_by_id(2).get_children(), {3: 1})
        self.assertEqual(g1.get_node_by_id(3).get_parents(), {2: 1})
        self.assertEqual(g1.get_input_ids(), [0, 2])
        self.assertEqual(g1.get_output_ids(), [1, 3])

    def test_parallel(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        g1 = open_digraph([0], [1], [n0, n1])

        n2 = node(0, 'i', {}, {1: 1})
        n3 = node(1, 'o', {0: 1}, {})
        g2 = open_digraph([0], [1], [n2, n3])

        g3 = g1.parallel(g2)

        self.assertEqual(g3.get_node_by_id(0).get_children(), {1: 1})
        self.assertEqual(g3.get_node_by_id(1).get_parents(), {0: 1})
        self.assertEqual(g3.get_node_by_id(2).get_children(), {3: 1})
        self.assertEqual(g3.get_node_by_id(3).get_parents(), {2: 1})
        self.assertEqual(g3.get_input_ids(), [0, 2])
        self.assertEqual(g3.get_output_ids(), [1, 3])

    def test_icompose(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        g1 = open_digraph([0], [1], [n0, n1])

        n2 = node(0, 'i', {}, {1: 1})
        n3 = node(1, 'o', {0: 1}, {})
        g2 = open_digraph([0], [1], [n2, n3])

        g1.icompose(g2)

        self.assertEqual(g1.get_node_by_id(3).get_parents(), {2: 1})
        self.assertEqual(g1.get_input_ids(), [2])
        self.assertEqual(g1.get_output_ids(), [1])

    def test_compose(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        g1 = open_digraph([0], [1], [n0, n1])

        n2 = node(0, 'i', {}, {1: 1})
        n3 = node(1, 'o', {0: 1}, {})
        g2 = open_digraph([0], [1], [n2, n3])

        g3 = g1.compose(g2)

        self.assertEqual(g3.get_node_by_id(3).get_parents(), {2: 1})
        self.assertEqual(g3.get_input_ids(), [2])
        self.assertEqual(g3.get_output_ids(), [1])

    def test_identity(self):
        g = open_digraph.identity(3)

        self.assertEqual(len(g.get_nodes()), 6)
        for i in range(3):
            self.assertEqual(g.get_node_by_id(i).get_children(), {i + 3: 1})
            self.assertEqual(g.get_node_by_id(i + 3).get_parents(), {i: 1})
        self.assertEqual(g.get_input_ids(), [0, 1, 2])
        self.assertEqual(g.get_output_ids(), [3, 4, 5])

    def test_connected_components(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {})
        n2 = node(2, 'c', {}, {3: 1})
        n3 = node(3, 'd', {2: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3])
        num_components, components = g.connected_components()
        self.assertEqual(num_components, 2)
        self.assertEqual(components[0], 0)
        self.assertEqual(components[1], 0)
        self.assertEqual(components[2], 1)
        self.assertEqual(components[3], 1)

    def test_extract_components(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {})
        n2 = node(2, 'c', {}, {3: 1})
        n3 = node(3, 'd', {2: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3])
        subgraphs = g.extract_components()
        self.assertEqual(len(subgraphs), 2)
        subgraph1 = subgraphs[0]
        subgraph2 = subgraphs[1]
        self.assertEqual(len(subgraph1.get_nodes()), 2)
        self.assertEqual(len(subgraph2.get_nodes()), 2)

    def test_dijkstra(self):
        n0 = node(0, 'a', {}, {1: 1, 2: 4})
        n1 = node(1, 'b', {0: 1}, {2: 2, 3: 5})
        n2 = node(2, 'c', {0: 4, 1: 2}, {3: 1})
        n3 = node(3, 'd', {1: 5, 2: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3])

        distances, _ = g.dijkstra(0)
        expected_distances = {0: 0, 1: 1, 2: 3, 3: 4}
        self.assertEqual(distances, expected_distances)

        distances_reverse, _ = g.dijkstra(3, direction=-1)
        expected_distances_reverse = {0: float('inf'), 1: 5, 2: 1, 3: 0}
        #self.assertEqual(distances_reverse, expected_distances_reverse)

    def test_dijkstra_with_target(self):
        n0 = node(0, 'a', {}, {1: 1, 2: 4})
        n1 = node(1, 'b', {0: 1}, {2: 2, 3: 5})
        n2 = node(2, 'c', {0: 4, 1: 2}, {3: 1})
        n3 = node(3, 'd', {1: 5, 2: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3])

        distances, prev = g.dijkstra(0, tgt=3)
        expected_distances = {0: 0, 1: 1, 2: 3, 3: 4}
        self.assertEqual(distances, expected_distances)

    def test_common_ancestors(self):
        n0 = node(0, 'a', {}, {1: 1, 2: 4})
        n1 = node(1, 'b', {0: 1}, {2: 2, 3: 5})
        n2 = node(2, 'c', {0: 4, 1: 2}, {3: 1})
        n3 = node(3, 'd', {1: 5, 2: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3])

        ancestors = g.common_ancestors(3, 2)
        expected_ancestors = {0: (4, 4), 1: (3, 2), 2: (1, 0), 3: (0, float('inf'))}
        #self.assertEqual(ancestors, expected_ancestors)       

    def test_topological_sort(self):
        n0 = node(0, '', {}, {3: 1})
        n1 = node(1, '', {}, {3: 1, 4: 1})
        n2 = node(2, '', {}, {4: 1})
        n3 = node(3, '', {0: 1, 1: 1}, {5: 1})
        n4 = node(4, '', {1: 1, 2: 1}, {6: 1})
        n5 = node(5, '', {3: 1}, {})
        n6 = node(6, '', {4: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3, n4, n5, n6])
        topological_order = g.topological_sort()
        expected_order = [0, 1, 2, 3, 4, 5, 6]  # C'est un exemple d'ordre topologique valide
        self.assertEqual(topological_order, expected_order)

    def test_compressed_topological_sort(self):
        n0 = node(0, '', {}, {3: 1})
        n1 = node(1, '', {}, {3: 1, 4: 1})
        n2 = node(2, '', {}, {4: 1})
        n3 = node(3, '', {0: 1, 1: 1}, {5: 1})
        n4 = node(4, '', {1: 1, 2: 1}, {6: 1})
        n5 = node(5, '', {3: 1}, {})
        n6 = node(6, '', {4: 1}, {})
        g = open_digraph([], [], [n0, n1, n2, n3, n4, n5, n6])
        compressed_order = g.compressed_topological_sort()
        expected_compressed_order = [{0, 1, 2}, {3, 4}, {5, 6}]
        self.assertEqual(compressed_order, expected_compressed_order)

    def test_is_acyclic(self):
        n0 = node(0, '', {}, {1: 1})
        n1 = node(1, '', {0: 1}, {2: 1})
        n2 = node(2, '', {1: 1}, {})  # Pas de cycle ici
        g = open_digraph([], [], [n0, n1, n2])
        self.assertTrue(g.is_acyclic())
        g = open_digraph([], [], [n0, n1, n2])
        self.assertTrue(g.is_acyclic())

    def test_depth(self):
        # Test de la profondeur d'un nœud
        self.assertEqual(self.g3.depth(0), 0)
        self.assertEqual(self.g3.depth(1), 1)
        self.assertEqual(self.g3.depth(2), 2)
        self.assertEqual(self.g3.depth(3), 2)
        self.assertEqual(self.g3.depth(4), 3)

    def test_graph_depth(self):
        # Test de la profondeur maximale du graphe entier
        self.assertEqual(self.g3.graph_depth(), 3)

    def test_longest_path(self):
        # Test du plus long chemin entre deux nœuds
        distance, path = self.g3.longest_path(0, 4)
        self.assertEqual(distance, 3)
        self.assertIn(path, [[0, 1, 2, 4], [0, 1, 3, 4]])
        print(f"Longest path from 0 to 4 is: {path} with distance {distance}")

    #test du ex 2 td 9
    def setUp(self):
        # Création des graphes nécessaires pour les tests
        n0 = node(0, 'a', {}, {1: 1, 2: 1})
        n1 = node(1, 'b', {0: 1}, {3: 1})
        n2 = node(2, 'c', {0: 1}, {3: 1})
        n3 = node(3, 'd', {1: 1, 2: 1}, {})
        self.g1 = open_digraph([0], [3], [n0, n1, n2, n3])

        m0 = node(0, 'x', {}, {1: 1})
        m1 = node(1, 'y', {0: 1}, {})
        self.g2 = open_digraph([0], [1], [m0, m1])

        # Création d'un autre graphe pour les tests de profondeur et de chemin
        n0 = node(0, '', {}, {1: 1})
        n1 = node(1, '', {0: 1}, {2: 1, 3: 1})
        n2 = node(2, '', {1: 1}, {4: 1})
        n3 = node(3, '', {1: 1}, {4: 1})
        n4 = node(4, '', {2: 1, 3: 1}, {})
        self.g3 = open_digraph([], [], [n0, n1, n2, n3, n4])

    def test_fusion(self):
        n0 = node(0, 'x', {}, {1: 1})
        n1 = node(1, 'y', {0: 1}, {2: 1})
        n2 = node(2, 'z', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])
        g.fusion(1, 2, 'w')
        self.assertIn(1, g.nodes)
        self.assertNotIn(2, g.nodes)
        self.assertEqual(g.nodes[1].label, 'w')
       # self.assertEqual(g.nodes[1].get_children(), {})
       # self.assertEqual(g.nodes[1].get_parents(), {0: 1})
    
 
#test du ex 3+4 du td 9 
    def test_parse_parentheses(self):
        formula = "(x0&((x1)&(x2)))|((x1)&(~(x2)))"
        g = bool_circ.parse_parentheses(formula)
        self.assertIsInstance(g, bool_circ)
       # self.assertTrue(g.is_bool_circ())

    def test_from_formula(self):
        formula = "(x0&((x1)&(x2)))|((x1)&(~(x2)))"
        g, vars = bool_circ.from_formula(formula)
        self.assertIsInstance(g, bool_circ)
       # self.assertTrue(g.is_bool_circ())
       # self.assertEqual(vars, ['x0', 'x1', 'x2'])


    def test_from_multiple_formulas(self):
        formulas = [
            "(x0&((x1)&(x2)))|((x1)&(~(x2)))",
            "((x0)&((x1)))|(x2)"
        ]
       # g = bool_circ.from_multiple_formulas(*formulas)
        #self.assertIsInstance(g, bool_circ)
        #self.assertTrue(g.is_bool_circ())  
 #ex 5 td 9 
    def test_parse_optimized_parentheses(self):
        formula = "((x0&((x1)&(x2)))|((x1)&(~(x2))))"
        g = bool_circ.parse_optimized_parentheses(formula)
        self.assertIsInstance(g, bool_circ)
        #self.assertTrue(g.is_bool_circ())

    def test_from_optimized_formula(self):
        formula = "((x0&((x1)&(x2)))|((x1)&(~(x2))))"
        g, vars = bool_circ.from_optimized_formula(formula)
        self.assertIsInstance(g, bool_circ)
        #self.assertTrue(g.is_bool_circ())
       # self.assertEqual(vars, ['x0', 'x1', 'x2'])
#ex 1 td 10:générons un circuit booléen aléatoire avec un nombre spécifié de nœuds d'entrée et de sortie, et nous validons la correction de l'implémentation en utilisant des tests unitaires
    def test_random_bool_circ(self):
        n = 10
        inputs_count = 3
        outputs_count = 2
        bool_circuit = bool_circ.random_bool_circ(n, inputs_count, outputs_count)
        self.assertEqual(len(bool_circuit.get_input_ids()), inputs_count)
        self.assertEqual(len(bool_circuit.get_output_ids()), outputs_count)
       # self.assertTrue(bool_circuit.is_bool_circ())

       #ex 2 td 10 : implémentation pour générer des circuits booléens aléatoires avec un nombre spécifié d'entrées et de sorties
    def test_random_bool_circ(self):
        n = 10
        inputs_count = 3
        outputs_count = 2
        bool_circuit = bool_circ.random_bool_circ(n, inputs_count, outputs_count)
        self.assertEqual(len(bool_circuit.get_input_ids()), inputs_count)
        self.assertEqual(len(bool_circuit.get_output_ids()), outputs_count)
       # self.assertTrue(bool_circuit.is_bool_circ())
  
    def test_adder_half_adder(self):
        n = 4  # Taille du registre
        adder_circuit = bool_circ.adder_n(n)
        half_adder_circuit = bool_circ.half_adder_n(n)

        print("Circuit Adder_n:")
        adder_circuit.display(verbose=True)

        print("Circuit Half_Adder_n:")
        half_adder_circuit.display(verbose=True)


    def test_carry_lookahead_adder_4(self):
        cla4 = bool_circ.carry_lookahead_adder_4()
       # self.assertTrue(cla4.is_bool_circ())
        expected_node_count = 25  # Ajustez ce nombre selon votre implémentation
        self.assertEqual(len(cla4.get_nodes()), expected_node_count)   


    def test_carry_lookahead_adder_4n(self):
        n = 4
        cla4n = bool_circ.carry_lookahead_adder_4n(n)
       # self.assertTrue(cla4n.is_bool_circ())
        expected_node_count_4n = 6.25 * n  # Ajustez ce nombre selon votre implémentation
        self.assertEqual(len(cla4n.get_nodes()), expected_node_count_4n)    

    def test_estimate_depth_and_gates(self):
        n = 4
        depth, gates = bool_circ.estimate_depth_and_gates(n)
        self.assertEqual(depth, 5)  # Ajustez si nécessaire
        self.assertEqual(gates, 100)  # Ajustez si nécessaire

    def test_is_well_formed(self):
        # Créer un circuit booléen valide avec les nouvelles primitives
        g = bool_circ.empty()
        n0 = g.add_node('0')
        n1 = g.add_node('1')
        n_and = g.add_node('&')
        n_or = g.add_node('|')
        n_not = g.add_node('~')
        n_xor = g.add_node('^')

        g.add_edge(n0, n_and)
        g.add_edge(n1, n_and)
        g.add_edge(n_and, n_or)
        g.add_edge(n_or, n_not)
        g.add_edge(n_not, n_xor)

        bc = bool_circ([n0], [n_xor], g.get_nodes())
        #self.assertTrue(bc.is_bool_circ())

        #ex 2 du td 11: crée un circuit booléen représentant un entier donné en binaire, avec une taille de registre par défaut de 8 bits. Ensuite, nous devons écrire des tests pour vérifier que cette méthode fonctionne correctement
    def test_from_integer(self):
        bc = bool_circ.from_integer(11)
        expected_labels = ['0', '0', '0', '0', '1', '0', '1', '1']
        node_labels = [node.get_label() for node in bc.get_nodes()]
        self.assertEqual(node_labels, expected_labels)

        bc = bool_circ.from_integer(255)
        expected_labels = ['1', '1', '1', '1', '1', '1', '1', '1']
        node_labels = [node.get_label() for node in bc.get_nodes()]
        self.assertEqual(node_labels, expected_labels)

        bc = bool_circ.from_integer(0)
        expected_labels = ['0', '0', '0', '0', '0', '0', '0', '0']
        node_labels = [node.get_label() for node in bc.get_nodes()]
        self.assertEqual(node_labels, expected_labels)

        #test 5 td11:
    # Test pour vérifier que l'évaluation fonctionne correctement pour un additionneur de taille 4
    def test_evaluate_on_adder(self):
        adder = bool_circ.adder_n(4)
        self.set_inputs(adder, 3, 5)  # Initialiser les entrées avec 3 et 5
        adder.evaluate()
        for node in adder.get_nodes():
            self.assertIn(node.get_label(),  "All nodes should be evaluated to '0' or '1'")

    def test_addition(self):
        adder = bool_circ.adder_n(4)
        self.set_inputs(adder, 16, 5)  # 16 + 5 = 21
        adder.evaluate()
        result = self.get_output(adder)
        self.assertEqual(result, 21, "16 + 5 should be 21")

    def set_inputs(self, adder, a, b):
        binary_a = bin(a)[2:].zfill(4)
        binary_b = bin(b)[2:].zfill(4)
        inputs = adder.get_input_ids()

        for i in range(4):
            adder.get_node_by_id(inputs[i]).set_label(binary_a[i])
        for i in range(4):
            adder.get_node_by_id(inputs[4 + i]).set_label(binary_b[i])

    def get_output(self, adder):
        outputs = adder.get_output_ids()
        output_values = [adder.get_node_by_id(output).get_label() for output in outputs]
        output_binary = ''.join(output_values)
        return int(output_binary, 2)

    

    

    
 



if __name__ == '__main__':
    unittest.main()

# Import des modules nécessaires
import sys
import os
import unittest

# Ajout de la racine du projet au chemin pour l'import des modules
root = os.path.normpath(os.path.join(__file__, './../..'))
sys.path.append(root)
from modules.open_digraph import *

# Définition de la classe de tests unitaires
class InitTest(unittest.TestCase):
    # Test du constructeur de la classe 'node'
    def test_init_node(self):
        n0 = node(0, 'i', {}, {1: 1})  # Cree un nœud
        self.assertEqual(n0.get_id(), 0)  # Vérifie l'ID
        self.assertEqual(n0.get_label(), 'i')  # Verifie le label
        self.assertEqual(n0.get_parents(), {})  # Vérifie les parents
        self.assertEqual(n0.get_children(), {1: 1})  # Verifie les enfants
        self.assertIsInstance(n0, node)  # Vérifie l'instance

    # Test du constructeur de la classe 'open_digraph'
    def test_init_open_digraph(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        G = open_digraph([0], [1], [n0, n1])  # Crée un graphe
        self.assertEqual(G.get_input_ids(), [0])  # Vérifie les entrées
        self.assertEqual(G.get_output_ids(), [1])  # Vérifie les sorties
        self.assertEqual(G.get_node_by_id(0), n0)  # Vérifie les nœuds
        self.assertEqual(G.get_node_by_id(1), n1)
        self.assertIsInstance(G, open_digraph)  # Vérifie l'instance

    # Test de la méthode de copie pour la classe 'node'
    def test_copy_node(self):
        n0 = node(0, 'i', {}, {1: 1})  # Cree un nœud
        n0_copy = n0.copy()  # Copie le nœud
        self.assertEqual(n0_copy.get_id(), n0.get_id())  # Vérifie l'ID
        self.assertEqual(n0_copy.get_label(), n0.get_label())  # Vérifie le label
        self.assertEqual(n0_copy.get_parents(), n0.get_parents())  # Vérifie les parents
        self.assertEqual(n0_copy.get_children(), n0.get_children())  # Vérifie les enfants
        self.assertIsNot(n0, n0_copy)  # Vérifie que les instances sont différentes
        self.assertIsInstance(n0_copy, node)  # Vérifie l'instance

    # Test de la méthode de copie pour la classe 'open_digraph'
    def test_copy_open_digraph(self):
        n0 = node(0, 'i', {}, {1: 1})
        n1 = node(1, 'o', {0: 1}, {})
        G = open_digraph([0], [1], [n0, n1])  # Crée un graphe
        G_copy = G.copy()  # Copie le graphe
        self.assertEqual(G_copy.get_input_ids(), G.get_input_ids())  # Vérifie les entrées
        self.assertEqual(G_copy.get_output_ids(), G.get_output_ids())  # Vérifie les sorties
        self.assertEqual(len(G_copy.get_nodes()), len(G.get_nodes()))  # Vérifie le nombre de nœuds
        for node_id in G.get_node_ids():
            self.assertEqual(G_copy.get_node_by_id(node_id).get_id(), G.get_node_by_id(node_id).get_id())  # Vérifie l'ID des nœuds
            self.assertEqual(G_copy.get_node_by_id(node_id).get_label(), G.get_node_by_id(node_id).get_label())  # Vérifie le label des nœuds
            self.assertEqual(G_copy.get_node_by_id(node_id).get_parents(), G.get_node_by_id(node_id).get_parents())  # Vérifie les parents des nœuds
            self.assertEqual(G_copy.get_node_by_id(node_id).get_children(), G.get_node_by_id(node_id).get_children())  # Vérifie les enfants des nœuds
            self.assertIsNot(G.get_node_by_id(node_id), G_copy.get_node_by_id(node_id))  # Vérifie que les instances des nœuds sont différentes
        self.assertIsNot(G, G_copy)  # Vérifie que les instances des graphes sont différentes
        self.assertIsInstance(G_copy, open_digraph)  # Vérifie l'instance

# Exécution des tests si le script est lancé directement
if __name__ == '__main__':
    unittest.main()

class WellFormedTest(unittest.TestCase):
    def setUp(self):
        # Préparation des graphes de test
        self.G = open_digraph.empty()
        self.n0 = self.G.add_node('n0')
        self.n1 = self.G.add_node('n1')
        self.G.add_edge(self.n0, self.n1)

    def test_is_well_formed(self):
        self.assertTrue(self.G.is_well_formed())

        # Manipulations pour rendre le graphe mal formé
        self.G.add_edge(self.n0, self.n1)  # Ajout d'une deuxième arête
        self.assertFalse(self.G.is_well_formed())
        self.G.remove_parallel_edges(self.n0, self.n1)  # Suppression des arêtes en double
        self.assertTrue(self.G.is_well_formed())

    def test_add_remove_node(self):
        n2 = self.G.add_node('n2')
        self.assertTrue(self.G.is_well_formed())
        self.G.remove_node_by_id(n2)
        self.assertTrue(self.G.is_well_formed())

    def test_add_remove_edge(self):
        self.G.add_edge(self.n0, self.n1)
        self.assertTrue(self.G.is_well_formed())
        self.G.remove_edge(self.n0, self.n1)
        self.assertTrue(self.G.is_well_formed())

    def test_add_input_output_node(self):
        self.G.add_input_node(self.n0)
        self.assertTrue(self.G.is_well_formed())
        self.G.add_output_node(self.n1)
        self.assertTrue(self.G.is_well_formed())

# Exécution des tests si le script est lancé directement
if __name__ == '__main__':
    unittest.main()

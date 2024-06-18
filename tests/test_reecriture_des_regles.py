import unittest
from open_digraph_test import node, open_digraph, bool_circ
#ex 2 td 12: es règles de réécriture dans le contexte des circuits booléens ou des graphes dirigés sont utilisées pour simplifier et optimiser ces circuits ou graphes
class TestReecriture(unittest.TestCase):

    def test_xor_associativity(self):
        g = bool_circ.empty()
        n1 = g.add_node(label='^')
        n2 = g.add_node(label='^')
        n3 = g.add_node(label='input')
        n4 = g.add_node(label='input')
        n5 = g.add_node(label='input')

        g.add_edge(n3, n2)
        g.add_edge(n4, n2)
        g.add_edge(n2, n1)
        g.add_edge(n5, n1)

        print(f"Graph before apply_xor_associativity: {g}")

        g.apply_xor_associativity()

        print(f"Graph after apply_xor_associativity: {g}")

        # Vérifiez que n2 a été supprimé et que les arêtes ont été réarrangées correctement
       
        self.assertIn(n5, g.get_node_by_id(n1).get_parent_ids())


        def test_copy_associativity(self):
         g = bool_circ.empty()
         n1 = g.add_node(label='copy')
         n2 = g.add_node(label='copy')
         n3 = g.add_node(label='input')
         n4 = g.add_node(label='input')
         n5 = g.add_node(label='input')

         g.add_edge(n3, n2)
         g.add_edge(n4, n2)
         g.add_edge(n2, n1)
         g.add_edge(n5, n1)

         print(f"Graph before apply_copy_associativity: {g}")

         g.apply_copy_associativity()

         print(f"Graph after apply_copy_associativity: {g}")

         self.assertNotIn(n2, g.get_node_ids())
         self.assertIn(n3, g.get_node_by_id(n1).get_parents_ids())
         self.assertIn(n4, g.get_node_by_id(n1).get_parents_ids())
         self.assertIn(n5, g.get_node_by_id(n1).get_parents_ids())
    
    def test_not_through_xor(self):
        g = bool_circ.empty()
        n1 = g.add_node(label='~')
        n2 = g.add_node(label='^')
        n3 = g.add_node(label='input')
        n4 = g.add_node(label='input')
        n5 = g.add_node(label='input')

        g.add_edge(n3, n2)
        g.add_edge(n4, n2)
        g.add_edge(n2, n1)
        g.add_edge(n5, n1)
        print(f"Graph before apply_not_through_xor: {g}")

        g.apply_not_through_xor()

        print(f"Graph after apply_not_through_xor: {g}") 
        self.assertIn(n5, g.get_node_by_id(n1).get_parent_ids())

    def test_not_through_copy(self):
        g = bool_circ.empty()
        n1 = g.add_node(label='~')
        n2 = g.add_node(label='copy')
        n3 = g.add_node(label='input')
        n4 = g.add_node(label='input')
        n5 = g.add_node(label='input')

        g.add_edge(n5, n1)
        g.add_edge(n3, n2)
        g.add_edge(n4, n2)
        g.add_edge(n2, n1)

        print(f"Graph before apply_not_through_copy: {g}")

        g.apply_not_through_copy()

        print(f"Graph after apply_not_through_copy: {g}")
        self.assertIn(n5, g.get_node_by_id(n1).get_parent_ids())

    def test_apply_all_rewriting_rules(self):
        g = bool_circ.empty()
        n1 = g.add_node(label='~')
        n2 = g.add_node(label='^')
        n3 = g.add_node(label='copy')
        n4 = g.add_node(label='input')
        n5 = g.add_node(label='input')

        g.add_edge(n5, n3)
        

        print(f"Graph before apply_rewriting_rules: {g}")

        g.apply_rewriting_rules()

        print(f"Graph after apply_rewriting_rules: {g}")

      
        self.assertIn(n5, g.get_node_by_id(n3).get_parent_ids())   


    #ex 4 td 12:Tests Unitaires pour les Règles de Réécriture
    # def test_circuit_simplification(self):
        for _ in range(100):  # Test on 100 randomly generated circuits
            g = bool_circ.random(10, 3)  # Example: 10 nodes, 3 inputs
            initial_gate_count = len(g.get_node_ids())
            g.apply_all_rewriting_rules()
            simplified_gate_count = len(g.get_node_ids())
            print(f"Initial gates: {initial_gate_count}, Simplified gates: {simplified_gate_count}")     

if __name__ == '__main__':
    unittest.main()

# open_digraph_compositions_mx.py

class open_digraph_compositions_mx:
    #ex 1 td6: implémenter les méthodes iparallel et parallel dans la classe open_digraph. Ces méthodes permettront de composer des graphes booléens en parallèle

    def iparallel(self, g):
        """
        Ajoute les noeuds et arêtes de g à self sans chevauchement d'indices.
        """
        shift_value = self.max_id() + 1
        g.shift_indices(shift_value)

        # Ajouter les noeuds de g à self
        for node in g.get_nodes():
            self.nodes[node.get_id()] = node

        # Ajouter les inputs et outputs de g à self
        self.inputs.extend(g.get_input_ids())
        self.outputs.extend(g.get_output_ids())

    def parallel(self, g):
        """
        Renvoie un nouveau graphe qui est la composition parallèle de self et g.
        """
        new_g = self.copy()
        new_g.iparallel(g)
        return new_g

    # ex 2+3 du td 6  :La méthode icompose relie les entrées de self aux sorties de f. Si les nombres d'entrées et de sorties ne coïncident pas, une exception est levée.

    def icompose(self, f):
        """
        Fait la composition séquentielle de self et f.
        """
        if len(self.inputs) != len(f.outputs):
            raise ValueError("Les nombres d'entrées de self et de sorties de f doivent coïncider.")
        
        shift_value = self.max_id() + 1
        f_shifted = f.copy()
        f_shifted.shift_indices(shift_value)

        # Relier les sorties de f_shifted aux entrées de self
        for output_id, input_id in zip(f_shifted.outputs, self.inputs):
            self.add_edge(output_id, input_id)
        
        # Ajouter les noeuds de f_shifted à self
        for node in f_shifted.get_nodes():
            self.nodes[node.get_id()] = node
        
        self.inputs = f_shifted.inputs

    def compose(self, f):
        """
        Renvoie un nouveau graphe qui est la composition séquentielle de self et f.
        """
        new_g = self.copy()
        new_g.icompose(f)
        return new_g
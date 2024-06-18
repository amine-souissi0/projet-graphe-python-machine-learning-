import unittest
import unittest
from open_digraph_test import node, open_digraph, bool_circ

class TestHammingCode(unittest.TestCase):

    def setUp(self):
        self.encoder = bool_circ.encoder_hamming_7_4()
        self.decoder = bool_circ.decoder_hamming_7_4()

    def set_inputs(self, circuit, values):
        inputs = circuit.get_input_ids()
        for i, value in enumerate(values):
            circuit.get_node_by_id(inputs[i]).set_label(str(value))

    def get_output(self, circuit):
        outputs = circuit.get_output_ids()
        return [int(circuit.get_node_by_id(output).get_label() or 0) for output in outputs]

    def test_single_error_correction(self):
        input_data = [0, 0, 0, 0]
        self.set_inputs(self.encoder, input_data)
        self.encoder.evaluate()
        encoded_data = self.get_output(self.encoder)

        # Introduire une seule erreur
        encoded_data_with_error = encoded_data.copy()
        encoded_data_with_error[0] = 1 - encoded_data_with_error[0]

        self.set_inputs(self.decoder, encoded_data_with_error)
        self.decoder.evaluate()
        decoded_data = self.get_output(self.decoder)

        self.assertEqual(decoded_data[:4], input_data, "The decoded data should match the input data with single error correction")

    def test_double_error_detection(self):
        input_data = [1, 0, 1, 1]
        self.set_inputs(self.encoder, input_data)
        self.encoder.evaluate()
        encoded_data = self.get_output(self.encoder)

        # Introduire deux erreurs
        encoded_data_with_errors = encoded_data.copy()
        encoded_data_with_errors[0] = 1 - encoded_data_with_errors[0]
        encoded_data_with_errors[1] = 1 - encoded_data_with_errors[1]

        self.set_inputs(self.decoder, encoded_data_with_errors)
        self.decoder.evaluate()
        decoded_data = self.get_output(self.decoder)

        self.assertNotEqual(decoded_data[:4], input_data, "The decoded data should not match the input data with double error")

if __name__ == '__main__':
    unittest.main()

class TestHammingCode(unittest.TestCase):

    def setUp(self):
        self.encoder = bool_circ.encoder_hamming_7_4()
        self.decoder = bool_circ.decoder_hamming_7_4()

    def set_inputs(self, circuit, values):
        inputs = circuit.get_input_ids()
        for i, value in enumerate(values):
            circuit.get_node_by_id(inputs[i]).set_label(str(value))

    def get_output(self, circuit):
        outputs = circuit.get_output_ids()
        return [int(circuit.get_node_by_id(output).get_label() or 0) for output in outputs]

    def test_single_error_correction(self):
        test_cases = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1]
        ]
        for input_data in test_cases:
            with self.subTest(input_data=input_data):
                self.set_inputs(self.encoder, input_data)
                self.encoder.evaluate()
                encoded_data = self.get_output(self.encoder)

                # Introduire une seule erreur
                for i in range(len(encoded_data)):
                    encoded_data_with_error = encoded_data.copy()
                    encoded_data_with_error[i] = 1 - encoded_data_with_error[i]

                    self.set_inputs(self.decoder, encoded_data_with_error)
                    self.decoder.evaluate()
                    decoded_data = self.get_output(self.decoder)

                    self.assertEqual(decoded_data[:4], input_data, f"The decoded data should match the input data with single error correction for input {input_data} with error at position {i}")
#ex 3 td 12 :Vérification de la Propriété Principale du Code de Hamming
    def test_double_error_detection(self):
        input_data = [1, 0, 1, 1]
        self.set_inputs(self.encoder, input_data)
        self.encoder.evaluate()
        encoded_data = self.get_output(self.encoder)

        # Introduire deux erreurs
        encoded_data_with_errors = encoded_data.copy()
        encoded_data_with_errors[0] = 1 - encoded_data_with_errors[0]
        encoded_data_with_errors[1] = 1 - encoded_data_with_errors[1]

        self.set_inputs(self.decoder, encoded_data_with_errors)
        self.decoder.evaluate()
        decoded_data = self.get_output(self.decoder)

        self.assertNotEqual(decoded_data[:4], input_data, "The decoded data should not match the input data with double error")

if __name__ == '__main__':
    unittest.main()

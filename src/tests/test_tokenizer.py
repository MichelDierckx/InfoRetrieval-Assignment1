import unittest

from src.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenize(self):
        text = "Science & Mathematics PhysicsThe hot glowing surfaces of stars emit energy in the form of electromagnetic radiation.?It is a good approximation to assume that the emissivity e is equal to 1 for these surfaces."
        expected_tokens = ['science',
                           'mathematics',
                           'physicsthe',
                           'hot',
                           'glowing',
                           'surface',
                           'star',
                           'emit',
                           'energy',
                           'form',
                           'electromagnetic',
                           'radiation',
                           'good',
                           'approximation',
                           'assume',
                           'emissivity',
                           'e',
                           'equal',
                           '1',
                           'surface']
        result = self.tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, result)


if __name__ == "__main__":
    unittest.main()

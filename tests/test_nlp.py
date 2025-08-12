import unittest
from src.nlp import parser, complexity_score
from src.nlp.simplifier import PromptSimplifier


class TestNLPComponents(unittest.TestCase):

    def test_prompt_parser(self):
        prompt = "Generate a climate report based on 2023 data."
        cleaned = parser.clean_prompt(prompt)
        tokens = parser.tokenize_prompt(cleaned)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_complexity_score(self):
        prompt = "Explain photosynthesis."
        features = complexity_score.compute_complexity_features(prompt)

        self.assertIsInstance(features, dict)
        self.assertIn("token_count", features)
        self.assertIn("readability_score", features)
        self.assertIsInstance(features["token_count"], int)
        self.assertIsInstance(features["readability_score"], float)

    def test_simplifier(self):
        simplifier = PromptSimplifier()
        complex_prompt = (
            "Could you elucidate the mechanisms by which flora engage "
            "in the biochemical process of photosynthesis?"
        )
        simplified = simplifier.simplify_prompt(complex_prompt)

        self.assertIsInstance(simplified, str)
        self.assertLessEqual(len(simplified), len(complex_prompt))


if __name__ == '__main__':
    unittest.main()
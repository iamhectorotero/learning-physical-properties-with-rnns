import unittest
import numpy as np
from isaac.statistical_tests import z_test


class TestZTest(unittest.TestCase):
    @staticmethod
    def get_answers_to_get_a_determined_accuracy(correct_answers, accuracy, n_possible_answers=3):
        answers = []
        possible_answers = set([i for i in range(n_possible_answers)])

        for true_class in correct_answers:
            if np.random.rand() < accuracy:
                answers.append(true_class)
            else:
                incorrect_answers = list(possible_answers - set([true_class]))
                answers.append(np.random.choice(incorrect_answers))

        return answers

    def test_z_statistic_for_significantly_different_model(self):
        correct_answers = np.random.randint(low=0, high=3, size=200)

        # the first model should be significantly better than a random model
        first_model_answers = np.random.randint(low=0, high=3, size=200)
        second_model_answers = self.get_answers_to_get_a_determined_accuracy(correct_answers, 0.66)

        z = z_test(correct_answers, first_model_answers, second_model_answers)
        self.assertLess(z, -1.645)

    def test_z_statistic_for_random_models(self):
        correct_answers = np.random.randint(low=0, high=3, size=100)

        # two random models shouldn't be significantly different
        first_model_answers = np.random.randint(low=0, high=3, size=100)
        second_model_answers = np.random.randint(low=0, high=3, size=100)

        z = z_test(correct_answers, first_model_answers, second_model_answers)
        self.assertGreater(z, -1.645)

    def test_z_statistic_for_equally_accurate_models(self):
        correct_answers = np.random.randint(low=0, high=3, size=100)

        # two random models shouldn't be significantly different
        first_model_answers = self.get_answers_to_get_a_determined_accuracy(correct_answers, 0.66)
        second_model_answers = self.get_answers_to_get_a_determined_accuracy(correct_answers, 0.66)

        z = z_test(correct_answers, first_model_answers, second_model_answers)
        self.assertGreater(z, -1.645)

if __name__ == "__main__":
    unittest.main()

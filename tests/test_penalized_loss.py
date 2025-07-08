import unittest

import tensorflow as tf

from araras.keras.utils.punish import (
    compute_flops_penalized_loss,
    compute_params_penalized_loss,
)
from araras.keras.utils.profiler import get_flops


class PenalizedLossTests(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(3,)),
        ])

    def test_flops_penalty_single_and_list(self):
        penalty_factor = 1e-10
        base_loss = 1.0
        expected_penalty = penalty_factor * get_flops(self.model)

        single = compute_flops_penalized_loss(
            base_loss, self.model, penalty_factor, "add"
        )
        self.assertAlmostEqual(single, base_loss + expected_penalty)

        loss_list = [1.0, 2.0]
        result_list = compute_flops_penalized_loss(
            loss_list, self.model, penalty_factor, "subtract"
        )
        self.assertIsInstance(result_list, list)
        expected_list = [l - expected_penalty for l in loss_list]
        for r, e in zip(result_list, expected_list):
            self.assertAlmostEqual(r, e)

    def test_params_penalty_single_and_list(self):
        penalty_factor = 1e-9
        base_loss = 2.0
        expected_penalty = penalty_factor * self.model.count_params()

        single = compute_params_penalized_loss(
            base_loss, self.model, penalty_factor, "add"
        )
        self.assertAlmostEqual(single, base_loss + expected_penalty)

        loss_list = [2.0, 3.0]
        result_list = compute_params_penalized_loss(
            loss_list, self.model, penalty_factor, "subtract"
        )
        self.assertIsInstance(result_list, list)
        expected_list = [l - expected_penalty for l in loss_list]
        for r, e in zip(result_list, expected_list):
            self.assertAlmostEqual(r, e)


if __name__ == "__main__":
    unittest.main()


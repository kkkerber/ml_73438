"""Independent numerical and edge-case checks for Lab 02."""

import contextlib
import io
import unittest

import numpy as np

from perceptron import (
    NeuralNetwork, centered_cross_entropy, cross_entropy_from_logits,
    gradient_check, prepare_data, softmax,
)


class NeuralNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = prepare_data()

    def test_split_scaling_and_one_hot(self):
        self.assertEqual(self.X_train.shape, (1257, 64))
        self.assertEqual(self.X_test.shape, (540, 64))
        self.assertEqual(self.y_train.shape, (1257, 10))
        for X in (self.X_train, self.X_test):
            self.assertGreaterEqual(X.min(), 0)
            self.assertLessEqual(X.max(), 1)
        for y in (self.y_train, self.y_test):
            np.testing.assert_array_equal(y.sum(axis=1), 1)
            self.assertTrue(np.isin(y, [0, 1]).all())
        full_counts = self.y_train.sum(axis=0) + self.y_test.sum(axis=0)
        np.testing.assert_allclose(self.y_test.sum(axis=0) / full_counts,
                                   0.3, rtol=0, atol=0.006)
        for first, second in zip(
            (self.X_train, self.X_test, self.y_train, self.y_test), prepare_data()
        ):
            np.testing.assert_array_equal(first, second)

    def test_forward_shapes_for_both_widths_and_partial_batches(self):
        for width in (16, 64):
            model = NeuralNetwork(width)
            for size in (1, 9, 32):
                with self.subTest(width=width, batch_size=size):
                    probs, cache = model.forward(self.X_train[:size])
                    self.assertEqual(probs.shape, (size, 10))
                    np.testing.assert_allclose(probs.sum(axis=1), 1,
                                               rtol=0, atol=1e-6)
                    self.assertTrue((probs >= 0).all())
                    gradients = model.backward(cache, self.y_train[:size])
                    for name, param in model.params.items():
                        self.assertEqual(gradients[name].shape, param.shape)

    def test_softmax_and_loss_with_extreme_logits(self):
        logits = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
        labels = np.array([[1., 0., 0.], [1., 0., 0.]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            probabilities = softmax(logits)
            loss = cross_entropy_from_logits(logits, labels)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-15)
        self.assertAlmostEqual(loss, 1000.)
        self.assertAlmostEqual(cross_entropy_from_logits(np.zeros((2, 3)), labels),
                               np.log(3))
        self.assertAlmostEqual(cross_entropy_from_logits(logits + 12345., labels), loss)

    def test_random_weight_checks_and_parameter_restoration(self):
        for width in (16, 64):
            model = NeuralNetwork(width)
            before = {name: value.copy() for name, value in model.params.items()}
            with contextlib.redirect_stdout(io.StringIO()):
                rows = gradient_check(model, self.X_train[:32], self.y_train[:32])
            self.assertEqual(len(rows), 10)
            self.assertGreaterEqual(sum(r["passed"] for r in rows), 9)
            for name, value in before.items():
                np.testing.assert_array_equal(model.params[name], value)

    def test_directional_derivative_includes_all_biases_and_weights(self):
        # Move activations away from ReLU's nondifferentiable point.
        # A single independent finite difference perturbs every parameter,
        # including biases, and checks the derivative in that joint direction.
        rng = np.random.default_rng(123)
        model = NeuralNetwork(16)
        model.params["b1"][:] = 0.3
        model.params["b2"][:] = 0.3
        X, y = self.X_train[:9], self.y_train[:9]
        _, cache = model.forward(X)
        gradients = model.backward(cache, y)
        directions = {name: rng.normal(size=p.shape) for name, p in model.params.items()}
        analytic = sum(np.sum(gradients[name] * direction)
                       for name, direction in directions.items())
        original = {name: p.copy() for name, p in model.params.items()}
        epsilon = 1e-6
        losses = []
        for sign in (1, -1):
            for name in model.params:
                model.params[name][:] = original[name] + sign * epsilon * directions[name]
            _, perturbed = model.forward(X)
            losses.append(cross_entropy_from_logits(perturbed["z3"], y))
        numerical = (losses[0] - losses[1]) / (2 * epsilon)
        error = abs(analytic - numerical) / (abs(analytic) + abs(numerical))
        self.assertLess(error, 1e-6)

    def test_sgd_lowers_loss_and_predict_matches_forward(self):
        model = NeuralNetwork(16)
        X, y = self.X_train[:32], self.y_train[:32]
        probabilities, cache = model.forward(X)
        np.testing.assert_array_equal(model.predict(X), probabilities.argmax(axis=1))
        before = cross_entropy_from_logits(cache["z3"], y)
        self.assertAlmostEqual(centered_cross_entropy(cache["z3"], y) + np.log(10), before)
        model.step(model.backward(cache, y), learning_rate=0.1)
        _, after = model.forward(X)
        self.assertLess(cross_entropy_from_logits(after["z3"], y), before)


if __name__ == "__main__":
    unittest.main()

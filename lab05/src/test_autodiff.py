import math
import unittest

import numpy as np
import torch

from autodiff import Value
from model import ScalarMLP, TorchMLP, initial_arrays


class AutodiffTests(unittest.TestCase):
    def test_shared_operand_and_branch(self):
        x = Value(3)
        u = x * x
        out = u * u + u + x
        out.backward()
        self.assertEqual(x.grad, 115)  # 4*x**3 + 2*x + 1

    def test_repeated_backward_accumulates_leaves(self):
        x = Value(3)
        out = (x * x + 1) ** 2
        out.backward()
        self.assertEqual(x.grad, 120)
        out.backward()
        self.assertEqual(x.grad, 240)

    def test_reverse_arithmetic(self):
        x = Value(2)
        out = (3 - x) / x + 2 * x
        out.backward()
        self.assertAlmostEqual(out.data, 4.5)
        self.assertAlmostEqual(x.grad, 1.25)

    def test_relu(self):
        for value, derivative in [(-2, 0), (0, 0), (2, 1)]:
            x = Value(value)
            x.relu().backward()
            self.assertEqual(x.grad, derivative)

    def test_power_zero(self):
        x = Value(0)
        y = x ** 0
        y.backward()
        self.assertEqual(y.data, 1)
        self.assertEqual(x.grad, 0)

    def test_deep_graph(self):
        x = Value(1)
        out = x
        for _ in range(3000):
            out = out + 1
        out.backward()
        self.assertEqual(x.grad, 1)

    def test_topological_dependencies(self):
        x = Value(2)
        u = x * x
        out = (u + x) * (u - 1)
        order = out.topological()
        positions = {node: i for i, node in enumerate(order)}
        self.assertEqual(len(order), len(positions))
        for node in order:
            for parent in node.parents:
                self.assertLess(positions[parent], positions[node])

    def test_exp_log_sigmoid(self):
        x = Value(0.7)
        out = x.exp() + x.log() + x.sigmoid()
        out.backward()
        sigmoid = 1 / (1 + math.exp(-0.7))
        self.assertAlmostEqual(x.grad, math.exp(0.7) + 1/0.7 + sigmoid*(1-sigmoid))

    def test_stable_bce_extremes(self):
        for z in [-1000, -3, 0, 3, 1000]:
            for label in [0, 1]:
                x = Value(z)
                loss = x.softplus() - label * x
                loss.backward()
                t = torch.tensor(float(z), dtype=torch.float64, requires_grad=True)
                ref = torch.nn.functional.binary_cross_entropy_with_logits(
                    t, torch.tensor(float(label), dtype=torch.float64))
                ref.backward()
                self.assertAlmostEqual(loss.data, ref.item(), places=12)
                self.assertAlmostEqual(x.grad, t.grad.item(), places=12)

    def test_requested_expression_and_numeric(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            data = rng.uniform(-2, 2, 3)
            a, b, c = [Value(x) for x in data]
            loss = ((a*b + c)**2).relu().sum()
            loss.backward()
            ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in data]
            ref = ((ts[0]*ts[1] + ts[2])**2).relu().sum()
            ref.backward()
            for i, (node, t) in enumerate(zip([a, b, c], ts)):
                delta = np.zeros(3)
                delta[i] = 1e-5
                def f(v):
                    return max((v[0]*v[1]+v[2])**2, 0)
                numeric = (f(data+delta) - f(data-delta)) / 2e-5
                self.assertLess(abs(node.grad - t.grad.item()), 1e-6)
                self.assertLess(abs(node.grad - numeric), 1e-6)

    def test_model_loss_all_parameter_gradients_and_step(self):
        arrays = initial_arrays()
        scalar, reference = ScalarMLP(arrays), TorchMLP(arrays)
        x = np.array([[0.2, -0.7], [-1.3, 0.4], [1.2, 0.8]])
        y = np.array([0., 1., 1.])
        loss = scalar.loss(x, y)
        loss.backward()
        target = torch.nn.functional.binary_cross_entropy_with_logits(
            reference(torch.from_numpy(x)), torch.from_numpy(y))
        target.backward()
        self.assertAlmostEqual(loss.data, target.item(), places=12)
        for name, p in reference.parameter_map().items():
            np.testing.assert_allclose(scalar.arrays(True)[name], p.grad.numpy(), rtol=0, atol=1e-12)
        for p in scalar.parameters():
            p.data -= 0.1 * p.grad
        with torch.no_grad():
            for p in reference.parameters():
                p -= 0.1 * p.grad
        for name, p in reference.parameter_map().items():
            np.testing.assert_allclose(scalar.arrays()[name], p.detach().numpy(), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

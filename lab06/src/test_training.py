"""Numerical checks for accumulation and exact checkpoint recovery on small data."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset
from train import (CNN, checkpoint, load_config, make_loader, make_optimizer,
                   make_scheduler, seed_all, train_epoch, REPO)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_all(42)
        self.data = TensorDataset(torch.randn(69, 1, 28, 28), torch.randint(0, 10, (69,)))
        self.c = load_config(REPO / 'lab06/config.yaml')

    def test_accumulation_including_short_final_window(self):
        first = CNN()
        second = copy.deepcopy(first)
        a = torch.optim.SGD(first.parameters(), lr=.01, momentum=.9)
        b = torch.optim.SGD(second.parameters(), lr=.01, momentum=.9)
        train_epoch(first, make_loader(self.data, 32, 0), a)
        train_epoch(second, make_loader(self.data, 8, 0), b, accumulation=4)
        for x, y in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)

    def test_recovery_restores_optimizer_scheduler_and_shuffle(self):
        g = torch.Generator().manual_seed(42)
        model = CNN()
        optimizer = make_optimizer(model, self.c['training'])
        scheduler = make_scheduler(optimizer, self.c['training'])
        loader = make_loader(self.data, 16, 0, g, True)
        train_epoch(model, loader, optimizer)
        scheduler.step()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'last.pt'
            checkpoint(path, model, optimizer, scheduler, g, 1, [], self.c, 'test')
            expected_random = (random.random(), np.random.rand(), torch.rand(1))
            train_epoch(model, loader, optimizer)
            scheduler.step()
            expected = copy.deepcopy(model.state_dict())
            state = torch.load(path, weights_only=False)
        restored = CNN()
        restored.load_state_dict(state['model'])
        opt = make_optimizer(restored, self.c['training'])
        opt.load_state_dict(state['optimizer'])
        sched = make_scheduler(opt, self.c['training'])
        sched.load_state_dict(state['scheduler'])
        g.set_state(state['generator'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        self.assertTrue(torch.equal(torch.rand(1), expected_random[2]))
        train_epoch(restored, make_loader(self.data, 16, 0, g, True), opt)
        sched.step()
        for name, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertEqual(scheduler.state_dict(), sched.state_dict())


if __name__ == '__main__':
    unittest.main()

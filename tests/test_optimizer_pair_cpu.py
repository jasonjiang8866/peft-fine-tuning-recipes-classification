"""Optional real PyTorch CPU checks, with synthetic tensors and no HF packages."""

import copy
import importlib.util
from types import SimpleNamespace
import unittest

from benchmark_muon import make_muon_and_adamw


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU PyTorch not installed")
class OptimizerPairCPUTests(unittest.TestCase):
    def test_native_muon_adamw_linear_schedule_and_state_roundtrip(self):
        import torch
        from torch.optim.lr_scheduler import LambdaLR

        if not hasattr(torch.optim, "Muon"):
            self.skipTest("this PyTorch has no native Muon")
        args = SimpleNamespace(learning_rate=2e-4, weight_decay=0.01,
                               adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)

        def new_pair():
            lora = torch.nn.Parameter(torch.ones(2, 2))
            head = torch.nn.Parameter(torch.ones(2, 2))
            return make_muon_and_adamw([lora], [head], args, 0.02), lora, head

        # HF's default linear scheduler uses LambdaLR with this warmup/decay
        # multiplier. This tests the real PyTorch contract, not Trainer itself.
        def linear(step):
            return float(step) / 2 if step < 2 else max(0.0, (6 - step) / 4)

        pair, lora, head = new_pair()
        self.assertIsInstance(pair, torch.optim.Optimizer)
        for group, child in zip(pair.param_groups, (pair.muon, pair.adamw)):
            self.assertIs(group, child.param_groups[0])
        scheduler = LambdaLR(pair, linear)
        initial = [lora.detach().clone(), head.detach().clone()]
        for step in range(3):
            self.assertEqual([group["lr"] for group in pair.param_groups],
                             [0.02 * linear(step), 2e-4 * linear(step)])
            pair.zero_grad()
            (lora.square().sum() + head.square().sum()).backward()
            pair.step()
            scheduler.step()
        for before, after in zip(initial, (lora, head)):
            self.assertFalse(torch.equal(before, after))
        self.assertTrue(pair.muon.state)
        self.assertTrue(pair.adamw.state)

        restored, restored_lora, restored_head = new_pair()
        restored_scheduler = LambdaLR(restored, linear)
        with torch.no_grad():
            restored_lora.copy_(lora)
            restored_head.copy_(head)
        restored.load_state_dict(copy.deepcopy(pair.state_dict()))
        restored_scheduler.load_state_dict(scheduler.state_dict())
        for group, child in zip(restored.param_groups, (restored.muon, restored.adamw)):
            self.assertIs(group, child.param_groups[0])
        for optimizer, sched, a, b in ((pair, scheduler, lora, head),
                                      (restored, restored_scheduler, restored_lora, restored_head)):
            optimizer.zero_grad(set_to_none=True)
            self.assertIsNone(a.grad)
            self.assertIsNone(b.grad)
            (a.square().sum() + b.square().sum()).backward()
            optimizer.step()
            sched.step()
        torch.testing.assert_close(restored_lora, lora)
        torch.testing.assert_close(restored_head, head)
        self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())


if __name__ == "__main__":
    unittest.main()

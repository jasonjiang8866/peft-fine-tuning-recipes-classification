"""CPU regressions for the equal-time endpoint and its evidence replay."""
import copy
import unittest
from runpod.qwen.profiles import EQUAL_TIME, profile
from runpod.qwen.step_budget import step_budget_result, verify_timed_arm


def trace(durations):
    return [{"step": step, "ms": ms} for step, ms in enumerate(durations, 1)]


class EqualTimeTests(unittest.TestCase):
    def test_first_crossing_and_step_ceiling(self):
        for ms, cap, expected in (([450, 550], 3, "reached"), ([450, 600], 3, "reached"),
                                  ([450, 550], 2, "reached"), ([450, 450], 2, "step_limit")):
            result = step_budget_result(trace(ms), len(ms), cap, 1)
            self.assertEqual(result["outcome"], expected)
            self.assertEqual(result["elapsed_seconds"], sum(ms) / 1000)
        for ms in ([400], [1000, 20], [100, float("nan")], [True, 10], [0, 500]):
            with self.assertRaises(ValueError):
                step_budget_result(trace(ms), len(ms), 2, 1)
        for rows in (trace([500, 500])[::-1], trace([500, 500])[:1]):
            with self.assertRaises(ValueError):
                step_budget_result(rows, 2, 2, 1)

    def test_controller_replays_timer_and_lr_schedule(self):
        config = profile("performance", EQUAL_TIME)
        row = dict(mode="muon", global_steps=3, step_time_ms=trace([60000]*3),
                   timing={"mean_ms": 60000}, train_wall_seconds=185,
                   step_budget={"elapsed_seconds": 180, "outcome": "reached", "target_seconds": 180},
                   lr_trace=[{"step": i, "lr": [base * i / 20 for base in (.002, .0002)]}
                             for i in (1, 2, 3)])
        self.assertTrue(verify_timed_arm(row, config))
        for field, value in (("step_budget", {}), ("lr_trace", row["lr_trace"][:-1]),
                             ("train_wall_seconds", 179), ("timing", {"mean_ms": 30000})):
            with self.assertRaises(ValueError):
                verify_timed_arm(dict(row, **{field: value}), config)
        wrong_schedule = copy.deepcopy(row)
        wrong_schedule["lr_trace"][-1]["lr"] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "schedule"):
            verify_timed_arm(wrong_schedule, config)

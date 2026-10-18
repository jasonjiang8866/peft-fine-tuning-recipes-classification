"""Dependency-free regressions; run with python -S -m unittest discover -s tests -v."""

import contextlib
import importlib.abc
import io
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import benchmark_muon as benchmark


ROOT = Path(__file__).resolve().parents[1]


class Parameter:
    def __init__(self, shape, requires_grad=True):
        self.shape, self.requires_grad = shape, requires_grad
        self.ndim = len(shape)

    def numel(self):
        return math.prod(self.shape)


def model_with(*parameters):
    return SimpleNamespace(named_parameters=lambda: iter(parameters))


def optimizer_args():
    return SimpleNamespace(learning_rate=2e-4, weight_decay=0.01,
                           adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)


def make_run(mode):
    return benchmark.BenchmarkRun(mode, Path("unused"), 20, 8, 0.02, 5)


class ParameterRoutingTests(unittest.TestCase):
    def test_recipe_has_128_lora_matrices_and_adamw_score_head(self):
        parameters = []
        for layer in range(16):
            for projection, width in (("q_proj", 2048), ("k_proj", 512),
                                      ("v_proj", 512), ("o_proj", 2048)):
                prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
                parameters.extend([
                    (prefix + ".lora_A.default.weight", Parameter((16, 2048))),
                    (prefix + ".lora_B.default.weight", Parameter((width, 16))),
                    (prefix + ".base_layer.weight", Parameter((width, 2048), False)),
                ])
        head = Parameter((4, 2048))
        parameters.append(("base_model.model.score.modules_to_save.default.weight", head))
        rows, muon, adamw = benchmark.trainable_parameters(model_with(*parameters))
        self.assertEqual((len(rows), len(muon)), (129, 128))
        self.assertEqual(adamw, [head])
        self.assertEqual(sum(row["numel"] for row in rows), 3416064)
        self.assertEqual(rows[-1], {
            "name": parameters[-1][0], "shape": [4, 2048], "ndim": 2,
            "numel": 8192, "optimizer_in_muon_run": "adamw",
        })
        self.assertTrue(all(row["optimizer_in_muon_run"] == "muon" for row in rows[:-1]))

    def test_only_exact_lora_segments_route_to_muon(self):
        a, b = Parameter((2, 4)), Parameter((4, 2))
        others = [(name, Parameter(shape)) for name, shape in [
            ("model.score.modules_to_save.default.weight", (4, 4)),
            ("model.score.bias", (4,)), ("model.norm.weight", (4,)),
            ("model.not_lora_A.default.weight", (4, 4)),
            ("model.lora_B_extra.default.weight", (4, 4)),
        ]]
        rows, muon, adamw = benchmark.trainable_parameters(model_with(
            ("model.q_proj.lora_A.default.weight", a),
            ("model.q_proj.lora_B.adapter2.weight", b), *others))
        self.assertEqual(muon, [a, b])
        self.assertEqual(adamw, [param for _, param in others])
        self.assertEqual([row["name"] for row in rows[2:]], [name for name, _ in others])

    def test_frozen_parameters_are_excluded_before_dimension_validation(self):
        live = Parameter((2, 4))
        rows, muon, adamw = benchmark.trainable_parameters(model_with(
            ("model.q.lora_A.default.weight", live),
            ("model.q.lora_B.frozen.weight", Parameter((4,), False)),
            ("model.score.original_module.weight", Parameter((4, 4), False)),
            ("model.base.weight", Parameter((4, 4), False))))
        self.assertEqual((len(rows), muon, adamw), (1, [live], []))

    def test_non_matrix_lora_is_rejected_with_parameter_name(self):
        for kind in ("lora_A", "lora_B"):
            for shape in ((), (4,), (2, 3, 4)):
                with self.subTest(kind=kind, shape=shape):
                    name = f"model.q.{kind}.default.weight"
                    with self.assertRaisesRegex(ValueError, "2D LoRA parameter: " + name):
                        benchmark.trainable_parameters(model_with((name, Parameter(shape))))

    def test_empty_or_frozen_only_lora_is_rejected(self):
        for parameters in ([], [("model.score.weight", Parameter((4, 4)))],
                           [("model.q.lora_A.default.weight", Parameter((2, 4), False))]):
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, "No trainable LoRA"):
                    benchmark.trainable_parameters(model_with(*parameters))


class OptimizerContractTests(unittest.TestCase):
    """Small doubles test delegation only, not PyTorch or Trainer execution."""

    def setUp(self):
        class Optimizer:
            def __init__(self, groups, defaults):
                self.param_groups = groups

        def child(params, **defaults):
            optimizer = Mock()
            optimizer.param_groups = [{"params": params, **defaults}]
            return optimizer

        self.muon = Mock(side_effect=child)
        self.adamw = Mock(side_effect=child)
        self.torch = SimpleNamespace(optim=SimpleNamespace(
            Optimizer=Optimizer, Muon=self.muon, AdamW=self.adamw))
        self.muon_params, self.adamw_params = [Parameter((2, 4))], [Parameter((4, 4))]

    def pair(self):
        with patch.dict(sys.modules, {"torch": self.torch}):
            return benchmark.make_muon_and_adamw(
                self.muon_params, self.adamw_params, optimizer_args(), 0.02)

    def test_missing_muon_and_missing_head_fail_before_constructing_optimizers(self):
        del self.torch.optim.Muon
        with self.assertRaisesRegex(RuntimeError, "Muon is unavailable"):
            self.pair()
        self.torch.optim.Muon = self.muon
        self.adamw_params = []
        with self.assertRaisesRegex(ValueError, "classification head"):
            self.pair()
        self.muon.assert_not_called()
        self.adamw.assert_not_called()

    def test_pair_delegates_to_disjoint_optimizers_and_shares_scheduler_groups(self):
        pair = self.pair()
        self.muon.assert_called_once_with(self.muon_params, lr=0.02, weight_decay=0.01)
        self.adamw.assert_called_once_with(self.adamw_params, lr=2e-4, weight_decay=0.01,
                                          betas=(0.9, 0.999), eps=1e-8)
        for group, child in zip(pair.param_groups, (pair.muon, pair.adamw)):
            self.assertIs(group, child.param_groups[0])
            group["lr"] *= 0.5
            self.assertEqual(child.param_groups[0]["lr"], group["lr"])
        closure = Mock()
        self.assertIs(pair.step(closure), pair.muon.step.return_value)
        pair.muon.step.assert_called_once_with(closure=closure)
        pair.adamw.step.assert_called_once_with()
        pair.zero_grad(set_to_none=False)
        pair.muon.zero_grad.assert_called_once_with(set_to_none=False)
        pair.adamw.zero_grad.assert_called_once_with(set_to_none=False)

    def test_load_reconnects_groups_if_children_replace_their_dictionaries(self):
        pair = self.pair()
        saved = pair.state_dict()
        self.assertEqual(saved, {"muon": pair.muon.state_dict.return_value,
                                 "adamw": pair.adamw.state_dict.return_value})
        for child in (pair.muon, pair.adamw):
            def replace_groups(state, child=child):
                child.param_groups = [{**child.param_groups[0], "lr": 0.001}]
            child.load_state_dict.side_effect = replace_groups
        pair.load_state_dict(saved)
        for key, child, group in zip(("muon", "adamw"), (pair.muon, pair.adamw),
                                     pair.param_groups):
            child.load_state_dict.assert_called_once_with(saved[key])
            self.assertIs(group, child.param_groups[0])


class EntrypointTests(unittest.TestCase):
    def test_cli_help_needs_no_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "benchmark_muon.py"),
                                 "--help"], capture_output=True, text=True, check=True)
        self.assertIn("--muon-lr", result.stdout)

    def test_adamw_arm_omits_optimizer_override(self):
        run = make_run("adamw")
        modules = {"torch": SimpleNamespace(),
                   "transformers": SimpleNamespace(TrainerCallback=object)}
        model = model_with(("model.q.lora_A.default.weight", Parameter((2, 4))),
                           ("model.score.weight", Parameter((4, 4))))
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(benchmark, "make_muon_and_adamw") as make_pair:
            kwargs = run.trainer_kwargs(model, optimizer_args())
        self.assertEqual(set(kwargs), {"callbacks"})
        make_pair.assert_not_called()

    def test_muon_arm_passes_pair_and_leaves_scheduler_creation_to_trainer(self):
        run = make_run("muon")
        args = optimizer_args()
        a, head = Parameter((2, 4)), Parameter((4, 4))
        modules = {"torch": SimpleNamespace(),
                   "transformers": SimpleNamespace(TrainerCallback=object)}
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(benchmark, "make_muon_and_adamw") as make_pair:
            kwargs = run.trainer_kwargs(model_with(
                ("model.q.lora_A.default.weight", a), ("model.score.weight", head)), args)
        make_pair.assert_called_once_with([a], [head], args, 0.02)
        self.assertEqual(kwargs["optimizers"], (make_pair.return_value, None))

    def test_default_training_does_not_import_benchmark_or_override_optimizer(self):
        # Execute the entire entry point with inert external modules. Any benchmark
        # import is blocked; no data/model/network/GPU work is possible here.
        class BlockBenchmark(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "benchmark_muon":
                    raise AssertionError("default training imported the benchmark")

        dataset = Mock()
        dataset.features = {"label": SimpleNamespace(num_classes=4)}
        dataset.train_test_split.return_value = {"train": dataset, "test": dataset}
        dataset.map.return_value = dataset
        model = SimpleNamespace(config=SimpleNamespace())
        tokenizer = SimpleNamespace(eos_token="eos", eos_token_id=0)
        trainer_init = Mock()
        train, evaluate = Mock(), Mock(return_value={"eval_loss": 1.0})

        class Trainer:
            def __init__(self, **kwargs):
                trainer_init(**kwargs)

        Trainer.train = train
        Trainer.evaluate = evaluate

        def module(name, **attributes):
            result = ModuleType(name)
            result.__dict__.update(attributes)
            return result

        modules = {
            "numpy": module("numpy"),
            "torch": module("torch", float16="float16"),
            "torch.nn": module("torch.nn"),
            "torch.nn.functional": module("torch.nn.functional"),
            "datasets": module("datasets", load_dataset=Mock(return_value={
                "train": dataset, "test": dataset})),
            "transformers": module("transformers", Trainer=Trainer,
                TrainingArguments=lambda **kwargs: SimpleNamespace(**kwargs),
                BitsAndBytesConfig=lambda **kwargs: kwargs,
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                AutoModelForSequenceClassification=SimpleNamespace(
                    from_pretrained=Mock(return_value=model))),
            "peft": module("peft", LoraConfig=lambda **kwargs: kwargs,
                prepare_model_for_kbit_training=lambda model: model,
                get_peft_model=lambda model, config: model),
            "sklearn": module("sklearn"),
            "sklearn.metrics": module("sklearn.metrics", accuracy_score=Mock(),
                                      precision_recall_fscore_support=Mock()),
        }
        for value in (None, ""):
            with self.subTest(environment=value), patch.dict(sys.modules, modules), \
                    patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                os.environ.pop(benchmark.PREFIX, None)
                if value is not None:
                    os.environ[benchmark.PREFIX] = value
                # Force import attempts through the blocker, even if imported above.
                sys.modules.pop("benchmark_muon", None)
                blocker = BlockBenchmark()
                sys.meta_path.insert(0, blocker)
                try:
                    state = runpy.run_path(str(ROOT / "training.py"), run_name="__main__")
                finally:
                    sys.meta_path.remove(blocker)
                self.assertIsNone(state["benchmark"])
                kwargs = trainer_init.call_args.kwargs
                self.assertNotIn("optimizers", kwargs)
                self.assertNotIn("callbacks", kwargs)
                args = kwargs["args"]
                self.assertEqual((args.learning_rate, args.per_device_train_batch_size,
                                  args.logging_steps, args.output_dir),
                                 (2e-4, 64, 50, "ag_news_lora"))
                self.assertEqual(args.save_strategy, "epoch")
                self.assertTrue(args.load_best_model_at_end)
                train.assert_called()
                evaluate.assert_called_with(dataset)


if __name__ == "__main__":
    unittest.main()

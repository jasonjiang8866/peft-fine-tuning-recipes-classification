"""Real Trainer/Accelerate/PEFT resume test; tiny random Llama, CPU, no downloads."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "accelerate", "peft"))


@unittest.skipUnless(AVAILABLE, "optional Trainer integration dependencies not installed")
class TrainerPairCPUTests(unittest.TestCase):
    def test_real_trainer_peft_checkpoint_resume_both_arms(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (LlamaConfig, LlamaForSequenceClassification,
                                  Trainer, TrainerCallback, TrainingArguments, set_seed)
        from benchmark_muon import make_muon_and_adamw, trainable_parameters
        from runpod.validate import compare_resume

        if not hasattr(torch.optim, "Muon"):
            self.skipTest("native Muon required")
        torch.set_num_threads(1)
        dataset = [{"input_ids": [1, 3 + i % 8, 2, 2], "attention_mask": [1, 1, 1, 1],
                    "labels": i % 4} for i in range(16)]
        c = {"checkpoint_step": 3, "resume_atol": 1e-6, "resume_rtol": 1e-5}

        def train(directory, mode, resume=None):
            set_seed(42)
            model = LlamaForSequenceClassification(LlamaConfig(
                vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                num_labels=4, pad_token_id=0, attention_dropout=0.0))
            model = get_peft_model(model, LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.1, bias="none", task_type="SEQ_CLS",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"]))
            rows, lora, head = trainable_parameters(model)
            self.assertEqual((len(lora), len(head)), (16, 1))
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            args = TrainingArguments(
                output_dir=str(directory / "checkpoints"), use_cpu=True,
                max_steps=6, per_device_train_batch_size=2, gradient_accumulation_steps=2,
                learning_rate=2e-4, warmup_steps=1, logging_steps=1,
                save_strategy="steps" if not resume else "no", save_steps=3,
                report_to="none", disable_tqdm=True, seed=42, data_seed=42)
            kwargs = {}
            if mode == "muon":
                kwargs["optimizers"] = (make_muon_and_adamw(lora, head, args, 0.02), None)
            lr_trace = []
            outer = self
            class Audit(TrainerCallback):
                def on_train_begin(self, args, state, control, optimizer=None, **kw):
                    outer.assertEqual(state.global_step, 3 if resume else 0)
                    outer.assertEqual(type(optimizer).__name__, "AcceleratedOptimizer")
                    if mode == "muon":
                        pair = optimizer.optimizer
                        for group, child in zip(pair.param_groups, (pair.muon, pair.adamw)):
                            outer.assertIs(group, child.param_groups[0])
                            if resume:
                                outer.assertTrue(child.state)
                def on_step_end(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
                    lrs = [float(g["lr"]) for g in optimizer.param_groups]
                    outer.assertEqual(lrs, list(lr_scheduler.get_last_lr()))
                    lr_trace.append({"step": state.global_step, "lr": lrs})
            trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=[Audit()], **kwargs)
            result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            self.assertEqual(trainer.state.global_step, 6)
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            after = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(any(not torch.equal(before[n], after[n]) for n in after if ".lora_B." in n))
            self.assertTrue(any(not torch.equal(before[n], after[n]) for n in after if ".score." in n))
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(after, directory / "final-trainables.pt")
            torch.save({"optimizer": trainer.optimizer.state_dict(),
                        "scheduler": trainer.lr_scheduler.state_dict()}, directory / "final-state.pt")
            from runpod.validate import write_json
            write_json(directory / "result.json", {"lr_trace": lr_trace})

        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("adamw", "muon"):
                with self.subTest(mode=mode):
                    original, resumed = Path(tmp) / mode, Path(tmp) / (mode + "-resumed")
                    train(original, mode)
                    train(resumed, mode, original / "checkpoints/checkpoint-3")
                    result = compare_resume(original, resumed, c)
                    self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()

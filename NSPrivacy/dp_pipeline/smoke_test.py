"""Synthetic functional test for the NSPrivacy private trainer."""

import math

import tempfile

import torch

from nsprivacy_dp import TrainConfig, save_run, train_private


def main():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(32, 4, generator=generator)
    y = torch.randint(0, 3, (32,), generator=generator)
    config = TrainConfig(
        input_dim=4,
        num_classes=3,
        target_epsilon=8.0,
        target_delta=1e-5,
        epochs=1,
        batch_size=8,
        max_grad_norm=1.0,
        erase_sigma=0.01,
        warm_epochs=1,
        device="cpu",
    )
    model, audit = train_private(x, y, config)
    assert math.isfinite(audit["achieved_epsilon"])
    assert audit["achieved_epsilon"] <= config.target_epsilon + 1e-6
    assert audit["completed_steps"] > 0
    assert len(audit["epoch_records"]) == audit["completed_epochs"]
    assert "epsilon_budget" in audit["epoch_records"][0]
    with tempfile.TemporaryDirectory() as output_dir:
        save_run(output_dir, model, config, audit)
        record = __import__("json").loads(
            (__import__("pathlib").Path(output_dir) / "run_record.json").read_text()
        )
        assert "private_seed" not in record["config"]
        assert "class_weights" not in record["config"]
    print("Smoke test passed.")
    print(audit)


if __name__ == "__main__":
    main()

"""Train a tiny model whose objective is reported only through W&B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import wandb


def main(argv: list[str] | None = None) -> int:
    """Train and evaluate once using the supplied parameters and W&B identity."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--learning_rate", type=float, required=True, help="SGD learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="SGD weight decay")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch training device"
    )
    parser.add_argument("--receipt", type=Path, required=True, help="Parameter-only receipt path")
    parser.add_argument(
        "--device-receipt", type=Path, required=True, help="Executed-device receipt path"
    )
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("error: CUDA was requested but PyTorch cannot access it", file=sys.stderr)
        return 1
    device = torch.device(args.device)
    parameters = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
    print(
        f"{args.device.upper()} budget: one weight, zero initialization, "
        f"64 training examples, "
        f"20 steps, batch=64, 32 held-out examples; parameters={parameters}",
        flush=True,
    )
    args.receipt.write_text(json.dumps(parameters) + "\n", encoding="utf-8")
    torch.set_num_threads(1)
    train_x = torch.linspace(-1.0, 1.0, 64, device=device).reshape(-1, 1)
    eval_x = torch.linspace(-0.95, 0.95, 32, device=device).reshape(-1, 1)
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # W&B consumes the target, immutable run ID, and never-resume policy from
    # the supplied environment. Do not override them in the trainer.
    with wandb.init(
        config=parameters, dir=str(args.receipt.parent), settings=wandb.Settings(silent=True)
    ) as run:
        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(train_x), 2 * train_x)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            measured_loss = torch.nn.functional.mse_loss(model(eval_x), 2 * eval_x).item()
        run.log({"eval/loss": measured_loss})
        run.summary["eval/loss"] = measured_loss
    device_index = torch.cuda.current_device() if device.type == "cuda" else None
    device_receipt = {
        "requested_device": args.device,
        "model_device": str(model.weight.device),
        "train_tensor_device": str(train_x.device),
        "eval_tensor_device": str(eval_x.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": (
            torch.cuda.get_device_name(device_index) if device_index is not None else None
        ),
    }
    args.device_receipt.write_text(json.dumps(device_receipt) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

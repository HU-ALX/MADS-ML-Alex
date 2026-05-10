from __future__ import annotations

import argparse
import csv
import platform
import random
import time
import warnings
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics


warnings.simplefilter("ignore", UserWarning)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class NeuralNetwork(nn.Module):
    """Simple feed-forward neural network for Fashion-MNIST.

    Fashion-MNIST images are 28 x 28 grayscale images.
    The image is flattened into 784 input values.

    Architecture:
        784 -> units1 -> ReLU -> units2 -> ReLU -> 10
    """

    def __init__(self, num_classes: int, units1: int, units2: int) -> None:
        super().__init__()

        # These public attributes are useful because mltrainer/tomlserializer
        # can save them into model.toml.
        self.num_classes = num_classes
        self.units1 = units1
        self.units2 = units2

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    run_name: str
    units1: int
    units2: int
    parameter_count: int
    batchsize: int
    epochs: int
    train_steps: int
    valid_steps: int
    learning_rate: float
    optimizer: str
    scheduler: str
    device: str
    platform: str
    python_version: str
    torch_version: str
    final_valid_accuracy: float
    run_seconds: float
    logdir: str


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility.

    Exact numerical results may still differ slightly across Windows, Linux,
    macOS, CUDA, MPS, and CPU backends.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More reproducible, sometimes slightly slower.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def select_device(requested_device: str) -> torch.device:
    """Select CUDA, Apple Silicon MPS, or CPU.

    requested_device options:
        auto: choose CUDA if available, else MPS if available, else CPU
        cuda: force NVIDIA CUDA GPU
        mps: force Apple Silicon GPU
        cpu: force CPU
    """

    requested_device = requested_device.lower()

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if requested_device != "auto":
        raise ValueError("device must be one of: auto, cuda, mps, cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: Iterable,
    steps: int,
    device: torch.device,
) -> float:
    """Calculate accuracy manually on the validation stream.

    This is added because TensorBoard may only show Loss/train, Loss/test,
    and Learning rate when using mltrainer. This function gives us a reliable
    final validation accuracy for the CSV and final report.
    """

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        predictions = logits.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)

    if total == 0:
        raise RuntimeError("Validation accuracy could not be calculated because total == 0.")

    return correct / total


def write_results_csv(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(result) for result in results]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def append_result_csv(result: ExperimentResult, output_path: Path) -> None:
    """Append after each run so partial results are saved if training stops."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_run_name(units1: int, units2: int, batchsize: int, epochs: int, lr: float) -> str:
    lr_text = str(lr).replace(".", "p").replace("-", "minus")
    return f"u1_{units1}_u2_{units2}_bs_{batchsize}_ep_{epochs}_lr_{lr_text}"


# -----------------------------------------------------------------------------
# One experiment run
# -----------------------------------------------------------------------------


def run_single_experiment(
    *,
    train,
    valid,
    units1: int,
    units2: int,
    batchsize: int,
    epochs: int,
    train_steps: int,
    valid_steps: int,
    learning_rate: float,
    device: torch.device,
    log_root: Path,
    seed: int,
) -> ExperimentResult:
    set_seed(seed)

    run_name = make_run_name(
        units1=units1,
        units2=units2,
        batchsize=batchsize,
        epochs=epochs,
        lr=learning_rate,
    )
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Hidden layers: 784 -> {units1} -> {units2} -> 10")
    print("=" * 90)

    model = NeuralNetwork(num_classes=10, units1=units1, units2=units2)
    parameter_count = count_parameters(model)

    accuracy = metrics.Accuracy()
    loss_fn = torch.nn.CrossEntropyLoss()

    settings = TrainerSettings(
        epochs=epochs,
        metrics=[accuracy],
        logdir=run_dir,
        train_steps=train_steps,
        valid_steps=valid_steps,
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
        optimizer_kwargs={"lr": learning_rate},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train.stream(),
        validdataloader=valid.stream(),
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device=str(device),
    )

    start_time = time.perf_counter()
    trainer.loop()
    run_seconds = time.perf_counter() - start_time

    final_valid_accuracy = evaluate_accuracy(
        model=model,
        dataloader=valid.stream(),
        steps=valid_steps,
        device=device,
    )

    # Manually add final validation accuracy to TensorBoard.
    # This makes it easier to find in the Scalars tab.
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_scalar("Accuracy/valid_final", final_valid_accuracy, epochs)
    writer.add_scalar("Model/parameter_count", parameter_count, 0)
    writer.close()

    print(f"Parameters: {parameter_count:,}")
    print(f"Final validation accuracy: {final_valid_accuracy:.4f}")
    print(f"Run time: {run_seconds:.1f} seconds")
    print()

    return ExperimentResult(
        run_name=run_name,
        units1=units1,
        units2=units2,
        parameter_count=parameter_count,
        batchsize=batchsize,
        epochs=epochs,
        train_steps=train_steps,
        valid_steps=valid_steps,
        learning_rate=learning_rate,
        optimizer="Adam",
        scheduler="ReduceLROnPlateau",
        device=str(device),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        final_valid_accuracy=final_valid_accuracy,
        run_seconds=run_seconds,
        logdir=str(run_dir),
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden-layer size experiments on Fashion-MNIST."
    )

    parser.add_argument(
        "--units",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
        help="Hidden-layer sizes to test for both units1 and units2.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="Batch size for train and validation datastreamers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs per model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Train batches per epoch. Default: full train datastreamer length.",
    )
    parser.add_argument(
        "--valid-steps",
        type=int,
        default=None,
        help="Validation batches per epoch. Default: full validation datastreamer length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device backend. Use auto for CUDA -> MPS -> CPU.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="modellogs",
        help="Directory for TensorBoard and TOML logs.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="reports/hidden_size_results.csv",
        help="CSV file where final validation accuracies are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_root = Path(args.logdir)
    results_csv = Path(args.results_csv)
    log_root.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)

    print("Loading Fashion-MNIST through mads_datasets...")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Selected device: {device}")

    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()

    streamers = fashionfactory.create_datastreamer(
        batchsize=args.batchsize,
        preprocessor=preprocessor,
    )

    train = streamers["train"]
    valid = streamers["valid"]

    train_steps = args.train_steps if args.train_steps is not None else len(train)
    valid_steps = args.valid_steps if args.valid_steps is not None else len(valid)

    print(f"Batch size: {args.batchsize}")
    print(f"Train batches available: {len(train)}")
    print(f"Valid batches available: {len(valid)}")
    print(f"Train steps used per epoch: {train_steps}")
    print(f"Valid steps used per epoch: {valid_steps}")
    print(f"Units grid: {args.units}")
    print(f"Total runs: {len(args.units) * len(args.units)}")
    print()

    results: list[ExperimentResult] = []

    for run_index, (units1, units2) in enumerate(product(args.units, args.units), start=1):
        result = run_single_experiment(
            train=train,
            valid=valid,
            units1=units1,
            units2=units2,
            batchsize=args.batchsize,
            epochs=args.epochs,
            train_steps=train_steps,
            valid_steps=valid_steps,
            learning_rate=args.learning_rate,
            device=device,
            log_root=log_root,
            seed=args.seed + run_index,
        )
        results.append(result)
        append_result_csv(result, results_csv)

    write_results_csv(results, results_csv)

    df = pd.DataFrame([asdict(result) for result in results])
    best = df.sort_values("final_valid_accuracy", ascending=False).iloc[0]

    print("Experiment complete.")
    print(f"Saved results to: {results_csv}")
    print(f"Saved TensorBoard/TOML logs to: {log_root}")
    print()
    print("Best run:")
    print(
        f"units1={int(best['units1'])}, "
        f"units2={int(best['units2'])}, "
        f"accuracy={best['final_valid_accuracy']:.4f}, "
        f"parameters={int(best['parameter_count']):,}"
    )
    print()
    print("To inspect TensorBoard, run:")
    print(f"tensorboard --logdir={log_root}")


if __name__ == "__main__":
    main()

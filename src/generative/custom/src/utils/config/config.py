from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Google Drive project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/NMA-DL2026/NMA_BrainMRI_Project"
)

DATA_ROOT = PROJECT_ROOT / "data"

DATASET_ROOT = (
    DATA_ROOT
    / "processed"
    / "generation"
)

DATASET_IDS_ROOT = (
    DATA_ROOT
    / "ids"
    / "raw"
)

MODELS_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "custom"
    / "runs"
)


def get_next_attempt(models_root: Path) -> int:
    """
    Return the next numeric experiment ID.

    Examples
    --------
    No existing run directories:
        next run = 1

    Existing directories:
        01, 02, 03
        next run = 4
    """

    models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_attempts = [
        int(path.name)
        for path in models_root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
    ]

    return max(
        existing_attempts,
        default=0,
    ) + 1


@dataclass
class ExperimentConfig:
    # Experiment tracking
    use_comet_ml: bool = False

    # Reproducibility
    seed: int = 0

    # DataLoader settings
    train_batch_size: int = 2
    val_batch_size: int = 2
    test_batch_size: int = 64
    gen_batch_size: int = 24
    num_workers: int = 4

    # Generation preview
    images_to_generate: int = 8

    # Training
    epochs: int = 200
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    end_loss_threshold: Optional[float] = 0.0002

    # Dataset paths
    dataset_root_path: str = str(
        DATASET_ROOT
    )

    dataset_ids_path: str = str(
        DATASET_IDS_ROOT
    )

    training_ids: str = "train.tsv"
    validation_ids: str = "validation.tsv"
    test_ids: str = "test.tsv"

    # Diffusion
    num_train_timesteps: int = 2000

    # Validation and evaluation frequency
    model_eval_epochs: int = 50
    model_val_epochs: int = 5

    # Results
    results_dir_root: str = str(
        MODELS_ROOT
    )

    run_id: Optional[int] = None
    results_dir: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        results_root = Path(
            self.results_dir_root
        )

        results_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.run_id is None:
            self.run_id = get_next_attempt(
                results_root
            )

        if self.run_id < 1:
            raise ValueError(
                "run_id must be greater than or equal to 1."
            )

        self.results_dir = str(
            results_root
            / f"{self.run_id:02d}"
        )
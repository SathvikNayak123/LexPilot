"""Training configuration schema with Pydantic validation and YAML loading.

Defines all hyperparameters for QLoRA fine-tuning of Phi-3-mini on the
Indian financial document QA task. Every field carries a sensible default
matching the project spec so that ``TrainingConfig()`` works out-of-the-box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import structlog
import yaml
from pydantic import BaseModel, Field, model_validator

logger = structlog.get_logger(__name__)


class TrainingConfig(BaseModel):
    """Complete set of hyperparameters for QLoRA fine-tuning.

    All fields have defaults that match the project specification.
    Override individual values via ``TrainingConfig(lr=1e-5, ...)`` or
    load from a YAML file with ``TrainingConfig.from_yaml(path)``.
    """

    # ── Model ────────────────────────────────────────────────────────────
    base_model: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="HuggingFace model ID for the base model to fine-tune.",
    )
    output_dir: str = Field(
        default="./outputs/phi3-findocs-qlora",
        description="Directory where checkpoints and final adapter are saved.",
    )

    # ── Quantisation (BitsAndBytes) ──────────────────────────────────────
    load_in_4bit: bool = Field(default=True, description="Enable 4-bit quantisation.")
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(
        default="nf4",
        description="Quantisation data type: NormalFloat4 or FP4.",
    )
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16",
        description="Compute dtype during forward pass inside quantised layers.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Apply nested quantisation to further reduce memory.",
    )

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_r: int = Field(default=16, description="LoRA rank (r).")
    lora_alpha: int = Field(default=32, description="LoRA scaling factor (alpha).")
    lora_target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Linear layers targeted by LoRA adapters.",
    )
    lora_dropout: float = Field(
        default=0.05,
        description="Dropout probability applied to LoRA layers.",
    )

    # ── Training ─────────────────────────────────────────────────────────
    num_train_epochs: int = Field(default=3, description="Total training epochs.")
    per_device_train_batch_size: int = Field(
        default=4,
        description="Micro-batch size per device during training.",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Number of forward passes before a weight update.",
    )
    learning_rate: float = Field(default=2e-4, description="Peak learning rate.")
    lr_scheduler_type: str = Field(
        default="cosine",
        description="Learning-rate scheduler (e.g. cosine, linear).",
    )
    warmup_ratio: float = Field(
        default=0.03,
        description="Fraction of total steps used for linear warmup.",
    )
    weight_decay: float = Field(default=0.01, description="Weight decay coefficient.")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping norm.")
    max_seq_length: int = Field(
        default=2048,
        description="Maximum sequence length (tokens) for input + output.",
    )
    fp16: bool = Field(default=False, description="Use FP16 mixed-precision training.")
    bf16: bool = Field(default=True, description="Use BF16 mixed-precision training.")
    gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for memory via gradient checkpointing.",
    )

    # ── Logging / Saving ─────────────────────────────────────────────────
    logging_steps: int = Field(default=10, description="Log metrics every N steps.")
    save_steps: int = Field(default=100, description="Save checkpoint every N steps.")
    save_total_limit: int = Field(
        default=3,
        description="Keep at most this many checkpoints on disk.",
    )
    evaluation_strategy: str = Field(
        default="steps",
        description="When to run evaluation: 'steps', 'epoch', or 'no'.",
    )
    eval_steps: int = Field(default=100, description="Run eval every N steps.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    # ── MLflow ───────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI.",
    )
    mlflow_experiment_name: str = Field(
        default="findocs-phi3-qlora",
        description="MLflow experiment name.",
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_target_size: int = Field(
        default=3000,
        description="Target number of QA pairs in the training set.",
    )
    eval_dataset_size: int = Field(
        default=200,
        description="Number of QA pairs reserved for evaluation.",
    )
    train_val_split_ratio: float = Field(
        default=0.9,
        description="Fraction of dataset used for training (rest is validation).",
    )
    dedup_similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity threshold above which QA pairs are considered duplicates.",
    )

    # ── HuggingFace Hub ──────────────────────────────────────────────────
    hub_repo_id: str = Field(
        default="your-username/findocs-phi3-finetuned",
        description="HuggingFace Hub repository ID for pushing the adapter.",
    )

    @model_validator(mode="after")
    def _validate_lora_alpha_ge_rank(self) -> "TrainingConfig":
        """Ensure lora_alpha >= lora_r, the standard effective-rank guideline."""
        if self.lora_alpha < self.lora_r:
            logger.warning(
                "training_config.lora_alpha_less_than_rank",
                lora_alpha=self.lora_alpha,
                lora_r=self.lora_r,
                hint="Typically lora_alpha >= lora_r for stable training.",
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load a ``TrainingConfig`` from a YAML file.

        Keys in the YAML file must match the field names of this model.
        Unknown keys are silently ignored so that the same YAML can carry
        comments or future fields without breaking older code.

        Args:
            path: Filesystem path to a ``.yaml`` / ``.yml`` file.

        Returns:
            A validated ``TrainingConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file is not valid YAML.
            pydantic.ValidationError: If values fail schema validation.
        """
        resolved = Path(path).resolve()
        logger.info("training_config.loading_yaml", path=str(resolved))

        with open(resolved, encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        config = cls(**raw)
        logger.info(
            "training_config.loaded",
            base_model=config.base_model,
            lora_r=config.lora_r,
            epochs=config.num_train_epochs,
            lr=config.learning_rate,
        )
        return config

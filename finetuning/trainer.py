"""QLoRA fine-tuning of Phi-3-mini on the FinDocs Indian financial QA dataset.

Loads the base model in 4-bit NF4 quantisation, attaches LoRA adapters to
the attention projection layers, and trains using ``SFTTrainer`` from the
``trl`` library.  Training metrics (loss, ROUGE scores) are logged to MLflow.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import structlog
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from findocs.finetuning.config_schema import TrainingConfig

logger = structlog.get_logger(__name__)

# ARCHITECTURE DECISION: Why QLoRA over full fine-tune
# Full fine-tuning of even a 3.8B model like Phi-3-mini requires ~30GB VRAM and risks
# catastrophic forgetting of general capabilities. QLoRA (4-bit NF4 quantization +
# Low-Rank Adapters) reduces memory to ~6GB while training only 0.1% of parameters.
# The adapter-only approach preserves the base model's language understanding while
# specializing it on Indian financial domain QA. Merging adapters at inference time
# adds negligible latency.


class TrainingResult(BaseModel):
    """Structured output returned after a completed training run."""

    checkpoint_path: str = Field(..., description="Path to the final saved adapter checkpoint.")
    train_loss: float = Field(..., description="Final training loss at the end of the last epoch.")
    val_loss: float = Field(..., description="Validation loss computed after the last epoch.")
    rouge_scores: dict[str, float] = Field(
        default_factory=dict,
        description="ROUGE-1, ROUGE-2, and ROUGE-L F1 scores on the validation set.",
    )
    total_steps: int = Field(..., description="Total optimiser steps executed.")
    training_duration_seconds: float = Field(..., description="Wall-clock training time in seconds.")


class QLoRATrainer:
    """Manages the full QLoRA fine-tuning lifecycle.

    Responsibilities:
    * Load Phi-3-mini with 4-bit NF4 quantisation via BitsAndBytes.
    * Attach LoRA adapters to ``q_proj / k_proj / v_proj / o_proj``.
    * Format prompts in Phi-3 chat template.
    * Run ``SFTTrainer`` with cosine LR schedule and gradient checkpointing.
    * Log hyperparameters, loss curves, and ROUGE scores to MLflow.

    Args:
        config: A validated ``TrainingConfig`` instance.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        logger.info(
            "qlora_trainer.init",
            base_model=config.base_model,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            epochs=config.num_train_epochs,
            lr=config.learning_rate,
        )

    # ------------------------------------------------------------------
    # Model & tokenizer setup
    # ------------------------------------------------------------------

    def setup_model_and_tokenizer(self) -> None:
        """Load Phi-3-mini in 4-bit NF4 and attach LoRA adapters.

        Steps:
        1. Build a ``BitsAndBytesConfig`` for 4-bit NF4 quantisation with
           double quantisation and bfloat16 compute dtype.
        2. Load the base model with ``device_map="auto"`` so layers are
           distributed across available GPUs / CPU.
        3. Prepare the quantised model for k-bit training (freeze base,
           cast layer norms to float32).
        4. Apply ``LoraConfig`` targeting the four attention projections.
        5. Load the tokenizer and set the pad token to EOS.

        After this method returns, ``self.model`` and ``self.tokenizer``
        are ready for ``SFTTrainer``.
        """
        cfg = self.config
        logger.info("qlora_trainer.loading_model", model=cfg.base_model)

        # ── Quantisation config ──────────────────────────────────────────
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

        # ── Base model ───────────────────────────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        self.model = prepare_model_for_kbit_training(self.model)

        # ── LoRA adapters ────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

        trainable, total = 0, 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        logger.info(
            "qlora_trainer.model_loaded",
            total_params=total,
            trainable_params=trainable,
            trainable_pct=round(trainable / total * 100, 4) if total else 0,
        )

        # ── Tokenizer ───────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("qlora_trainer.tokenizer_loaded", vocab_size=self.tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_prompt(sample: dict[str, str]) -> str:
        """Format a single Alpaca-style sample into the Phi-3 chat template.

        The Phi-3 instruct template uses special tokens to delimit turns::

            <|user|>
            {instruction}
            <|end|>
            <|assistant|>
            {output}
            <|end|>

        Args:
            sample: A dict with at least ``instruction`` and ``output`` keys.
                    An optional ``input`` key is prepended to the instruction
                    when non-empty.

        Returns:
            A formatted prompt string ready for tokenisation.
        """
        instruction = sample.get("instruction", "")
        context = sample.get("input", "")
        output = sample.get("output", "")

        if context and context.strip():
            user_msg = f"{instruction}\n\nContext:\n{context}"
        else:
            user_msg = instruction

        return (
            f"<|user|>\n{user_msg}\n<|end|>\n"
            f"<|assistant|>\n{output}\n<|end|>"
        )

    # ------------------------------------------------------------------
    # ROUGE evaluation
    # ------------------------------------------------------------------

    def _compute_rouge_on_validation(self, val_path: str | Path) -> dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 on the validation set.

        Generates predictions by running the model in eval mode over each
        validation sample and comparing to the ground-truth answer.

        Args:
            val_path: Path to the validation JSONL file.

        Returns:
            Dict mapping metric names to average F1 scores.
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("qlora_trainer.rouge_skipped_no_model")
            return {}

        val_ds = load_dataset("json", data_files=str(val_path), split="train")
        if len(val_ds) == 0:
            return {}

        # Sample at most 50 examples to keep eval fast
        eval_samples = val_ds.select(range(min(50, len(val_ds))))

        self.model.eval()
        all_scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

        for sample in eval_samples:
            instruction = sample.get("instruction", "")  # type: ignore[union-attr]
            context = sample.get("input", "")  # type: ignore[union-attr]
            reference = sample.get("output", "")  # type: ignore[union-attr]

            if context and str(context).strip():
                user_msg = f"{instruction}\n\nContext:\n{context}"
            else:
                user_msg = instruction

            prompt = f"<|user|>\n{user_msg}\n<|end|>\n<|assistant|>\n"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            scores = self._rouge.score(str(reference), generated_text)
            for key in all_scores:
                all_scores[key].append(scores[key].fmeasure)

        avg_scores: dict[str, float] = {}
        for key, values in all_scores.items():
            avg_scores[key] = round(sum(values) / len(values), 4) if values else 0.0

        logger.info("qlora_trainer.rouge_scores", **avg_scores)
        return avg_scores

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_path: str | Path,
        val_path: str | Path,
    ) -> TrainingResult:
        """Run the full QLoRA training loop.

        Steps:
        1. Load train / val JSONL files into HuggingFace ``Dataset`` objects.
        2. Build ``TrainingArguments`` from the config.
        3. Initialise ``SFTTrainer`` with the Phi-3 prompt formatter.
        4. Log hyperparameters and final metrics to MLflow.
        5. Compute ROUGE scores on the validation set.
        6. Save the adapter to ``config.output_dir``.

        Args:
            train_path: Path to ``train.jsonl`` in Alpaca format.
            val_path: Path to ``val.jsonl`` in Alpaca format.

        Returns:
            A ``TrainingResult`` containing the checkpoint path, losses,
            ROUGE scores, step count, and wall-clock duration.

        Raises:
            RuntimeError: If ``setup_model_and_tokenizer`` has not been called.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer are not initialised. "
                "Call setup_model_and_tokenizer() before train()."
            )

        cfg = self.config
        train_path = Path(train_path)
        val_path = Path(val_path)

        logger.info(
            "qlora_trainer.train_start",
            train_path=str(train_path),
            val_path=str(val_path),
            epochs=cfg.num_train_epochs,
        )

        # ── Datasets ────────────────────────────────────────────────────
        train_ds = load_dataset("json", data_files=str(train_path), split="train")
        val_ds = load_dataset("json", data_files=str(val_path), split="train")

        logger.info(
            "qlora_trainer.datasets_loaded",
            train_size=len(train_ds),
            val_size=len(val_ds),
        )

        # ── Training arguments ──────────────────────────────────────────
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            evaluation_strategy=cfg.evaluation_strategy,
            eval_steps=cfg.eval_steps,
            seed=cfg.seed,
            report_to="none",  # We handle MLflow manually for finer control
            remove_unused_columns=False,
        )

        # ── SFTTrainer ──────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            formatting_func=self.format_prompt,
            max_seq_length=cfg.max_seq_length,
        )

        # ── MLflow tracking ─────────────────────────────────────────────
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)

        start_time = time.monotonic()

        with mlflow.start_run(run_name=f"qlora-phi3-{int(time.time())}") as run:
            # Log hyperparameters
            mlflow.log_params(
                {
                    "base_model": cfg.base_model,
                    "lora_r": cfg.lora_r,
                    "lora_alpha": cfg.lora_alpha,
                    "lora_dropout": cfg.lora_dropout,
                    "lora_target_modules": ",".join(cfg.lora_target_modules),
                    "num_train_epochs": cfg.num_train_epochs,
                    "per_device_train_batch_size": cfg.per_device_train_batch_size,
                    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                    "learning_rate": cfg.learning_rate,
                    "lr_scheduler_type": cfg.lr_scheduler_type,
                    "warmup_ratio": cfg.warmup_ratio,
                    "max_seq_length": cfg.max_seq_length,
                    "bf16": cfg.bf16,
                    "gradient_checkpointing": cfg.gradient_checkpointing,
                    "train_size": len(train_ds),
                    "val_size": len(val_ds),
                }
            )

            # Train
            train_result = trainer.train()
            train_loss = train_result.training_loss
            total_steps = train_result.global_step

            # Evaluate
            eval_result = trainer.evaluate()
            val_loss = eval_result.get("eval_loss", 0.0)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "total_steps": float(total_steps),
                },
                step=total_steps,
            )

            # ROUGE
            rouge_scores = self._compute_rouge_on_validation(val_path)
            if rouge_scores:
                mlflow.log_metrics(
                    {f"rouge_{k}": v for k, v in rouge_scores.items()},
                    step=total_steps,
                )

            # Save adapter
            final_adapter_path = output_dir / "final_adapter"
            self.model.save_pretrained(str(final_adapter_path))
            self.tokenizer.save_pretrained(str(final_adapter_path))
            logger.info("qlora_trainer.adapter_saved", path=str(final_adapter_path))

            mlflow.log_param("mlflow_run_id", run.info.run_id)

        elapsed = time.monotonic() - start_time

        result = TrainingResult(
            checkpoint_path=str(final_adapter_path),
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            rouge_scores=rouge_scores,
            total_steps=total_steps,
            training_duration_seconds=round(elapsed, 2),
        )

        logger.info(
            "qlora_trainer.train_complete",
            checkpoint=result.checkpoint_path,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            rouge_scores=result.rouge_scores,
            duration_s=result.training_duration_seconds,
        )

        return result

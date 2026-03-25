"""Push fine-tuned LoRA adapter to HuggingFace Hub.

Handles adapter upload, model card generation, and versioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from huggingface_hub import HfApi, ModelCard, ModelCardData
from peft import PeftModel  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

from findocs.config.config import get_settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger()
settings = get_settings()


class HubPusher:
    """Pushes a fine-tuned LoRA adapter to HuggingFace Hub."""

    def __init__(
        self,
        adapter_path: str | None = None,
        repo_id: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        """Initialise with adapter path and Hub coordinates.

        Args:
            adapter_path: Local path to the saved LoRA adapter directory.
            repo_id: HuggingFace Hub repo id (e.g. "user/findocs-phi3-finetuned").
            hf_token: HuggingFace API token.
        """
        self.adapter_path = Path(adapter_path or settings.FINETUNED_MODEL_PATH)
        self.repo_id = repo_id or settings.HF_MODEL_REPO
        self.hf_token = hf_token or settings.HF_TOKEN
        self.api = HfApi(token=self.hf_token)

    def push_adapter(self, commit_message: str = "Update LoRA adapter") -> str:
        """Push the LoRA adapter files to the Hub.

        Args:
            commit_message: Git commit message for the Hub upload.

        Returns:
            The URL of the uploaded model on the Hub.
        """
        logger.info(
            "pushing_adapter_to_hub",
            repo_id=self.repo_id,
            adapter_path=str(self.adapter_path),
        )

        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            exist_ok=True,
            token=self.hf_token,
        )

        self.api.upload_folder(
            folder_path=str(self.adapter_path),
            repo_id=self.repo_id,
            commit_message=commit_message,
            token=self.hf_token,
        )

        url = f"https://huggingface.co/{self.repo_id}"
        logger.info("adapter_pushed_successfully", url=url)
        return url

    def push_merged_model(self, commit_message: str = "Upload merged model") -> str:
        """Merge the adapter into the base model and push the full model.

        This creates a standalone model that does not require PEFT at inference.

        Args:
            commit_message: Git commit message for the Hub upload.

        Returns:
            The URL of the uploaded model on the Hub.
        """
        logger.info("merging_adapter_with_base_model", base=settings.BASE_MODEL_FOR_FINETUNING)

        base_model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL_FOR_FINETUNING,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        merged_model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(
            settings.BASE_MODEL_FOR_FINETUNING,
            trust_remote_code=True,
        )

        merged_repo_id = f"{self.repo_id}-merged"
        self.api.create_repo(
            repo_id=merged_repo_id,
            repo_type="model",
            exist_ok=True,
            token=self.hf_token,
        )

        merged_path = self.adapter_path.parent / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)

        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))

        self.api.upload_folder(
            folder_path=str(merged_path),
            repo_id=merged_repo_id,
            commit_message=commit_message,
            token=self.hf_token,
        )

        url = f"https://huggingface.co/{merged_repo_id}"
        logger.info("merged_model_pushed_successfully", url=url)
        return url

    def generate_model_card(self, metrics: dict[str, Any] | None = None) -> str:
        """Generate and upload a model card for the adapter.

        Args:
            metrics: Optional dict of evaluation metrics to include.

        Returns:
            The generated model card content as a string.
        """
        card_data = ModelCardData(
            language="en",
            license="apache-2.0",
            library_name="peft",
            tags=["finance", "india", "qlora", "phi-3", "findocs"],
            base_model=settings.BASE_MODEL_FOR_FINETUNING,
            model_name="FinDocs Phi-3 Financial QA",
        )

        metrics_section = ""
        if metrics:
            metrics_section = "\n## Evaluation Results\n\n| Metric | Score |\n|--------|-------|\n"
            for name, value in metrics.items():
                metrics_section += f"| {name} | {value:.4f} |\n"

        card_content = f"""---
{card_data.to_yaml()}
---

# FinDocs Phi-3 Financial QA (QLoRA Adapter)

Fine-tuned [Phi-3-mini-4k-instruct](https://huggingface.co/{settings.BASE_MODEL_FOR_FINETUNING})
on synthetic Indian financial document QA pairs using QLoRA.

## Intended Use

Answer questions about Indian financial documents including RBI circulars,
SEBI mutual fund factsheets, and NSE annual reports. Designed for the FinDocs
RAG pipeline serving Indian retail investors.

## Training Details

- **Method**: QLoRA (4-bit NF4 quantisation, double quantisation)
- **LoRA rank**: 16, alpha: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj
- **Epochs**: 3
- **Learning rate**: 2e-4 (cosine schedule)
- **Batch size**: 4 (gradient accumulation: 4)
{metrics_section}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{settings.BASE_MODEL_FOR_FINETUNING}")
model = PeftModel.from_pretrained(base, "{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{settings.BASE_MODEL_FOR_FINETUNING}")
```
"""

        card = ModelCard(card_content)
        card.push_to_hub(self.repo_id, token=self.hf_token)

        logger.info("model_card_pushed", repo_id=self.repo_id)
        return card_content

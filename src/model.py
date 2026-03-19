import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import CanineModel

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

@dataclass
class ModelConfig:
    base_model: str = "google/canine-c"
    vocab_size: int = 8000
    head_hidden_size: int = 512
    dropout: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = (
        "query",
        "key",
        "value",
        "attention.output.dense",
    )


class CharacterVocab:
    """
    Character-level vocabulary with explicit PAD/UNK ids.
    """

    def __init__(self, char_to_id: Dict[str, int]) -> None:
        self.char_to_id = char_to_id
        self.id_to_char = {idx: ch for ch, idx in char_to_id.items()}
        if PAD_TOKEN not in self.char_to_id or UNK_TOKEN not in self.char_to_id:
            raise ValueError("Vocabulary must contain <PAD> and <UNK> tokens.")
        if self.char_to_id[PAD_TOKEN] != PAD_ID or self.char_to_id[UNK_TOKEN] != UNK_ID:
            raise ValueError("<PAD>/<UNK> must map to ids 0/1 respectively.")

    def __len__(self) -> int:
        return len(self.char_to_id)

    def get_id(self, char: str) -> int:
        return self.char_to_id.get(char, UNK_ID)

    def get_char(self, idx: int) -> str:
        return self.id_to_char.get(idx, "")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharacterVocab":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        char_to_id = {str(k): int(v) for k, v in payload["char_to_id"].items()}
        return cls(char_to_id=char_to_id)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_size: int = 8000,
    ) -> "CharacterVocab":
        if max_size < 2:
            raise ValueError("max_size must be at least 2 to include PAD/UNK.")

        counter = Counter()
        for text in texts:
            counter.update(text)

        char_to_id: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for char, freq in counter.most_common(max_size - 2):
            if freq < min_freq:
                continue
            if char in char_to_id:
                continue
            char_to_id[char] = len(char_to_id)
        return cls(char_to_id=char_to_id)


def text_to_codepoints(text: str, max_len: Optional[int] = None) -> List[int]:
    """
    Convert raw text to Unicode codepoint ids for CANINE input.
    """
    if max_len is not None:
        text = text[-max_len:]
    return [ord(c) for c in text]


class CanineLoRACharPredictor(nn.Module):
    """
    CANINE encoder + LoRA adapters + next-character classification head.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        adapter_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        # initialize model
        self.config = config or ModelConfig()
        encoder = CanineModel.from_pretrained(self.config.base_model)
        if adapter_dir is None:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
                bias="none",
            )
            encoder = get_peft_model(encoder, lora_cfg)
        else:
            encoder = PeftModel.from_pretrained(encoder, adapter_dir)
        self.encoder = encoder

        # classification head
        hidden_size = int(self.encoder.config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, self.config.head_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden_size, self.config.vocab_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, lengths]
        return self.head(last_hidden)

    @torch.no_grad()
    def predict_topk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int = 3,
        forbidden_ids: Sequence[int] = (PAD_ID, UNK_ID),
    ) -> torch.Tensor:

        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = logits.clone()
        vocab_size = logits.size(-1)
        for idx in forbidden_ids:
            if 0 <= idx < vocab_size:
                logits[:, idx] = float("-inf")

        safe_k = min(k, vocab_size)
        return torch.topk(logits, k=safe_k, dim=-1).indices

    def count_parameters(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def save(self, work_dir: str) -> None:
        os.makedirs(work_dir, exist_ok=True)

        with open(os.path.join(work_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        # save classification head
        torch.save(self.head.state_dict(), os.path.join(work_dir, "char_head.pt"))
        # save lora adapters
        self.encoder.save_pretrained(os.path.join(work_dir, "canine_lora_adapters"))

    @classmethod
    def load(cls, work_dir: str, map_location: str = "cpu") -> "CanineLoRACharPredictor":
        with open(os.path.join(work_dir, "model_config.json"), "r", encoding="utf-8") as f:
            config_payload = json.load(f)
        config = ModelConfig(**config_payload)

        # load lora adapters
        adapter_dir = os.path.join(work_dir, "canine_lora_adapters")
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_dir}")
        model = cls(config=config, adapter_dir=adapter_dir)

        # load classification head
        head_path = os.path.join(work_dir, "char_head.pt")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Missing character head checkpoint: {head_path}")
        model.head.load_state_dict(torch.load(head_path, map_location=map_location))

        return model

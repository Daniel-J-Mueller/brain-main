import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Iterable

class BrocasArea:
    """Simple wrapper around GPT-2 for text generation.

    This module represents a language motor region. It expects hidden
    state embeddings from upstream modules (e.g. :class:`WernickesArea`)
    and produces plain text output. The model weights are frozen; online
    training should be performed via LoRA adapters as described in
    ``AGENTS.md``.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def decode(
        self,
        hidden: torch.Tensor,
        temperature: float = 1.0,
        num_samples: int = 1,
        *,
        history: list[int] | None = None,
        repetition_penalty: float = 1.0,
    ) -> Iterable[tuple[str, float, int]]:
        """Generate one or more tokens from ``hidden``.

        Returns an iterable of ``(text, probability, token_id)`` tuples.
        ``num_samples`` controls how many speculative tokens are produced.
        """
        embeds = hidden.to(self.device)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(1)
        outputs = self.model(inputs_embeds=embeds)
        logits = outputs.logits
        if history and repetition_penalty > 1.0:
            hist_ids = torch.tensor(history, dtype=torch.long, device=logits.device)
            logits[..., hist_ids] /= repetition_penalty
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1).squeeze(1)

        # ``torch.multinomial`` with ``num_samples > 1`` would produce a single
        # sequence containing multiple tokens.  We instead sample tokens
        # independently so that each candidate represents exactly one token.
        sample_ids = torch.multinomial(probs, num_samples=num_samples, replacement=True).squeeze(0)
        sample_probs = probs.squeeze(0)[sample_ids]
        tokens = [self.tokenizer.decode([tok_id], skip_special_tokens=True) for tok_id in sample_ids]
        return list(zip(tokens, sample_probs.cpu().tolist(), sample_ids.cpu().tolist()))

if __name__ == "__main__":
    area = BrocasArea("../models/gpt2")
    dummy = torch.zeros(1, 1, area.model.config.n_embd)
    print(area.decode(dummy))

import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
from typing import Iterable, Optional
import numpy as np
from pathlib import Path

class WernickesArea:
    """Front half of GPT-2 (or alternative model) used for semantic encoding.

    The language modeling head is discarded so only hidden state embeddings are
    produced. Tokens are transient and never kept in memory after encoding.
    """

    def __init__(self, model_dir: str, device: str = "cpu", token_table_path: Optional[str] = None, trust_remote_code: bool = True):
        # Load tokenizer using ``AutoTokenizer`` to allow non-GPT models.
        # ``trust_remote_code=True`` enables loading custom code from local
        # repositories without prompting the user.  This is required for some
        # embedding models packaged with additional modules.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        # Some tokenizers may not define a padding token which breaks batching
        # when ``padding=True`` is requested.  Use the EOS token as padding to
        # keep sequence length consistent across calls.
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ``AutoModel`` can load both GPT-2 and alternative embedding models.
        # ``trust_remote_code=True`` allows models that ship custom code to
        # initialise correctly when loaded from a local directory.
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        self.model.to(device)
        self.model.eval()
        self.device = device

        # Determine embedding dimension reported by the model config.
        self.embed_dim = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", 768))
        # Project to 768-dim if the model uses a different size so downstream
        # modules remain compatible with GPT-2 embeddings.
        self.proj = None
        if self.embed_dim != 768:
            self.proj = torch.nn.Linear(self.embed_dim, 768, bias=False, device=device)

        self.token_table: Optional[torch.Tensor] = None
        if token_table_path:
            path = Path(token_table_path)
            if path.exists():
                data = np.load(path, allow_pickle=True).item()
                table = torch.tensor(data["embeddings"], dtype=torch.float32)
                self.token_table = table.to(device)

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Return hidden state embeddings for given ``texts``.

        Tokens are immediately discarded after computing embeddings.
        """
        # ``GPT2Tokenizer`` returns an empty ``input_ids`` sequence when an
        # empty string is provided.  This leads to a runtime error inside the
        # model when it tries to reshape a ``(batch, 0)`` tensor.  Replace empty
        # strings with the EOS token so they encode to a valid single token.
        fixed = [str(t) if t else self.tokenizer.eos_token for t in texts]
        # ``max_length`` previously forced sequences to pad to 1024 tokens which
        # meant short inputs ended up dominated by ``pad_token`` embeddings.
        # That drowned out the actual content when selecting the last token.
        # Remove the explicit length so padding is only applied up to the
        # longest input in the batch.
        tokens = self.tokenizer(
            fixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state
        if self.proj is not None:
            hidden = self.proj(hidden)
        del tokens  # ensure tokens don't persist
        return hidden

    @torch.no_grad()
    def lookup_tokens(self, token_ids: Iterable[int]) -> torch.Tensor:
        """Return embeddings for ``token_ids`` from the precomputed table."""
        if self.token_table is None:
            raise ValueError("token table not loaded")
        ids = torch.tensor(list(token_ids), dtype=torch.long, device=self.device)
        embeds = self.token_table[ids]
        return embeds.unsqueeze(1)


"""Microphone audio encoder using Whisper."""

from typing import Iterable

import torch
from transformers import (
    WhisperProcessor,
    WhisperModel,
    WhisperForConditionalGeneration,
)


class Cochlea:
    """Stream audio into Whisper to obtain embeddings or transcripts."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperModel.from_pretrained(model_dir)
        self.recognizer = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(device)
        self.recognizer.to(device)
        self.model.eval()
        self.recognizer.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, audio: Iterable[torch.Tensor]) -> torch.Tensor:
        """Return encoder features for ``audio`` samples."""
        # ``audio`` may be provided on GPU but Whisper's feature extractor
        # operates on CPU numpy arrays. Convert each tensor appropriately.
        cpu_audio = [a.detach().cpu().numpy() for a in audio]
        inputs = self.processor(cpu_audio, return_tensors="pt", sampling_rate=16000)
        inputs = inputs.to(self.device)
        features = self.model.encoder(inputs.input_features).last_hidden_state
        return features

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        *,
        language: str = "en",
        task: str = "transcribe",
        max_new_tokens: int = 16,
    ) -> str:
        """Return the transcription for ``audio`` using the decoder."""
        # ``audio`` may reside on GPU; Whisper's feature extractor expects
        # CPU numpy arrays. Convert before processing.
        audio_np = audio.detach().cpu().numpy()
        inputs = self.processor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        prompt_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
        generation_args = {"forced_decoder_ids": prompt_ids, "max_new_tokens": max_new_tokens}
        if attention_mask is not None:
            generation_args["attention_mask"] = attention_mask
        pred_ids = self.recognizer.generate(input_features, **generation_args)
        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        return text

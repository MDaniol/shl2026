"""Adapters for frozen foundation models (MOMENT, Chronos, BERT-family, ...).

Every adapter MUST:
- load weights from a pinned local cache directory (no live HF downloads at run time),
- expose an ``embed(frames: Tensor) -> Tensor`` function that runs under
  ``torch.inference_mode()`` with all parameters frozen,
- record the loaded revision SHA in MLflow tags.
"""

"""SHL 2026 — AGH submission pipeline.

Student-facing API (the inner loop)::

    from shl2026 import embeddings, make_head, evaluate, track, leaderboard

    Xtr = embeddings("moment", "train")          # shared cache, instant
    val = embeddings("moment", "validation")
    with track("alice", run_name="moment+mlp", seed=0,
               params={"fm": "moment", "head": "mlp"}) as run:
        head = make_head("mlp").fit(Xtr.X, Xtr.y)
        result = evaluate(head, val)
        run.log_eval(result)
    print(result.summary())

See ``STUDENTS.md`` for the full workflow.
"""

from __future__ import annotations

from .data.embeddings import EmbeddingSet, embeddings, list_available
from .eval.leaderboard import leaderboard
from .eval.metrics import EvalResult, evaluate, evaluate_predictions
from .heads.zoo import list_heads, make_head
from .submission.write import write_submission
from .tracking.autolog import track

__version__ = "0.0.0"

__all__ = [
    "EmbeddingSet",
    "EvalResult",
    "__version__",
    "embeddings",
    "evaluate",
    "evaluate_predictions",
    "leaderboard",
    "list_available",
    "list_heads",
    "make_head",
    "track",
    "write_submission",
]

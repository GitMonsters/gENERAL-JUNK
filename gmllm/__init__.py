"""gmllm — a small, honest, trainable LLM built from the salvageable parts of the
GitMonsters repositories (github.com/GitMonsters).

Provenance of each component is documented in ANALYSIS.md and in each module docstring.
The central change vs. the upstream code: every attention path is causal.
"""

__version__ = "0.1.0"

from .model import GMGPT, GMConfig
from .tokenizer import ByteBPETokenizer

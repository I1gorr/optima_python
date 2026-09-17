"""Kaggle heavy-LLM enrichment experiment support package.

Deliberately imports nothing heavy at package import time (no torch,
transformers, accelerate, or Optima's analyzer/clang bindings) so that
``import optima_kaggle`` is always cheap and safe, even before dependencies
are installed or a GPU runtime is selected.
"""

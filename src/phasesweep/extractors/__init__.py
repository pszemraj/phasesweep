"""Extractors pull a scalar value (metric or constraint) from a completed trial.

Each extractor is agnostic to the rest of the pipeline. It receives a TrialContext
and the extractor config, and returns a float (or raises ExtractorError).
"""

from phasesweep.extractors.base import ExtractorError, TrialContext, run_extractor

__all__ = ["ExtractorError", "TrialContext", "run_extractor"]

"""Errors whose message is shown to the user as it is (command line and Review Room). Kept free of heavy
imports so the command line and the API can catch them without loading OpenCV."""

from __future__ import annotations


class UnreadableVideoError(ValueError):
    """The file is not a video the pipeline can read. The message names the file, never its folder."""


class MissingModelError(FileNotFoundError):
    """A network checkpoint the analysis needs is not there (the weights are not in the repository)."""

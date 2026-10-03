"""CleanCmd: shared grid, file formats and scorer for language-to-floor-region grounding.

Every method (ETPNav in the old vlnce env, VLMaps and ours in the new env) writes predictions in the format in
cleancmd/io.py; cleancmd/score.py grades them against human masks. Only numpy is required.
"""
from .grid import Grid  # noqa: F401
from .io import Truth, load_jsonl, save_jsonl, to_cells  # noqa: F401

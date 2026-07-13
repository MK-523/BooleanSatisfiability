#!/usr/bin/env python3
"""Reproduce the two blocking issues in the repository-root SAT prototype."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LEGACY_ROOT = REPO_ROOT / "legacy" / "original"


def reproduce_preprocess_shape() -> dict:
    clauses = np.asarray([[1, 2, 3], [-1, 2, 3], [1, -2, 3]], dtype=np.int16)
    comparison = clauses[:, :, None] == -clauses[:, None, :]
    mask = comparison.sum(axis=2) == 0
    selected = clauses[mask]
    return {
        "input_shape": list(clauses.shape),
        "mask_shape": list(mask.shape),
        "output_shape": list(selected.shape),
        "output_rank": int(selected.ndim),
    }


def main() -> None:
    model_text = (LEGACY_ROOT / "model.py").read_text(encoding="utf-8")
    data_text = (LEGACY_ROOT / "data_utils.py").read_text(encoding="utf-8")
    source_digest = hashlib.sha256(
        model_text.encode("utf-8") + b"\0" + data_text.encode("utf-8")
    ).hexdigest()
    report = {
        "audit_source": "legacy/original",
        "audit_source_sha256": source_digest,
        "policy_input_expression": "torch.ones(self.num_variables)",
        "policy_uses_clauses": bool(re.search(r"self\.policy\(\s*clauses", model_text)),
        "policy_uses_constant_ones": "self.policy(torch.ones(self.num_variables))" in model_text,
        "preprocess_uses_two_dimensional_mask": "clauses[non_tautology_mask]" in data_text,
        "preprocess_shape_reproduction": reproduce_preprocess_shape(),
    }
    output = ROOT / "results" / "upstream_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

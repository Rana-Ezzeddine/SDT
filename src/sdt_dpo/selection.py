"""Shared eligibility rules for training and fixed-pair evaluation."""


def eligible_pair(row, *, all_confidence=False, confidence=("high", "medium"),
                  min_margin=0.0, min_sdt_margin=0.0):
    if not row.get("retain"):
        return False
    if row.get("label_mode") == "jev":
        basis = row["preference_basis"]
        if basis not in {"content_gate", "sdt_score"}:
            raise ValueError(f"Unknown Jev preference basis: {basis}")
        # A quality passer can win even with a lower SDT score.
        return basis == "content_gate" or float(row["sdt_margin"]) >= min_sdt_margin
    return (all_confidence or row.get("label_confidence") in confidence) and (
        float(row["label_margin"]) >= min_margin)

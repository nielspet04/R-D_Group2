"""Verb / noun / action top-1 and top-5 (EPIC protocol: action = ranked (verb, noun) pairs)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_hits(logits, target, ks=(1, 5)):
    pred = logits.topk(max(ks), dim=1).indices
    hit = pred.eq(target[:, None])
    return {k: hit[:, :k].any(1) for k in ks}


def action_hits(vl, nl, v, n, ks=(1, 5)):
    """Rank all (verb, noun) pairs by p(verb) * p(noun) and check the true pair."""
    joint = vl.log_softmax(1)[:, :, None] + nl.log_softmax(1)[:, None, :]
    pred = joint.flatten(1).topk(max(ks), dim=1).indices
    tgt = v * nl.shape[1] + n
    hit = pred.eq(tgt[:, None])
    return {k: hit[:, :k].any(1) for k in ks}


class Meter:
    def __init__(self):
        self.n = 0
        self.loss_sum = 0.0
        self.hits = {f"{t}_top{k}": 0 for t in ("verb", "noun", "action") for k in (1, 5)}

    @torch.no_grad()
    def update(self, vl, nl, v, n):
        m = (v >= 0) & (n >= 0)  # unlabelled clips (test csv) are skipped
        if int(m.sum()) == 0:
            return
        vl, nl, v, n = vl[m].float(), nl[m].float(), v[m], n[m]
        self.n += int(m.sum())
        self.loss_sum += float(F.cross_entropy(vl, v, reduction="sum")
                               + F.cross_entropy(nl, n, reduction="sum"))
        for name, h in (("verb", topk_hits(vl, v)), ("noun", topk_hits(nl, n)),
                        ("action", action_hits(vl, nl, v, n))):
            for k, x in h.items():
                self.hits[f"{name}_top{k}"] += int(x.sum())

    def compute(self):
        if self.n == 0:
            return {}
        out = {k: 100.0 * h / self.n for k, h in self.hits.items()}
        out["loss"] = self.loss_sum / self.n
        out["n"] = self.n
        return out

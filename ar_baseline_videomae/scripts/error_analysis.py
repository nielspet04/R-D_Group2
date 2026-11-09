"""Error analysis of the AR baseline (checklist: inspect strong/weak predictions, record error types).

Run in the rd_lab env, from ar_baseline_videomae/:
  python scripts/error_analysis.py --preds results/videomae_b_f16_seed1/preds_val.csv \
      --train_csv ../data/processed/EPIC_100_train_participant_split.csv

Writes into <preds folder>/analysis/: summary.txt and several csv tables.
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True, help="preds_val.csv written by train.py --export")
ap.add_argument("--train_csv", help="train csv: gives class names (verb/noun columns) and train frequencies")
ap.add_argument("--min_support", type=int, default=30, help="min val clips for a class to be ranked")
a = ap.parse_args()

out = os.path.join(os.path.dirname(os.path.abspath(a.preds)), "analysis")
os.makedirs(out, exist_ok=True)
d = pd.read_csv(a.preds)
d["participant"] = d["narration_id"].str.split("_").str[0]

# ---- class names and train frequencies (optional) ----
names = {"verb": {}, "noun": {}}
cnt = {"verb": {}, "noun": {}}
if a.train_csv:
    tr = pd.read_csv(a.train_csv)
    for k in ("verb", "noun"):
        cc = f"{k}_class"
        if k in tr.columns and cc in tr.columns:
            names[k] = tr.drop_duplicates(cc).set_index(cc)[k].to_dict()
        if cc in tr.columns:
            cnt[k] = tr[cc].value_counts().to_dict()


def nm(k, i):
    return f"{names[k].get(i, '?')} ({i})" if names[k] else str(i)


lines = []


def say(s=""):
    print(s)
    lines.append(s)


# ---- overall, recomputed from the csv (sanity check against metrics_eval.json) ----
d["verb_ok"] = d["verb_pred"] == d["verb_true"]
d["noun_ok"] = d["noun_pred"] == d["noun_true"]
d["act_ok"] = d["verb_ok"] & d["noun_ok"]
for k in ("verb", "noun"):
    d[f"{k}_ok5"] = [str(t) in str(s).split() for t, s in zip(d[f"{k}_true"], d[f"{k}_top5"])]
say(f"clips: {len(d)}")
say(f"verb  top1 {100 * d.verb_ok.mean():.2f}  top5 {100 * d.verb_ok5.mean():.2f}")
say(f"noun  top1 {100 * d.noun_ok.mean():.2f}  top5 {100 * d.noun_ok5.mean():.2f}")
say(f"action top1 {100 * d.act_ok.mean():.2f}")

# ---- how actions go wrong ----
say("\n== action outcome (top-1) ==")
for label, m in (("verb right + noun right", d.verb_ok & d.noun_ok),
                 ("verb right, noun wrong", d.verb_ok & ~d.noun_ok),
                 ("verb wrong, noun right", ~d.verb_ok & d.noun_ok),
                 ("both wrong", ~d.verb_ok & ~d.noun_ok)):
    say(f"{label:26} {100 * m.mean():5.1f} %")

# ---- per class, confusions, head/tail ----
for k in ("verb", "noun"):
    t, p = f"{k}_true", f"{k}_pred"
    rows = []
    for c, s in d.groupby(t):
        wrong = s.loc[s[p] != c, p]
        top = wrong.value_counts().index[0] if len(wrong) else None
        rows.append({"class": c, "name": names[k].get(c, ""), "val_clips": len(s),
                     "train_clips": cnt[k].get(c, ""), "recall_%": round(100 * (s[p] == c).mean(), 1),
                     "most_confused_with": nm(k, top) if top is not None else ""})
    pc = pd.DataFrame(rows).sort_values("val_clips", ascending=False)
    pc.to_csv(os.path.join(out, f"per_class_{k}.csv"), index=False)
    ranked = pc[pc.val_clips >= a.min_support].sort_values("recall_%", ascending=False)
    say(f"\n== {k}: classes with >= {a.min_support} val clips: {len(ranked)} of {len(pc)} ==")
    say("strongest:")
    for _, r in ranked.head(8).iterrows():
        say(f"  {nm(k, r['class']):28} recall {r['recall_%']:5.1f}%  ({r['val_clips']} clips)")
    say("weakest:")
    for _, r in ranked.tail(8).iloc[::-1].iterrows():
        say(f"  {nm(k, r['class']):28} recall {r['recall_%']:5.1f}%  ({r['val_clips']} clips)  -> often {r['most_confused_with']}")
    conf = (d[d[p] != d[t]].groupby([t, p]).size().sort_values(ascending=False).head(20)
            .rename("count").reset_index())
    conf["true_name"] = [nm(k, i) for i in conf[t]]
    conf["pred_name"] = [nm(k, i) for i in conf[p]]
    conf.to_csv(os.path.join(out, f"confusions_{k}.csv"), index=False)
    say("most frequent confusions (true -> predicted):")
    for _, r in conf.head(8).iterrows():
        say(f"  {r['true_name']:26} -> {r['pred_name']:26} {r['count']}x")
    if cnt[k]:
        tc = d[t].map(lambda c: cnt[k].get(c, 0))
        for label, m in (("frequent (>=100 train clips)", tc >= 100),
                         ("middle (10-99)", (tc >= 10) & (tc < 100)),
                         ("rare (<10)", tc < 10)):
            if m.sum():
                say(f"  {k} accuracy, {label:30} {100 * (d.loc[m, p] == d.loc[m, t]).mean():5.1f} %  ({int(m.sum())} clips)")

# ---- per participant ----
pp = d.groupby("participant").agg(clips=("act_ok", "size"), verb_top1=("verb_ok", "mean"),
                                  noun_top1=("noun_ok", "mean"), action_top1=("act_ok", "mean"))
pp[["verb_top1", "noun_top1", "action_top1"]] = (100 * pp[["verb_top1", "noun_top1", "action_top1"]]).round(1)
pp.to_csv(os.path.join(out, "per_participant.csv"))
say("\n== per participant (validation) ==")
say(pp.to_string())

# ---- examples to look at by hand (with video segment ids) ----
def view(s):
    s = s.copy()
    for k in ("verb", "noun"):
        s[f"{k}_true_name"] = [nm(k, i) for i in s[f"{k}_true"]]
        s[f"{k}_pred_name"] = [nm(k, i) for i in s[f"{k}_pred"]]
    cols = ["narration_id", "narration", "verb_true_name", "verb_pred_name", "noun_true_name", "noun_pred_name"]
    return s[[c for c in cols if c in s.columns]]


view(d[d.act_ok].sample(min(30, int(d.act_ok.sum())), random_state=0)).to_csv(os.path.join(out, "examples_right.csv"), index=False)
view(d[~d.act_ok].sample(min(30, int((~d.act_ok).sum())), random_state=0)).to_csv(os.path.join(out, "examples_wrong.csv"), index=False)
say(f"\nTables written to {out}")
with open(os.path.join(out, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
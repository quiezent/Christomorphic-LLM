"""Who is speaking, scene by scene (CPU; reads what speakers_probe.py scored). Added 21 Nov 2026 after an independent review of
speakers_probe.py, before any adapter had been scored on the table of scenes; base's 8-scene smoke run had been seen.

WHY. speakers_probe.py caps the margin speech by speech. A speech's margin, log p(speech | its own frame) − log p(speech | the other
speaker's frame), also holds how much each frame lifts whatever follows it: a long frame that names its people lifts both speeches.
On base's smoke run two speeches with short frames set the whole capped figure. The Isaiah 11:3-5 test was open to
change. The figure fixed on 15 Nov stays in speakers_probe.py read; this stands beside it.

THE SCENE MARGIN. Add a scene's two margins and halve:
    [log p(a | a's frame) + log p(b | b's frame) − log p(a | b's frame) − log p(b | a's frame)] / 2, in nats a speech.
It is how much more the model expects the Bible's joining of the two speeches to their speakers than the swapped joining. Whatever
a frame adds to both speeches cancels exactly. The twins (two verses with no speaker, split and swapped the same way) get the same sum.
- For base: the mean, the middle scene, and the share of scenes above zero.
- For an arm: its scene margin minus base's, scene by scene: the mean with a 95% interval in which the scenes of one chapter may
  move together, the middle scene, and the share of scenes where the arm stands above base. And the same beyond the twins.
- No cap: the mean is beside the middle scene, so a few large scenes cannot hide.
- Form B's adapters (names beginning "tongues_"): the scenes are split into those no step read (every frame, speech and twin verse
  outside runs/tongues/read_verses.json) and those a step read. Only the first are "on verses no step read".
It is shown, not read: a likelihood margin, not a measure of judgment.

    python speakers_scene.py            # -> runs/speakers/scene.md, scene.json
"""
import json
import math
from pathlib import Path

import numpy as np

from measure_base import in_venv
from speakers_probe import KINDS, OUT, interval, sha

HERE = Path(__file__).parent


def margins(j, part, rows):
    """The scene margin of each scene in rows (half the sum of its two speeches' margins)."""
    return np.array([(sum(j[part]["own"][n]) - sum(j[part]["other"][n])) / 2 for n in rows], dtype=float)


def main():
    in_venv()
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec_path = OUT / "scenes.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))["scenes"]
    now = sha(spec_path)
    runs = {}
    for p in sorted(OUT.glob("score_*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if j["limit"] == 0 and j["scenes_sha"] == now:
            runs[j["name"]] = j
    if "base" not in runs:
        raise SystemExit("base has not been scored on this table: python speakers_probe.py score --arm base --twice")
    read = set(json.loads((HERE / "runs" / "tongues" / "read_verses.json").read_text(encoding="utf-8")))

    def verses_of(sc):
        vs = {sc["a"]["verse"], sc["b"]["verse"], *sc["a"]["speech_verses"], *sc["b"]["speech_verses"]}
        if sc["twin"]:
            vs |= {sc["twin"]["a"]["verse"], sc["twin"]["b"]["verse"]}
        return vs
    was_read = [bool(verses_of(sc) & read) for sc in spec]
    subsets = {"all scenes": lambda n: True, "scenes the word counter does not settle": lambda n: not spec[n]["counter"]["right"]}
    splits = {"all": lambda n: True, "no step read": lambda n: not was_read[n], "a step read": lambda n: was_read[n]}
    kinds = list(KINDS.items()) + [("all", "all kinds")]
    fmt = lambda t: "–" if math.isnan(t[0]) else f"{t[0]:+.2f} (no interval)" if math.isnan(t[1]) else f"{t[0]:+.2f} ({t[1]:+.2f} to {t[2]:+.2f})"
    base = runs["base"]
    md = ["# Who is speaking, scene by scene", "",
          "The scene margin: how much more the model expects the Bible's joining of two speeches to their speakers than the swapped joining, in nats a speech."
          " A frame's own pull on whatever follows it cancels. Base's rows are its own values; an arm's rows are the arm minus base, scene by scene."
          " Intervals are 95%; the scenes of one chapter may move together. Shown, not read.", ""]
    if "again" in base:
        a, b = margins(base, "scenes", range(len(spec))), margins(base["again"], "scenes", range(len(spec)))
        md += [f"The card's own noise (base scored twice, in other batches): a scene margin differs by {np.abs(a - b).mean():.3f} nats on average and {np.abs(a - b).max():.3f} at most.", ""]
    out = {}
    for s, keep in subsets.items():
        md += [f"## {s[0].upper() + s[1:]}", "", "| | scenes | kind | scenes | scene margin | middle scene | share above | twins | scenes − twins |", "|---|---|---|---:|---|---:|---:|---|---|"]
        for arm, j in runs.items():
            for split, in_split in (splits.items() if arm.startswith("tongues_") else [("all", splits["all"])]):
                for k, label in kinds:
                    rows = [n for n in range(len(spec)) if keep(n) and in_split(n) and (k == "all" or spec[n]["kind"] == k)]
                    if len(rows) < 5 or (k != "all" and split != "all"):
                        continue
                    tw = [n for n in rows if spec[n]["twin"]]
                    cl, ctw = [spec[n]["cluster"] for n in rows], [spec[n]["cluster"] for n in tw]
                    S, T, P = margins(j, "scenes", rows), margins(j, "twins", tw), margins(j, "scenes", tw)
                    if arm != "base":
                        S, T, P = S - margins(base, "scenes", rows), T - margins(base, "twins", tw), P - margins(base, "scenes", tw)
                    rec = {"scenes": len(rows), "mean": interval((1, S, cl)), "middle": float(np.median(S)), "share above": float((S > 0).mean()),
                           "twins": interval((1, T, ctw)), "beyond": interval((1, P - T, ctw)), "twinned": len(tw)}
                    out.setdefault(arm, {}).setdefault(s, {}).setdefault(split, {})[k] = rec
                    md.append(f"| {arm if arm == 'base' else arm + ' − base'} | {split} | {label} | {len(rows)} | {fmt(rec['mean'])} | {rec['middle']:+.2f} | {rec['share above']:.0%} |"
                              f" {fmt(rec['twins'])} | {fmt(rec['beyond'])} |")
        md.append("")
    md += ["For base, \"share above\" is the share of scenes in which the Bible's joining is the likelier one. For an arm it is the share of scenes in which the arm's scene margin is above base's.",
           "", "A negative means the model remains closer to the pattern of this world, never evidence against Scripture."]
    clean = lambda x: None if isinstance(x, float) and math.isnan(x) else {k: clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x] if isinstance(x, (list, tuple)) else x
    (OUT / "scene.json").write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
    (OUT / "scene.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""
Methods A-D of eval_anchor.py on GOAT-Bench language goals (make_goat_cmds.py rows, HM3D floors).

Everything is eval_anchor's: the same candidate rules, the same fixed LLM prompt and options (and its reply cache),
the same 1 m scoring and failure attribution. Only the scene loader differs: rooms come from HM3D region ids
(make_goat_cmds.region_finder) instead of MP3D .house polygons. Method C keeps detections in the robot's current
region (GOAT commands never name a room).

python cleancmd/anchor/eval_goat.py --datasets TEEsavR23oF_1,TEEsavR23oF_2 --methods A,B,C,D
"""
import argparse
import json

import eval_anchor as E
from make_anchor_cmds import DATA_DIR
from make_goat_cmds import region_finder


def load_scene(tag):
    rows = [json.loads(l) for l in open(E.HERE / f"{tag}_anchor_cmds.jsonl")]
    det = json.load(open(E.HERE / f"{tag}_detections.json"))
    src = json.load(open(DATA_DIR / tag / "source.json"))
    room_of = region_finder(json.load(open(DATA_DIR / tag / "objects.json")), src["floor_y"])
    for cat, ds in det["detections"].items():
        for d in ds:
            d["category"] = cat
            d["region"], d["room"] = room_of(d["center_xz"])
    return {"scene": tag, "rows": rows, "det": det, "regions": {}, "room_of": room_of}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--methods", default="A,B,C,D")
    args = ap.parse_args()
    llm = E.LLM() if "D" in args.methods else None
    E.RESULTS.mkdir(exist_ok=True)
    allp = []
    for tag in args.datasets.split(","):
        S = load_scene(tag)
        for mk in args.methods.split(","):
            preds = []
            for row in S["rows"]:
                d, meta = E.pick(S, row, mk, llm)
                preds.append({"scene": tag, "command_id": row["command_id"], "pose_idx": row["pose_idx"],
                              "command": row["command"], "form": row["form"], "category": row["category"],
                              "method": E.METHODS[mk], "pick": d["det_id"] if d else None,
                              "pick_xz": d["center_xz"] if d else None, "gt_instance": row["gt"]["instance_id"],
                              "gt_xz": row["gt"]["center_xz"], "outcome": E.outcome(S, row, d), "meta": meta})
            with open(E.RESULTS / f"goat_{tag}_pred_{E.METHODS[mk]}.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            allp += preds
            acc = sum(p["outcome"] == "correct" for p in preds) / len(preds)
            print(f"{tag} {E.METHODS[mk]}: {acc:.1%} of {len(preds)}", flush=True)
    by_form, by_scene = E.table(allp, "form"), E.table(allp, "scene")
    print(by_form + "\n\n" + by_scene)
    with open(E.RESULTS / "goat_tables.md", "w") as f:
        f.write(f"# GOAT-Bench language goals, object picking ({args.datasets})\n\nCorrect = chosen detection centre "
                f"within {E.CORRECT_M} m of the GOAT goal instance's centre. LLM: {E.LLM_MODEL}, options {E.LLM_OPTIONS}."
                f"\n\n## Per command form\n{by_form}\n\n## Per floor dataset\n{by_scene}\n")


if __name__ == "__main__":
    main()

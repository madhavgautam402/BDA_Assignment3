import argparse
import os
import pandas as pd
from common import load_ratings, directed_edges, MOVIE_OFFSET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--output-dir", default="partA/output")
    args = ap.parse_args()

    src, dst = directed_edges(load_ratings(args.data_dir))
    src, dst = src.tolist(), dst.tolist()
    vertices = sorted(set(src) | set(dst))
    N = len(vertices)
    d = args.damping

    out_deg = {}
    for u in src:
        out_deg[u] = out_deg.get(u, 0) + 1

    pr = {v: 1.0 / N for v in vertices}                  # PR_0
    trace = [(0, sum(pr.values()), 0.0)]

    for t in range(args.iterations):
        incoming = {v: 0.0 for v in vertices}
        for u, v in zip(src, dst):                       # one message per directed edge
            incoming[v] += pr[u] / out_deg[u]
        new = {v: (1 - d) / N + d * incoming[v] for v in vertices}
        l1 = sum(abs(new[v] - pr[v]) for v in vertices)
        pr = new
        trace.append((t + 1, sum(pr.values()), l1))

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(trace, columns=["iteration", "rank_sum", "l1_change"]).to_csv(
        os.path.join(args.output_dir, "pagerank_trace.csv"), index=False)

    rows = []
    for v in vertices:
        is_movie = v >= MOVIE_OFFSET
        rows.append({"vertex_id": v,
                     "vertex_type": "movie" if is_movie else "user",
                     "original_id": v - MOVIE_OFFSET if is_movie else v,
                     "pagerank": pr[v]})
    out = pd.DataFrame(rows).sort_values(["pagerank", "vertex_id"],
                                         ascending=[False, True])
    out.to_csv(os.path.join(args.output_dir, "pagerank_local.csv"), index=False)
    print(out.head(10).to_string(index=False))
    print(pd.DataFrame(trace, columns=["iteration", "rank_sum", "l1_change"]).tail(3))


if __name__ == "__main__":
    main()
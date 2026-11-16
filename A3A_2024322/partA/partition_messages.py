import argparse
import os
import numpy as np
import pandas as pd
from common import load_ratings, directed_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--message-bytes", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--output", default="partA/output/partition_messages.csv")
    args = ap.parse_args()

    src, dst = directed_edges(load_ratings(args.data_dir))
    total = len(src)

    rows = []
    for P in args.workers:
        remote = int(np.sum((src % P) != (dst % P)))   # owner(v,P) = v mod P
        local = total - remote
        remote_bytes = remote * args.message_bytes
        rows.append({
            "workers": P,
            "local_directed_edges": local,
            "remote_directed_edges": remote,
            "remote_fraction": remote / total,
            "messages_per_iteration": total,
            "remote_bytes_per_iteration": remote_bytes,
            "remote_bytes_for_10_iterations": remote_bytes * args.iterations,
        })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_ratings, MOVIE_OFFSET

HUB_MOVIE = 200000
SYNTH_USER_BASE = 300000


def build_edges(df, H):
    """HUB-GRAPH logical edges as two arrays (a, b), one entry per undirected edge."""
    a = df["user_id"].to_numpy(dtype=np.int64)
    b = df["movie_id"].to_numpy(dtype=np.int64) + MOVIE_OFFSET
    syn = SYNTH_USER_BASE + np.arange(1, H + 1, dtype=np.int64)
    a = np.concatenate([a, syn])
    b = np.concatenate([b, np.full(H, HUB_MOVIE, dtype=np.int64)])
    return a, b


def simulate(df, P, H):
    a, b = build_edges(df, H)
    E = len(a)
    endpoints = np.concatenate([a, b])
    vertices, degree = np.unique(endpoints, return_counts=True)
    n_vertices = len(vertices)

    # ---- Pregel-style vertex partition: owner(v) = v mod P ----
    vwork = np.bincount(vertices % P, weights=degree, minlength=P).astype(np.int64)
    v_mean = vwork.mean()
    v_imb = vwork.max() / v_mean
    hub_owner = HUB_MOVIE % P
    hub_degree = int(degree[vertices == HUB_MOVIE][0])

    # ---- PowerGraph-style vertex cut: edge_id mod P over sorted (min,max) pairs ----
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    order = np.lexsort((hi, lo))              # sort by lo, then hi
    lo, hi = lo[order], hi[order]
    edge_id = np.arange(E, dtype=np.int64)
    ew = edge_id % P
    ework = np.bincount(ew, minlength=P)
    e_mean = ework.mean()
    e_imb = ework.max() / e_mean

    keys = np.unique(np.concatenate([lo * P + ew, hi * P + ew]))   # unique (vertex, worker)
    rep_vertices, replica_count = np.unique(keys // P, return_counts=True)
    assert len(rep_vertices) == n_vertices
    hub_replicas = int(replica_count[rep_vertices == HUB_MOVIE][0])
    repl_factor = replica_count.sum() / n_vertices
    gas = int(2 * (replica_count - 1).sum())

    return {
        "workers": P,
        "hub_degree": hub_degree,
        "vertex_partition_max_work": int(vwork.max()),
        "vertex_partition_mean_work": float(v_mean),
        "vertex_partition_imbalance": float(v_imb),
        "hub_owner": int(hub_owner),
        "hub_owner_work": int(vwork[hub_owner]),
        "edge_partition_max_work": int(ework.max()),
        "edge_partition_mean_work": float(e_mean),
        "edge_partition_imbalance": float(e_imb),
        "replication_factor": float(repl_factor),
        "hub_replica_count": hub_replicas,
        "gas_sync_messages": gas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hub-degree", type=int, default=50000)
    ap.add_argument("--scaling-degrees", type=int, nargs="+", default=[1000, 10000, 50000])
    ap.add_argument("--skip-scaling", action="store_true")
    ap.add_argument("--output-dir", default="partA/output")
    args = ap.parse_args()

    df = load_ratings(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    res = simulate(df, args.workers, args.hub_degree)
    out = pd.DataFrame([res])
    out.to_csv(os.path.join(args.output_dir, "powergraph_results.csv"), index=False)
    print(out.T.to_string(header=False))

    if not args.skip_scaling:
        rows = []
        for H in args.scaling_degrees:
            r = simulate(df, args.workers, H)
            rows.append({k: r[k] for k in ["hub_degree", "vertex_partition_imbalance",
                                           "edge_partition_imbalance",
                                           "replication_factor", "gas_sync_messages"]})
        sc = pd.DataFrame(rows)
        sc.to_csv(os.path.join(args.output_dir, "powergraph_scaling.csv"), index=False)
        print(sc.to_string(index=False))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sc["hub_degree"], sc["vertex_partition_imbalance"], marker="o",
                label="Vertex partition (Pregel-style)")
        ax.plot(sc["hub_degree"], sc["edge_partition_imbalance"], marker="s",
                label="Edge partition / vertex cut (PowerGraph-style)")
        ax.set_xlabel("Hub degree H")
        ax.set_ylabel("Load imbalance = max work / mean work")
        ax.set_title(f"Load imbalance vs hub degree (P={args.workers})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "powergraph_scaling.png"), dpi=150)


if __name__ == "__main__":
    main()
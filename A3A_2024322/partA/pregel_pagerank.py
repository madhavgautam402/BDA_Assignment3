import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame
from graphframes.lib import Pregel

MOVIE_OFFSET = 100000


def load_edges(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "u.data"), sep="\t", header=None,
                     names=["user_id", "movie_id", "rating", "timestamp"])
    u = df["user_id"].to_numpy(dtype=np.int64)
    m = df["movie_id"].to_numpy(dtype=np.int64) + MOVIE_OFFSET
    return np.concatenate([u, m]), np.concatenate([m, u])    # u->m and m->u


def local_pagerank(src, dst, iterations, d):
    """Plain reference implementation, same formulas as pagerank_local.py."""
    vertices = np.unique(np.concatenate([src, dst]))
    N = len(vertices)
    s = np.searchsorted(vertices, src)
    t = np.searchsorted(vertices, dst)
    out_deg = np.bincount(s, minlength=N).astype(np.float64)
    pr = np.full(N, 1.0 / N)
    for _ in range(iterations):
        inc = np.bincount(t, weights=pr[s] / out_deg[s], minlength=N)
        pr = (1 - d) / N + d * inc
    return vertices, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--output", default="partA/output/pregel_pagerank.csv")
    args = ap.parse_args()
    d = args.damping

    src, dst = load_edges(args.data_dir)
    ref_vertices, ref_pr = local_pagerank(src, dst, args.iterations, d)

    spark = (SparkSession.builder.appName("pregel_pagerank")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir(tempfile.mkdtemp(prefix="pregel_ckpt_"))

    v0 = spark.createDataFrame([(int(x),) for x in ref_vertices], "id long")
    e = spark.createDataFrame(list(zip(src.tolist(), dst.tolist())), "src long, dst long")
    N = v0.count()

    # static vertex column outDegree (needed to split a rank among out-edges)
    v = v0.join(GraphFrame(v0, e).outDegrees, on="id", how="left")
    g = GraphFrame(v, e)

    result = (
        g.pregel
        .setMaxIter(args.iterations)                                   # supersteps
        .withVertexColumn(                                             # vertex state + update
            "rank",
            F.lit(1.0 / N),                                            # PR_0
            F.lit((1 - d) / N) + F.lit(d) * F.coalesce(Pregel.msg(), F.lit(0.0)),
        )
        .sendMsgToDst(Pregel.src("rank") / Pregel.src("outDegree"))    # message on every edge
        .aggMsgs(F.sum(Pregel.msg()))                                  # message aggregation
        .run()
    )

    rows = result.select("id", "rank").collect()
    pdf = pd.DataFrame({"id": [int(r["id"]) for r in rows],
                        "rank": [float(r["rank"]) for r in rows]})
    pdf = pdf.sort_values("id").reset_index(drop=True)
    assert (pdf["id"].to_numpy() == ref_vertices).all()
    pregel_pr = pdf["rank"].to_numpy(dtype=np.float64)

    is_movie = pdf["id"] >= MOVIE_OFFSET
    out = pd.DataFrame({
        "vertex_id": pdf["id"],
        "vertex_type": np.where(is_movie, "movie", "user"),
        "original_id": np.where(is_movie, pdf["id"] - MOVIE_OFFSET, pdf["id"]),
        "pagerank": pregel_pr,
    }).sort_values(["pagerank", "vertex_id"], ascending=[False, True])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    diff = np.abs(pregel_pr - ref_pr)
    cmp_path = os.path.join(os.path.dirname(args.output) or ".", "pagerank_compare.csv")
    pd.DataFrame([{
        "iterations": args.iterations,
        "max_abs_difference": float(diff.max()),
        "mean_abs_difference": float(diff.mean()),
        "local_rank_sum": float(ref_pr.sum()),
        "pregel_rank_sum": float(pregel_pr.sum()),
    }]).to_csv(cmp_path, index=False)
    print(open(cmp_path).read())
    spark.stop()


if __name__ == "__main__":
    main()
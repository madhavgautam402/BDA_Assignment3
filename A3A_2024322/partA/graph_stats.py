import argparse
import os
from common import load_ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="partA/output/graph_stats.csv")
    args = ap.parse_args()

    df = load_ratings(args.data_dir)
    user_deg = df.groupby("user_id").size()    # index sorted ascending
    movie_deg = df.groupby("movie_id").size()

    rows = [
        ("logical_vertices", len(user_deg) + len(movie_deg)),
        ("logical_user_vertices", len(user_deg)),
        ("logical_movie_vertices", len(movie_deg)),
        ("logical_edges", len(df)),
        ("directed_edges", 2 * len(df)),
        ("min_rating", int(df["rating"].min())),
        ("max_rating", int(df["rating"].max())),
        ("max_user_degree", int(user_deg.max())),
        ("max_user_id", int(user_deg.idxmax())),      # ties -> smallest id
        ("max_movie_degree", int(movie_deg.max())),
        ("max_movie_id", int(movie_deg.idxmax())),    # original movie_id
    ]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        f.write("metric,value\n")
        for k, v in rows:
            f.write(f"{k},{v}\n")
    print(open(args.output).read())


if __name__ == "__main__":
    main()
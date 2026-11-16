import os
import numpy as np
import pandas as pd

MOVIE_OFFSET = 100000


def load_ratings(data_dir):
    """Read u.data: user_id, movie_id, rating, timestamp (tab separated)."""
    path = os.path.join(data_dir, "u.data")
    return pd.read_csv(path, sep="\t", header=None,
                       names=["user_id", "movie_id", "rating", "timestamp"])


def directed_edges(df):
    """Execution graph G_D: each rating {u,m} -> u->m and m->u. Movie id = 100000 + movie_id."""
    u = df["user_id"].to_numpy(dtype=np.int64)
    m = df["movie_id"].to_numpy(dtype=np.int64) + MOVIE_OFFSET
    src = np.concatenate([u, m])
    dst = np.concatenate([m, u])
    return src, dst
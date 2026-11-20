# Assignment 3A: Graph Processing, PageRank, Pregel and PowerGraph

Student: Madhav Gautam, roll number 2024322

## Hardware
OS: <Windows 11 ...> | CPU: <model> | Logical CPUs: <8> | RAM: <8> GB

## Software versions (exact output)
```
<paste: python --version>
<paste: java -version>
<paste: pip freeze>
```

## Dataset
Extracted MovieLens 100K at `<DATA>` (passed with `--data-dir`, not included in the zip).
Archive MD5 checked: 0e33842e24a9c977be4e0107933c0723 (matches). u.data has 100000 rows.

## Known limitations / deviations
None of the pinned versions were changed. Two extra things were needed on Windows and are
documented above: `typing_extensions` (undeclared dependency of graphframes-py) and `winutils`.
## Environment

Python 3.14.10, Java Temurin 21, Spark/PySpark 4.2.0, GraphFrames 0.12.3
(JVM package `io.graphframes:graphframes-spark4_2.13:0.12.3`), NetworkX 3.7,
pandas 3.1.0, matplotlib 3.11.2. Exact versions are listed in `environment.txt`.
Tested on Windows 11. Spark runs as `local[*]`; no cluster is used.

## Setup

```
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:       source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Java 21 must be the `java` on PATH (and JAVA_HOME) when running spark-submit.
- If importing GraphFrames fails with `No module named 'typing_extensions'`, run
  `pip install typing_extensions` (an undeclared dependency of graphframes-py 0.12.3).
- On Windows, Spark needs `winutils.exe` and `hadoop.dll` in `%HADOOP_HOME%\bin`.
  This is not needed on Linux/macOS.
- On Windows, set `PYSPARK_PYTHON` to the virtual environment's `python.exe`.
- The MovieLens data is NOT included in this submission. Pass the extracted
  `ml-100k` folder with `--data-dir`.

## Commands

Run from the submission root (`<DATA>` = path to the extracted ml-100k folder):

```
python partA/graph_stats.py --data-dir <DATA> --output partA/output/graph_stats.csv
python partA/partition_messages.py --data-dir <DATA> --workers 4 8 16 32 --message-bytes 16 --iterations 10 --output partA/output/partition_messages.csv
python partA/pagerank_local.py --data-dir <DATA> --iterations 10 --damping 0.85 --output-dir partA/output
spark-submit --master "local[*]" --packages io.graphframes:graphframes-spark4_2.13:0.12.3 partA/pregel_pagerank.py --data-dir <DATA> --iterations 10 --damping 0.85 --output partA/output/pregel_pagerank.csv
python partA/powergraph_sim.py --data-dir <DATA> --workers 8 --hub-degree 50000 --output-dir partA/output
```

Notes on the programs:
- `partA/common.py` is a small shared helper (data loading, edge construction) used by the
  plain-Python scripts.
- `pregel_pagerank.py` recomputes a local reference for the same number of iterations and
  writes `pagerank_compare.csv` into the same folder as its `--output` file.
- `powergraph_sim.py` writes `powergraph_results.csv` for the given `--workers` and
  `--hub-degree`. It also writes `powergraph_scaling.csv` and `powergraph_scaling.png` for
  H in {1000, 10000, 50000} using the same `--workers`. Use `--skip-scaling` to skip this
  and `--scaling-degrees` to change the H values.

## Results summary

- Graph: 2625 vertices (943 users, 1682 movies), 100000 logical edges, 200000 directed edges.
  Maximum user degree 737 (user 405); maximum movie degree 583 (movie 50).
- Remote message fraction with owner(v,P) = v mod P: P=4 75.1%, P=8 87.6%, P=16 93.8%,
  P=32 96.9%.
- PageRank: rank_sum = 1.0 at every iteration. Pregel vs local after 10 iterations:
  max absolute difference 4.3e-18 (limit 1e-8).
- PowerGraph simulation (P=8, H=50000): vertex-partition imbalance 2.13, edge-partition
  imbalance 1.00, hub replicated on all 8 workers, 30738 GAS synchronization messages.

## A2: Why can communication matter on one laptop?

Even when everything runs on one laptop, the message pattern we count would become network
traffic once vertices are spread over machines. With owner(v,P) = v mod P, most directed
edges are remote: 75.1% for P=4 and 96.9% for P=32. PageRank sends one message per directed
edge per iteration, so remote traffic is 2.4 MB to 3.1 MB per iteration and 24 MB to 31 MB
over 10 iterations at 16 bytes per message. Network bandwidth, serialization and latency
are much slower than local memory, and every superstep must wait for the slowest worker.
So communication volume, not arithmetic, often limits scaling.

## A4: Mapping PageRank to the GraphFrames Pregel program

All line numbers refer to `partA/pregel_pagerank.py`.

- **Vertex state:** the `rank` column created by `withVertexColumn("rank", ...)` (line 65),
  initialised to 1/N (line 67). The static `outDegree` column (lines 58-59) is also part of
  the vertex state.
- **Message generation:** `sendMsgToDst(Pregel.src("rank") / Pregel.src("outDegree"))`
  (line 70), executed for every directed edge u -> v.
- **Message aggregation:** `aggMsgs(F.sum(Pregel.msg()))` (line 71) sums the messages that
  arrive at each vertex.
- **Vertex update:** the update expression of `withVertexColumn` (line 68):
  `(1-d)/N + d * coalesce(msg, 0)`.
- **Superstep:** one send, aggregate and update round. `setMaxIter(args.iterations)`
  (line 64) repeats it exactly 10 times, matching the local implementation.

## A5: Why does a vertex cut help a hub?

In a vertex partition one worker owns the hub and must process all H of its adjacency
entries, so it becomes a straggler and the imbalance grows with H (2.13 at H=50000). The
vertex cut spreads the hub's edges over all workers by edge_id mod P, so each worker handles
about |E|/P edges and the imbalance stays near 1.0. The price is replication and
synchronization: the hub has replicas on all 8 workers and each iteration needs
2*(replicas-1) sync messages for it. But this cost is bounded by 2(P-1) per vertex and does
not grow with H, while the work removed from a single machine grows linearly with H. So a
small bounded synchronization cost buys much better balance on skewed graphs.

## Observations on the scaling run (P=8)

Vertex-partition imbalance rises from 1.10 (H=1000) to 1.27 (H=10000) to 2.13 (H=50000),
while the vertex cut stays at 1.00. `gas_sync_messages` is constant (30738) because the hub
always has 8 replicas and each added synthetic user has exactly one replica. The
`replication_factor` falls as H grows because it is divided by the number of vertices,
which increases by one for every added synthetic user.

## Assumptions

- `replication_factor` = sum of replica counts / number of vertices in the HUB-GRAPH
  (2625 + 1 + H).
- `hub_owner_work` = total vertex_partition_work of the worker that owns the hub vertex.
- Ties for the maximum degree are broken by the smallest id.
- PageRank runs exactly the requested number of iterations, with no early stopping, and
  values are written at full double precision.
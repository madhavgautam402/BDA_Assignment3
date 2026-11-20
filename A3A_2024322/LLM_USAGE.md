# LLM Usage: Assignment 3A

**Tool used:** Claude (Anthropic)
# LLM Usage: Assignment 3A

**LLM used:** Claude (Anthropic), chat interface.

## What I used it for
- Reading the assignment and planning the folder layout and setup steps.
- Generating first drafts of `graph_stats.py`, `partition_messages.py`, `pagerank_local.py`,
  `pregel_pagerank.py`, `powergraph_sim.py` and `common.py`.
- Debugging my Windows and Spark setup: Java 17 vs Java 21, `PYSPARK_PYTHON`,
  `winutils.exe` / `HADOOP_HOME`, and the missing `typing_extensions` dependency of
  graphframes-py 0.12.3.
- Drafting the README answers and this file, which I then reviewed and edited.

## Prompts that materially affected the implementation
1. "Help me do this assignment,tell
   what folder to create... first check all required tools installed or not" (with the
   assignment .docx attached). This produced the folder layout, venv steps and the first
   scripts (graph_stats, partition_messages, pagerank_local).
2. I pasted my terminal output of the tool checks. This revealed Python 3.12 and Java 17
   instead of 3.14.10 and 21, so I installed the correct versions and recreated the venv.
3. I pasted the spark-submit traceback (`No module named 'typing_extensions'`), then the
   `HADOOP_HOME and hadoop.home.dir are unset` error. These led to installing
   typing_extensions and winutils.exe/hadoop.dll.
4. I pasted the "Missing Python executable 'python3'... Failed to find Spark jars directory"
   error, which led to the env.ps1 script setting JAVA_HOME and PYSPARK_PYTHON.
5. "Before going to 3B help me to be ready to submit the zip", which led to a re-check of the
   README/environment requirements.

## Example where the LLM suggestion was incomplete
The first setup instructions and the Pregel script did not mention that on Windows Spark needs
winutils.exe, that PYSPARK_PYTHON and Java 21 must be set in every new terminal, or that
graphframes-py 0.12.3 imports `typing_extensions` without declaring it. The script was also
written without being run against the real GraphFrames JAR.
**How I detected it:** spark-submit failed on my machine with the errors above.
**What I changed:** installed typing_extensions and winutils, created env.ps1, and documented
all three in the README.

## What I understood
With owner(v,P) = v mod P, a directed edge is local only when both endpoints land on the
same worker, so as P grows almost every message becomes remote; that is why the remote
fraction rises from about 75% (P=4) to about 97% (P=32). In PageRank each vertex sends
rank/out_degree along every out-edge and sets its new rank to (1-d)/N plus d times the sum
of incoming messages. In the GraphFrames Pregel program these map to the `rank` vertex
column, `sendMsgToDst`, `aggMsgs` and the update expression of `withVertexColumn`, with
`setMaxIter` controlling the number of supersteps. In the hub experiment, the worker that
owns the hub has to process all of the hub's edges, so its load grows with H, while the
vertex cut spreads those edges evenly and only pays a bounded synchronization cost for the
hub's replicas.

## Changes and decisions I made
- I decided to compute the Pregel reference inside `pregel_pagerank.py`, so the comparison
  stays valid when the number of iterations is changed.
- I installed `typing_extensions` and the Windows `winutils` binaries to get Spark running
  on my machine; `requirements.txt` keeps exactly the six pinned packages.
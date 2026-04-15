# Milestone Report

---

## Updated Schedule

| Date | Tasks |
|---|---|
| **Week 4 Mid (Apr 15)** | Complete naive parallel implementation; study (V1–V1.5) results in response to varying graph topology [Sam]; determine load imbalance, memory access patterns, synchronization stalls for (V1–V1.5) [Elizabeth]; study (V1–V1.5) on RTX 2080s [Sam]; confirm access to PSC machines [Elizabeth] |
| **Week 4 End (Apr 19)** | Optimize memory layout / access pattern [Elizabeth, Sam]; study performance on different GPUs — V100s vs. RTX 2080s [Elizabeth, Sam] |
| **Week 5 Mid (Apr 22)** | Implement strided [Elizabeth] and sequential [Sam] work assignment patterns; optimize kernel launches — coalesce kernel launches per column / diagonal into a single kernel launch [Elizabeth] |
| **Week 5 End (Apr 26)** | Compressed integer storage [Sam]; implement heterogeneous computing pipeline [Sam]; begin draft of final project report [Elizabeth, Sam]; benchmark on real-world datasets [Elizabeth, Sam] |
| **Week 6 Mid (Apr 29)** | Replicate studies from Week 4 on V2 [Elizabeth]; finalize final project report [Elizabeth, Sam]; design and complete poster [Elizabeth, Sam] |
| **Week 6 End (May 1)** | Finalize and submit poster [Elizabeth, Sam] |

---

## Progress So Far

We have made solid foundational progress on our parallel POA (Partial Order Alignment) implementation. On the data side, we conducted a literature review to identify appropriate benchmarks and compiled both real-world and synthetic datasets for evaluation. We also set up visualization tooling to analyze graph structures. On the implementation side, we built a sequential version of the POA algorithm in both Python and C++, supporting graph loading and alignment with affine gap penalties. We have also implemented an initial CUDA kernel to parallelize the alignment computation.

---

## Preliminary Results

### Naive parallel aligner design

We have designed and implemented a naive parallel partial-order alignment algorithm. We launch a separate kernel for each anti-diagonal path in the dynamic programming matrix. Computation across an anti-diagonal is independent; therefore, we assign each thread in the CUDA block grid to compute the cost for a single cell within the current anti-diagonal. The pseudocode below illustrates this method.

```cuda
__global__ alignerKernel(int diagonal) {
    // anti-diagonal offset -> start top right
    int row = threadIdx.x;
    int col = diagonal - threadIdx.x;

    // handle bounds check
    if (out_of_bounds(row, col)) return;

    // extract bases to align from graph (row) and read (col)
    auto current_node = node_from_DP_row(row);
    auto read = col;

    // update thread's cell by iterating over all predecessors
    auto preds = get_predecessors(current_node);
    for (auto &pred : preds) populateDPMatrix(read, current_node, pred);
}

void align() {
    // ... move data to device
    int blockDimX = (bases_in_read + threadsPerBlock - 1) / threadsPerBlock;
    for (int diagonal = 0; diagonal < bases_in_read; diagonal++) {
        alignerKernel<<<blockDimX, threadsPerBlock>>>(diagonal);
    }
    // ... free device data
}
```

### Infrastructure

We have performed a literature review and compiled baseline datasets and POA implementations to use as benchmarks. We collected open-source datasets of genome assemblies parsed into pangenome graphs as well as query sequences (reads) for alignment evaluation. We installed and tested `vg`, which enables pangenome graph processing and POA alignment, and will benchmark against it for both runtime and correctness.

This process required identifying a file format to store pangenome graphs and reads that is both easy to interpret and widely used. After considering various alternatives, we selected Graphical Fragment Assembly (`.gfa`) and `.vg` for graphs, and `.fasta` for reads.

<img width="1034" height="440" alt="IMG_6212" src="https://github.com/user-attachments/assets/35f74aed-9a4b-4c9f-87b1-3c062ec2728c" />
*Figure 1: Visualization of a pangenome graph constructed from assemblies of 90 individuals of chromosome Y, produced with `vg`.*

### Graph topology

We developed a synthetic set of graphs and reads to debug our sequential aligner and study the impact of graph topology on alignment in a controlled fashion.

<img width="708" height="443" alt="image" src="https://github.com/user-attachments/assets/d025e5e5-4dda-4ac8-9900-3742cf6529fd" />

*Figure 2: Impact of graph topology on sequential runtime.*

As the figure highlights, the maximal sequence length per graph node and read length have the largest impact on sequential runtime. Notably, latency scales linearly with respect to both variables: runtime roughly doubles when either read length or maximal node sequence length doubles. Runtime also scales roughly linearly with the number of nodes in the graph — expected, as more nodes introduce more predecessor iterations per alignment step.

---

## Comparison to Goals

We are slightly behind the original schedule, primarily because dataset compilation took longer than anticipated and we have not yet been able to thoroughly characterize our parallel implementation's performance. That said, we remain confident we can meet our core deliverables: a working naive parallel implementation and at least one round of optimization.

**Original deliverables.** The main risk is optimizing across varied graph topologies — after working with the real-world datasets, we found they are substantially larger than expected, making processing and visualization more difficult than anticipated.

**"Nice to have" deliverables.** Our current assessment is as follows. Strided and sequential vectorization schemes for intra-sequence parallelism appear feasible; work assignment in the CUDA kernel is not the bottleneck — analysis is. A topological sort-informed alignment buffer remains vague in specification and may be cut. Compressed integer storage is straightforward to implement but unlikely to change asymptotic scaling behavior, so it may be deprioritized if time is short. Multi-GPU parallelization is contingent on available libraries for GPU-to-GPU scheduling, which we have not yet had time to evaluate. A heterogeneous CPU/GPU pipeline (short reads on CPU via OpenMP, long reads on GPU) is feasible and mostly depends on having solid results to analyze.

---

## Issues and Open Questions

Our two main blockers have been:

1. **Locating suitable datasets and reference implementations.** This required substantially more reading than expected, as there are few complete open-source benchmarks in this space.

2. **Developing a correct sequence-graph data structure.** Within the literature, notation and representations for pangenome graphs are inconsistent, ranging from de Bruijn graphs to DAGs to bidirectional DAGs. To match our baselines, we implemented a bidirectional DAG (sequence graph) representation, which introduced additional complexity in cycle detection.

Going forward, our primary concern is analysis rather than implementation. The POA algorithm itself is not algorithmically complex to parallelize, but characterizing its scaling behavior is nontrivial: the data structures are large and exhibit non-local memory access patterns, which makes profiling and optimization more involved. At this stage, the main challenge is less "what to implement" and more moving fast enough to generate and interpret results in time.

---

## Updated Goals

By the poster session, we will:

- Demonstrate speedup over the sequential baseline on real-world datasets
- Test and compare at least one work assignment strategy
- Report scaling behavior in a multi-processor context (multi-GPU or heterogeneous CPU/GPU)
- Present results as speedup and performance graphs

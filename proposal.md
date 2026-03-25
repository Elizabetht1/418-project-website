[PDF Project proposal](https://docs.google.com/document/d/1Z6tSwpX70g-vOBcn3sbAG257njpDDJL0FixXyypOGYM/edit?usp=sharing)

---

## Summary

We will implement a parallel Sequence-to-Graph aligner that maps long DNA reads (>1,000 characters) to pangenome variation graphs. We will evaluate performance on open-source pangenome datasets on the GHC lab machines with NVIDIA RTX 2080 GPUs, comparing performance against a sequential CPU baseline and finding the crossover point where GPU execution becomes favourable.

---

## Background

In modern genomics, sequencing machines produce short DNA fragments called reads, which are mapped to a reference genome to identify genetic variations. The standard linear human reference genome (GRCh38) doesn't capture the full genetic diversity of a population and is systematically biased against those whose ancestry is not represented in the reference. The pangenome graph instead captures an entire population's genetic diversity, but it contains billions of characters and millions of nodes. Mapping the queried read to this graph is computationally expensive, inviting acceleration through parallelism.

Two key algorithms serve as baselines in this field: Smith-Waterman (SW) and Partial Order Alignment (POA). SW is a Dynamic Programming algorithm for aligning a reference and target sequence. POA extends this to aligning a single reference to a graph. For a read of length M aligned against a graph node with a sequence of length L, the DP submatrix has dimensions M × L and follows the recurrence in Fig 1. For a graph with total sequence length N, this naive DP has cost O(NM), serving as the bottleneck to the alignment problem.

```
DP[i][j] = max(
    0,                            // local: can start fresh anywhere
    DP[i-1][j-1] + score(i,j),   // diagonal: match or mismatch
    DP[i-1][j]   - gap,           // up: gap in reference
    DP[i][j-1]   - gap            // left: gap in query
)
```
*Fig. 1: Smith-Waterman Recurrence with scope for anti-diagonal parallelism*

Literature has already demonstrated significant opportunities for parallelism:

- **(a) Intra-sequence:** make updates to the dynamic programming matrix in parallel, respecting the topological ordering of the graph.
- **(b) Inter-sequence:** when aligning multiple reads simultaneously. This has obvious parallelism — similar to a parallel for loop.

There is some literature implementing inter-sequence parallelism, but intra-sequence parallelism is less explored. We will be implementing both of these. The CPU will topologically sort the graph and launch CUDA kernels, and each thread block will correspond to a graph node. The GPU threads will collaborate on populating the dynamic matrix.

---

## The Challenge

### Workload

**Dependencies.** There are two levels of parallelism. Within a graph node, we can get the alignment of a query and the sequence represented by that node by processing the anti-diagonal DP matrix concurrently. The dependencies within the SW and POA alignment algorithms are well documented: the cost of a single pair of nodes depends on the cost of its left middle, upper middle, and upper left neighbors.

Across nodes, we can process nodes that don't depend on each other concurrently, a result we'll dynamically get from the topological sort of the graph on the CPU. Optimizing the parallelism across both levels presents interesting challenges.

**Memory Access Characteristics.** For intra-sequence parallelism, we intend to map each graph node to a thread block. However, given the size of the problem, even the DP matrix per-graph node won't fit in the shared memory of the GPU. This forces tiling: the submatrix must be processed in chunks and spilled to global memory, and the global memory accesses will need to be coalesced. This would require detailed consideration of how the columns are laid out in global and shared memory.

**Divergent Execution.** Pangenomic graphs have highly non-uniform node lengths. Thread blocks processing shorter nodes will finish faster than blocks processing long nodes. Further, the merge nodes with multiple predecessors will have to combine information from varying numbers of predecessors, which can cause warp divergence. Managing load imbalance poses an interesting challenge.

**Communication-to-Computation Ratio.** Inter-sequence should have high arithmetic intensity, as data does not need to be communicated. Intra-sequence has a higher communication-to-computation ratio — each cell needs 3 grid points communicated to it.

### Constraints

- The GPU memory capacity directly caps the amount of parallelism we can achieve in the inter-sequence approach.
- The problem distribution creates load-balancing issues as different reads can have different lengths for inter-sequence execution, and different graph nodes will have sequences of varying lengths for intra-sequence execution.
- Our performance is also highly dependent on graph topology. We expect performance to degrade with significant branching in the pangenome graph (indicating multiple alleles) and with longer reads.

---

## Resources

### Algorithm Definition

The Smith-Waterman algorithm is presented in the following survey by Xia et al.:

> Xia, Zeyu, et al. "A Review of Parallel Implementations for the Smith–Waterman Algorithm." *Interdisciplinary Sciences, Computational Life Sciences*, vol. 14, no. 1, 2022, pp. 1–14. [doi:10.1007/s12539-021-00473-0](https://doi.org/10.1007/s12539-021-00473-0)

The Partial Order Alignment algorithm was articulated by Lee et al.:

> Christopher Lee, Catherine Grasso, Mark F. Sharlow. "Multiple sequence alignment using partial order graphs." *Bioinformatics*, Volume 18, Issue 3, March 2002, Pages 452–464. [doi:10.1093/bioinformatics/18.3.452](https://doi.org/10.1093/bioinformatics/18.3.452)

### Starter Code

We will implement the algorithms from scratch. The algorithm structure is well documented and the implementation is not excessively complicated. Further, implementing the algorithms ourselves will help us understand the implementation, allowing us to more easily integrate optimizations.

### Data

We will use open source datasets to evaluate on. The Human Pangenome Reference Consortium provides references and constructed pangenomes: [humanpangenome.org/data](https://humanpangenome.org/data/).

We also plan to generate synthetic datasets for debugging and ablations using SimLoRD:

> Bianca K. Stöcker, Johannes Köster, Sven Rahmann. "SimLoRD: Simulation of Long Read Data." *Bioinformatics*, Volume 32, Issue 17, September 2016, Pages 2704–2706. [doi:10.1093/bioinformatics/btw286](https://doi.org/10.1093/bioinformatics/btw286)

### Compute Platform

We will evaluate our implementation on two compute platforms.

**Gates-Hillman cluster (primary):**
- Intel Core i7-9700 CPU @ 3.00GHz
- NVIDIA GeForce RTX 2080

The RTX 2080 is a fairly outdated GPU model, serving as a defensible representative of an accessible compute resource.

**Pittsburgh Supercomputing Cluster:** We wish to access the GPU partition of the cluster, enabling us to evaluate on V100 GPUs, a higher-end model than the RTX 2080. V100 GPUs have higher memory bandwidth and size compared to RTX 2080s. Comparing performance on the two clusters will enable us to develop an implementation that scales well in response to increased memory resources.

---

## Goals and Deliverables

### Performance Targets

Performance targets will depend on our workload — we expect different behavior based on the length of the reads and the topology of the reference graph. With this in mind, once we have compiled benchmarks, we will study their interplay with our hardware in order to develop a more concrete target.

We will also use external baselines to inform performance targets. Within the literature, Giga Cell Updates Per Second (GCUPS) is a common performance metric, calculated as:

```
GCUPS = (|V| × |Q|) / (t × 10⁹)
```

where t is the total alignment time in seconds, V is the number of graph vertices, and Q is the total number of read bases. Feng & Luo ([doi:10.1145/3472456.3472505](https://dl.acm.org/doi/10.1145/3472456.3472505)) implement a GPU-accelerated aligner on an RTX 2080, achieving >20 GCUPS on a diverse dataset with linear scaling as thread blocks and threads per block increase. We will target the same performance.
 
Moreover, our algorithm is exact, so we will expect the alignment score outputted by our algorithm to be equivalent to reference CPU-only baselines such as VG.

### Plan to Achieve

- Sequential POA implementation for sequence-to-graph alignment
- CPU/GPU accelerated inter-sequence parallelism for S2G
- GPU-accelerated intra-sequence POA algorithm using anti-diagonal layout for vectorization
- Assess GPU-accelerated POA algorithms' memory access patterns and load imbalance
- Implement tailored optimizations in response to profiling results
- Profiles of scaling behavior of GPU-accelerated POA algorithms based on graph topology
- Profiles of scaling behavior of GPU-accelerated POA algorithms based on GPU characteristics (RTX 2080 to V100)

### Hope to Achieve

- Implementation of strided and sequential vectorization schemes for intra-sequence parallelism
- Topological sort-informed alignment buffer
- Compressed integer storage
- Multi-GPU parallelization for POA
- A heterogeneous computing pipeline that leverages CPU for short reads and GPU for long reads

---

## Platform Choice

**Device choice:** GPU is the right choice for S2G graph alignment since the core computational bottleneck of this problem is populating a DP matrix, which is fundamentally an arithmetic problem that operates on a lot of data points but follows simple instructions. The memory bandwidth available on a GPU is much larger than that of a CPU, and since our workload is wide and memory bandwidth-bound, it is ideal for a GPU. Further, our experience with the CUDA programming model makes it ideal for this use case. The hardware used in our reference papers is also mostly NVIDIA GPUs.

**Language choice:** We will use C++ since most reference implementations are in C++, and CUDA is designed to work well with C/C++.

---

## Schedule

| Week | Dates | Tasks |
|---|---|---|
| 1 | Mar 23 | Compile simulated and real-world benchmarks and classify based on graph characteristics. Validate pipeline to perform I/O on graph and sequence data. |
| 2 | Mar 30 | Implement Sequential POA Algorithm (V0). Define performance targets for GPU acceleration. Implement anti-diagonal vectorization for sequential POA (V1). |
| 3 | Apr 6 | Implement inter-sequence parallelism based on topological sort to extract dependencies (V1.5). Determine load imbalance, memory access patterns, synchronization stalls for V1–V1.5. Study results in response to varying graph topology. Study results on V100s vs. RTX 2080s. Draft Project Milestone Report. |
| 4 | Apr 13 | Finalize Project Milestone Report. Brainstorm optimizations to address deficiencies in V1–V1.5. Begin implementing V2. |
| 5 | Apr 20 | Complete V2. Replicate studies completed in Week 3 on V2. Draft of final project report. |
| 6 | Apr 27 | Finalize final project report. Design and complete poster. |
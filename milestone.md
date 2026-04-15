---
## Updated Schedule 

Week 4 Mid Week [April 15th]  
Complete Naive Parallel Implementation 
Study (V1-1.5) results in response to varying graph topology [Sam]
Determine load imbalance, memory access patterns, synchronization stalls for (V1-1.5) [Elizabeth]
Study (V1-1.5) RTX 2080s [Sam]
Make sure we have access to PSC machines [Elizabeth]
Week 4 End of Week [April 19 th]  
Optimize memory layout / access pattern [Elizabeth, Sam] 
Study performance on different GPUS (V100s vs. RTX 2080s) [Elizabeth, Sam] 
Week 5 Mid Week 
[April 22nd] 
Implement strided [Elizabeth] and sequential [Sam] work assignment patterns 
Optimize kernel launches – coalesce kernel launches per column / diagonal into a single kernel launch [Elizabeth]
Week 5 End of week [April 26th] 
Compressed integer storage [Sam]
Implement heterogeneous computing pipeline –  [Sam]
Begin draft of  final project report [Elizabeth, Sam]
Benchmark on real world datasets 
Week 6 Mid Week [April 29th]
Replicate studies completed in Week 4 on (V2) [Elizabeth]
Finalize final project report [Elizabeth, Sam]
Design and complete poster [Elizabeth, Sam]
Week 6 End of Week [May 1st] 
Finalize poster design and complete [Elizabeth, Sam]

---
## Progress so Far 

We have made solid foundational progress on our parallel POA (Partial Order Alignment) implementation. On the data side, we conducted a literature review to identify appropriate benchmarks, and compiled both real-world and synthetic datasets for evaluation. We also set up visualization tooling to analyze graph structures. On the implementation side, we built a sequential version of the POA algorithm in both Python and C++, supporting graph loading and alignment with affine gap penalties. We have also implemented an initial CUDA kernel to parallelize the alignment computation.
Preliminary Results
Naive Parallel Aligner Design 
We have designed and implemented a naive parallel partial-order alignment algorithm. We launch a separate kernel for each anti-diagonal path in the dynamic programming matrix. Computation across an antidiagonal is independent; therefore, we assign each thread in the CUDA block grid to compute the cost for a single cell within the current anti diagonal. The figure below illustrates the pseudocode for this method. 
```
__global__ alginerKernel(int diagonal) {
// antidiaognal offset -> start top right 
	int row = threadIDx.x; 
	int col = diagonal - threadIDx.x; 
	
	// handle bounds check 
	if (out_of_bounds(row,col)) return;

// Extract bases to align from graph (row) and read (col)
	auto current_node = node_from_DP_row(row);
	auto read = col; 
	
	// Update thread's cell by iterating over all predecessors 
	auto preds = get_predecessors(current_node);
	for (auto &pred : preds) populateDPMatrix(read, current_node, pred); 

}

void align() {
	// ... move data to device 
	int blockDimX = (bases_in_read + threadsPerBlock - 1) / threadsPerBlock;
for (int diagonal = 0; diagonal < bases_in_read; diagonal ++) {
    alignerKernel<<<blockDimX,threadsPerBlock>>>(diagonal);
}
	// ... free device data
}
```

### Infrastructure 
We have performed a literature review and compiled baseline datasets and POA implementations to utilize as benchmarks. We have collected open source datasets of genome assemblies that have been parsed into pangenome graphs as well as query sequences (reads) to evaluate alignment. We installed and tested vg, which enables pangenome graph processing and POA alignment. We will benchmark against vg for runtime and correctness.  

Figure 1: Visualization of Pangenome graph constructed from assemblies of 90 individuals of chromosome Y produced with vg
The results above required identifying a file format to store pangenome graphs and reads that was easy to interpret and widely used. After considering various alternatives, we choose graphical fragment assembly and .vg to represent graphs, and .fasta to represent reads.  

Graph Topology 
We have developed a synthetic set of graphs and reads. We used this collection to debug our sequential aligner and study the impact of graph topology on alignment in a controlled fashion. 

Figure 2: Impact of Graph Topology on Sequential Runtime 
As the figure highlights, the maximal length of the sequence represented by each node in the graph (upper right) and read length (upper left) have the largest impact on sequential runtime. Notably, latency scales linearly with respect to these two variables: the milliseconds it takes for the aligner to produce a solution roughly doubles when read length or maximal sequence length doubles. 
	The runtime also scales roughly linearly with the number of nodes in the graph. This result is expected, as more nodes introduce more instances in which the aligner must iterate over predecessors. 


### Comparison to Goals 
We are slightly behind the original schedule, primarily because dataset compilation took longer than anticipated and we have not yet been able to thoroughly characterize our parallel implementation's performance. That said, we remain confident we can meet our core deliverables: a working naive parallel implementation and at least one round of optimization.
For our original deliverables, the main risk is optimizing across varied graph topologies — after working with the datasets, we found they are substantially larger than expected, making processing and visualization more difficult than anticipated.
For our "nice to have" deliverables, our current assessment is as follows. Strided and sequential vectorization schemes for intra-sequence parallelism appear feasible; work assignment in the CUDA kernel is not the bottleneck — analysis is. A topological sort-informed alignment buffer remains vague in specification and may be cut. Compressed integer storage is straightforward to implement but unlikely to change asymptotic scaling behavior, so it may be deprioritized if time is short. Multi-GPU parallelization is contingent on available libraries for GPU-to-GPU scheduling, which we haven't had time to evaluate yet. A heterogeneous CPU/GPU pipeline (short reads on CPU via OpenMP, long reads on GPU) is feasible and mostly depends on having solid results to analyze.
Main Bottlenecks  
Our two main blockers have been:
(1) locating suitable datasets and reference implementations. This required substantially more reading than expected as there are few complete open source benchmarks. 
(2) Developing a correct sequence graph data structure. Within the literature, the notation and representations used for pangenome graphs are inconsistent, ranging from De Bruijin graphs, to DAGs, to bidirectional DAGs. To match baselines, we implemented a bidirectional DAG (sequence graph) representation. This introduced additional complexity in cycle detection. 
Going forward, our primary concern is analysis rather than implementation. The POA algorithm itself is not algorithmically complex to parallelize, but characterizing its scaling behavior is nontrivial: the data structures are large and exhibit non-local memory access patterns, which makes profiling and optimization more involved. At this stage, the main challenge is less "what to implement" and more "moving fast enough to generate and interpret results in time."
### Updated Goals 
By the poster session, we will:
Demonstrate speedup over sequential baseline on real-world datasets
Test and compare at least one work assignment strategy
Report scaling behavior in a multi-processor context (multi-GPU or heterogeneous CPU/GPU)
Present results as speedup and performance graphs

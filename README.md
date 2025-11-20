# wgpu_parallel_threading_computing
a set of learning labs demonstrating parallel reductions on the GPU using WGPU and WGSL. Lab 1 shows a workgroup-shared-memory reduction (64 threads per workgroup) and an optimized variant (128 values). It includes host Rust code that dispatches compute shaders, reads back partial sums, verifies correctness, and measures GPU/roundtrip timings.

# **Week 4 — Real Compute Kernels & Performance Engineering**
---

## **Learning Goals**

By the end of Week 4, you should be able to:

* Implement non-trivial GPU kernels used in real workloads
* Apply **tiling and shared memory** to structured computations
* Distinguish between **compute-bound** and **memory-bound** kernels in practice
* Benchmark GPU kernels against optimized CPU and PyTorch baselines
* Reason about performance bottlenecks using measured data

This week prepares you for:

* Triton (Week 5)
* The final mini-project (Week 6)

---

## 📘 **Required Resources**

### **1. CUDA Programming Guide — Performance & Execution**

Review relevant sections from:

* **Chapter 6 — Performance Guidelines**
* **Chapter 7 — Execution Configuration**

📄
[https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)

Focus on:

* Occupancy (conceptual, not formula-heavy)
* Instruction throughput vs memory throughput
* When more threads stop helping

---

### **2. Matrix Multiplication on GPUs (Canonical Pattern)**

Read **any one** of the following:

* NVIDIA Blog — *An Efficient Matrix Transpose in CUDA C/C++*
  [https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

* CUDA Sample: `matrixMul` (inspect structure only)
  [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

Focus on:

* Tiling
* Shared memory reuse
* Thread cooperation

---

### **3. Real-World Kernel Examples**

You are not expected to understand every line.

* **tiny-cuda-nn (NVIDIA)**
  [https://github.com/NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

* **FlashAttention (Dao et al.)**
  [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

Look for:

* How kernels are structured
* How memory reuse is emphasized
* How complexity is managed

---

## 🧠 **Concepts Covered This Week**

* Structured parallelism (tiles, blocks, subproblems)
* Shared memory reuse across threads
* Compute intensity vs memory traffic
* Kernel fusion (conceptual)
* Benchmarking methodology
* Performance vs correctness trade-offs

---

## **What You Will Build This Week**

You will implement **at least one real compute kernel** that resembles a real research or ML workload, such as:

* Tiled matrix multiplication (GEMM)
* Softmax 
* Block-wise reduction or normalization

You will then:

* Benchmark it on GPU
* Compare against CPU and/or PyTorch
* Analyze where the speedup comes from

---

## **Week 4 Assignment (Summary)**

### **Task 1 — Implement a Real Compute Kernel**

Choose **one** kernel:

* Matrix multiplication 
* Softmax
* Reduction / normalization

Requirements:

* CUDA implementation
* Correctness verification
* Clear kernel structure

---

### **Task 2 — Shared Memory Optimization**

* Start with a naïve global-memory version
* Implement a shared-memory optimized version
* Use synchronization correctly
* Explain the optimization strategy

---

### **Task 3 — Benchmarking & Analysis**

* Compare:

  * GPU naive
  * GPU optimized
  * CPU or PyTorch baseline
* Report:

  * Execution time
  * Speedup
* Brief analysis of bottlenecks

## 📁 **Submission Folder**

```
week4/
 ├── assignment4.pdf
 ├── kernel_naive.cu
 ├── kernel_optimized.cu
 ├── correctness_check.py
```

---

## **Optional but Highly Recommended**

* CUDA Best Practices Guide - Performance Section
* Nsight Compute (link shared last week)
* NVIDIA blogs on GEMM and tiling strategies

---




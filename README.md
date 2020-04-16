Naive implementation of mt_rand using pyopencl and pycuda

Installation
============

Numpy and one of Pycuda or pyopencl should me installed to run this.

```
easy_install numpy pyopencl
```
OR 
```
pip install numpy pyopencl<br>
```

For pycuda you should run

```
easy_install numpy pycuda<br>
```
OR 
```
pip install numpy pycuda<br>
```

For any opencl compatible device you should also install opencl package for your device (AMD/NVIDIA/Intel/etc.)

For cuda - you should install nvcc compiler with headers

Execution
=========

One can run this implementation with start_ocl.sh and start_cuda.sh. This scripts will also produce profiling information of execution

PyOpencl implementation will try to run on first device it would found. I don't know for now how to fix it to run only on GPU.
If you would run it from shell - there would be some info regarding this

Speed tests
===========

```bash
$ grep SIZE config.py
SIZE = 16384
$ ./start_ocl.sh
$ tail -n 2 opencl_profile_0.log
method=[ mt_brute ] gputime=[ 72.640 ] cputime=[ 4.000 ] occupancy=[ 0.667 ] 
method=[ memcpyDtoHasync ] gputime=[ 12.512 ] cputime=[ 8.000 ] 

$ ./start_cuda.sh
$ tail -n 2 cuda_profile_0.log
method=[ mt_brute ] gputime=[ 68.928 ] cputime=[ 6.000 ] occupancy=[ 0.667 ]
method=[ memcpyDtoH ] gputime=[ 10.720 ] cputime=[ 88.000 ]
```

Results
=======

72.640+12.51 microseconds for 2^14 hashes = 1.9 * 10^8 seeds

OR

22 seconds for all 2^32 combinations

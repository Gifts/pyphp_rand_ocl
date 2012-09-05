Naive implementation of mt_rand using pyopencl and pycuda

Installation
============

Numpy and one of Pycuda or pyopencl should me installed to run this.

<code>
easy_install numpy pyopencl<br>
OR <br>
pip install numpy pyopencl<br>
</code>

For pycuda you should run

<code>
easy_install numpy pycuda<br>
OR <br>
pip install numpy pycuda<br>
</code>

For any opencl compatible device you should also install opencl package for your device (AMD/NVIDIA/Intel/etc.)

For cuda - you should install nvcc compiler with headers

Execution
=========

One can run this implementation with start_ocl.sh and start_cuda.sh. This scripts will also produce profiling information of execution

PyOpencl implementation will try to run on first device it would found. I don't know for now how to fix it to run only on GPU.
If you would run it from shell - there would be some info regarding this

Speed tests
===========

<code>$ ./start_ocl.sh<br>
$ tail -n 2 opencl_profile_0.log<br>
method=[ mt_brute ] gputime=[ 28462.783 ] cputime=[ 5.000 ] occupancy=[ 0.833 ]<br>
method=[ memcpyDtoHasync ] gputime=[ 2061.664 ] cputime=[ 34.000 ]<br>
<br>
$ ./start_cuda.sh<br>
$ tail -n 2 cuda_profile_0.log<br>
method=[ mt_brute ] gputime=[ 28348.801 ] cputime=[ 5.000 ] occupancy=[ 1.000 ] <br>
method=[ memcpyDtoHasync ] gputime=[ 1364.960 ] cputime=[ 1503.000 ] <br>
</code>

28348.801+1364.960 microseconds for 2^18 hashes = 8.8 * 10^6 seeds
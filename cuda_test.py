#coding: utf8

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv

import numpy as np
import time

from shared_gpu_kernels import gen_kernel, transform_to_cuda

from config import SIGNIFICANT_LENGTH, SIZE, MT_N, M, STATE_SIZE


MT_state_result = np.zeros((SIGNIFICANT_LENGTH, SIZE)).astype(np.uint32)

Stream = drv.Stream()
Stream2 = drv.Stream()

MT_state_buf = drv.mem_alloc(SIZE * MT_N * 4)
MT_state_res_buf = drv.mem_alloc(MT_state_result.nbytes)

prg = SourceModule(
    transform_to_cuda(
        gen_kernel(MT_N, STATE_SIZE, M, SIZE, SIGNIFICANT_LENGTH)
    )
)
prog = prg.get_function('mt_brute')

zzz = time.time()

ev = prog(np.uint32(0), MT_state_buf, MT_state_res_buf, block=(STATE_SIZE, 1, 1), grid=(SIZE/STATE_SIZE, 1), stream=Stream)
drv.memcpy_dtoh_async(MT_state_result, MT_state_res_buf, stream=Stream2)

for i in xrange(10):
    prog(np.uint32(i*SIZE), MT_state_buf, MT_state_res_buf, block=(STATE_SIZE, 1, 1), grid=(SIZE/STATE_SIZE, 1), stream=Stream)
    drv.memcpy_dtoh_async(MT_state_result, MT_state_res_buf, stream=Stream2)





print MT_state_result

zzz = time.time() - zzz
print zzz

for row in MT_state_result:
    print row[0]



MT_state_buf.free()
MT_state_res_buf.free()

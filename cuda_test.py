#coding: utf8

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv

import numpy as np
import time

from shared_gpu_kernels import gen_kernel, transform_to_cuda

SIZE = (2**18)
DIVIDER = 1
STATE_SIZE = (2**8)
SIGNIFICANT_LENGTH = 8

## PHP rand constants
MT_N = 624
M = 397
## END PHP rand constants

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
#instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(0), MT_state_buf, MT_state_res_buf)#, g_times_l=True)
#data_event = cl.enqueue_copy(queue_instruction, MT_state_result, MT_state_res_buf, wait_for=[instr_event,])
#
#for i in xrange(10):
#    instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(i*SIZE*0), MT_state_buf, MT_state_res_buf, wait_for=[data_event,])#, g_times_l=True)
#    data_event = cl.enqueue_copy(queue_instruction, MT_state_result, MT_state_res_buf, wait_for=[instr_event,])
#
#
#z2 = cl.enqueue_marker(queue_instruction)
#z2.wait()

ev = prog(np.uint32(0), MT_state_buf, MT_state_res_buf, block=(STATE_SIZE, 1, 1), grid=(SIZE/STATE_SIZE, 1), stream=Stream)
drv.memcpy_dtoh_async(MT_state_result, MT_state_res_buf, stream=Stream2)

for i in xrange(10):
    prog(np.uint32(i*SIZE), MT_state_buf, MT_state_res_buf, block=(STATE_SIZE, 1, 1), grid=(SIZE/STATE_SIZE, 1), stream=Stream)
    drv.memcpy_dtoh_async(MT_state_result, MT_state_res_buf, stream=Stream2)





print MT_state_result

zzz = time.time() - zzz
print zzz
#z2 = prg.sum(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(0), MT_state_buf)#, g_times_l=True)
#cl.enqueue_copy(queue_instruction, MT_state_result, MT_state_buf).wait()


#print MT_state_result
for i in xrange(200):
    print MT_state_result[0][i],
print
for row in MT_state_result:
    print row[0]

#print "Start: {0} End: {1} Difference: {2}".format(z.profile.start,z2.profile.end, z2.profile.end - z.profile.start)


MT_state_buf.free()
MT_state_res_buf.free()

#CLEAN UP

#queue_instruction.flush()
#MT_state_buf.release()

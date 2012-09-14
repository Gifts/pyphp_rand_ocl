#coding: utf8

import pyopencl as cl

import numpy as np
import time
from shared_gpu_kernels import gen_kernel

from config import SIGNIFICANT_LENGTH, SIZE, MT_N, M, STATE_SIZE

MT_state_result = np.zeros((SIGNIFICANT_LENGTH, SIZE)).astype(np.uint32)

ctx = cl.create_some_context()
queue_instruction = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
queue_data = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

MT_state_buf = cl.Buffer(ctx, mf.WRITE_ONLY, SIZE * MT_N * 4)
MT_state_res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, MT_state_result.nbytes)

prg = cl.Program(ctx, gen_kernel(MT_N, STATE_SIZE, M, SIZE, SIGNIFICANT_LENGTH)).build()
z = cl.enqueue_marker(queue_instruction)

zzz = time.time()
instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(0), MT_state_buf, MT_state_res_buf)#, g_times_l=True)
data_event = cl.enqueue_copy(queue_instruction, MT_state_result, MT_state_res_buf, wait_for=[instr_event,])

with open('result.txt', 'ab') as f:
    for i in xrange(20):#2**31 / SIZE):
        instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(i*SIZE), MT_state_buf, MT_state_res_buf, wait_for=[data_event,])#, g_times_l=True)
        data_event = cl.enqueue_copy(queue_instruction, MT_state_result, MT_state_res_buf, wait_for=[instr_event,])
        data_event.wait()
        #for row in (tmp for tmp in MT_state_result[0]):
        #    f.write('{0}\n'.format(row))


z2 = cl.enqueue_marker(queue_instruction)
z2.wait()
print '>>>', time.time() - zzz
for row in  MT_state_result:
    print row[0]

print "Start: {0} End: {1} Difference: {2}".format(z.profile.start,z2.profile.end, z2.profile.end - z.profile.start)


#CLEAN UP

queue_instruction.flush()
MT_state_buf.release()

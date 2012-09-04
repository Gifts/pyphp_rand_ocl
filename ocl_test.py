#coding: utf8

import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np
import numpy.linalg as la


SIZE = 102400*2
DIVIDER = 1
STATE_SIZE = 32*16


## PHP rand constants
MT_N = 624
M = 397
## END PHP rand constants

MT_state_result = np.zeros((8, SIZE)).astype(np.uint32)

ctx = cl.create_some_context()
queue_instruction = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
queue_data = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

MT_state_buf = cl.Buffer(ctx, mf.WRITE_ONLY, SIZE * MT_N * 4)
MT_state_res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, MT_state_result.nbytes)

prg = cl.Program(ctx, """
    #define MT_N {0}
    #define STATES_SIZE {1}
    #define M {2}
    #define ALL_SIZE {3}
    """.format(MT_N, STATE_SIZE, M, SIZE,"#define DIVIDED 1" if DIVIDER > 1 else '') +
    """
    #define hiBit(u)      ((u) & 0x80000000U)  /* mask all but highest   bit of u */
    #define loBit(u)      ((u) & 0x00000001U)  /* mask all but lowest    bit of u */
    #define loBits(u)     ((u) & 0x7FFFFFFFU)  /* mask     the highest   bit of u */
    #define mixBits(u, v) (hiBit(u)|loBits(v)) /* move hi bit of u to hi bit of v */

    #define twist(m,u,v)  (m ^ (mixBits(u,v)>>1) ^ ((unsigned int)(-(unsigned int)(loBit(u))) & 0x9908b0dfU))
    """

    """

    __kernel void mt_brute(
        unsigned int seed_start,
        __global unsigned int *c, //c[MT_N][ALL_SIZE], //
        __global unsigned int *res
    )
    {
      __private int gid = get_global_id(0);
      __private int lid = get_local_id(0);

      __local unsigned int state[3*STATES_SIZE];


      /* PHP_MT_VARIABLES*/
      __private unsigned int i;
      __private unsigned int s1;

      //__local unsigned int *next;
      /* END PHP_MT_VARIABLES*/


      /* PHP_MT_INITIALIZE */
      //s = state + lid;
      __private unsigned int s2;
      __private unsigned int r2;

      i = 1;

      //*s++ = (seed_start + gid) & 0xffffffffU;
      s2 = (seed_start + gid) & 0xffffffffU;
      r2 = s2;
      c[gid] = s2;

      for (; i < MT_N; ++i)
      {
        s2 = ( 1812433253U * ( r2 ^ (r2 >> 30) ) + i ) & 0xffffffffU;
        r2 = s2;
        c[ALL_SIZE*i + gid] = s2;
      }
      /* END PHP_MT_INITIALIZE */


      /* PHP_MT_RELOAD */

      state[STATES_SIZE + lid] = c[gid];

      for (i = 0; i < MT_N - M; ++i)
      {
        state[                lid] = state[STATES_SIZE + lid];  // +++ p[0] = c[0][gid]
        state[STATES_SIZE   + lid] = c[(i+1) * ALL_SIZE + gid];
        state[2*STATES_SIZE + lid] = c[(i+M) * ALL_SIZE + gid];
        c[ALL_SIZE*i + gid] = twist(state[2*STATES_SIZE + lid], state[lid], state[STATES_SIZE + lid]);
      }
      r2 = MT_N - M;
      for (i = MT_N - M; i < MT_N-1; ++i)
      {
        state[                lid] = state[STATES_SIZE + lid];  // +++ p[0] = c[0][gid]
        state[STATES_SIZE   + lid] = c[(i+1) * ALL_SIZE + gid];
        state[STATES_SIZE*2 + lid] = c[(i-r2) * ALL_SIZE + gid];
        c[ALL_SIZE*i + gid] = twist(state[2*STATES_SIZE + lid], state[lid], state[STATES_SIZE + lid]);
      }
      state[                lid] = state[STATES_SIZE + lid];  // +++ p[0] = c[0][gid]
      state[STATES_SIZE   + lid] = c[gid];
      state[STATES_SIZE*2 + lid] = c[(i-r2) * ALL_SIZE + gid];
      c[ALL_SIZE*i + gid] = twist(state[2*STATES_SIZE + lid], state[lid], state[STATES_SIZE + lid]);

      /* END PHP_MT_RELOAD */

      /* PHP_MT_RAND */

      for (i = 0; i < 8; ++i)
      {
          s1 = c[ALL_SIZE*i + gid];

          s1 ^= (s1 >> 11);
          s1 ^= (s1 <<  7) & 0x9d2c5680U;
          s1 ^= (s1 << 15) & 0xefc60000U;
          s1 ^= (s1 >> 18);
          res[ALL_SIZE*i + gid] = (long)(s1 >> 1);
      }
      /* END PHP_MT_RAND */

      return;
    }
    """).build()
z = cl.enqueue_marker(queue_instruction)


instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(0), MT_state_buf, MT_state_res_buf)#, g_times_l=True)
data_event = cl.enqueue_copy(queue_data, MT_state_result, MT_state_res_buf, wait_for=[instr_event,])

for i in xrange(10):
    instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(i*SIZE*0), MT_state_buf, MT_state_res_buf, wait_for=[data_event,])#, g_times_l=True)
    data_event = cl.enqueue_copy(queue_data, MT_state_result, MT_state_res_buf, wait_for=[instr_event,])


z2 = cl.enqueue_marker(queue_instruction)
z2.wait()
#z2 = prg.sum(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(0), MT_state_buf)#, g_times_l=True)
#cl.enqueue_copy(queue_instruction, MT_state_result, MT_state_buf).wait()


#print MT_state_result
for row in  MT_state_result:
    print row[0]

print "Start: {0} End: {1} Difference: {2}".format(z.profile.start,z2.profile.end, z2.profile.end - z.profile.start)


#CLEAN UP

queue_instruction.flush()
MT_state_buf.release()

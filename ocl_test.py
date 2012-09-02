#coding: utf8

import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np
import numpy.linalg as la


SIZE = 102400*2
DIVIDER = 1
STATE_SIZE = 32*32
n = 1024


## PHP rand constants
MT_N = 624
M = 397
## END PHP rand constants

MT_state_result = np.zeros((MT_N, SIZE)).astype(np.uint32)

ctx = cl.create_some_context()
queue_instruction = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
queue_data = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

MT_state_buf = cl.Buffer(ctx, mf.WRITE_ONLY, MT_state_result.nbytes)

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

    __kernel void mt_brute(unsigned int seed_start,__global unsigned int c[MT_N][ALL_SIZE])
    {
      int gid = get_global_id(0);
      int lid = get_local_id(0);

      __local unsigned int state[3][STATES_SIZE];


      /* PHP_MT_VARIABLES*/
      __private register unsigned int i;
      register unsigned int s1;

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
      c[0][gid] = s2;

      for (; i < MT_N; ++i)
      {
        s2 = ( 1812433253U * ( r2 ^ (r2 >> 30) ) + i ) & 0xffffffffU;
        r2 = s2;
        c[i][gid] = s2;
      }
      /* END PHP_MT_INITIALIZE */


      /* PHP_MT_RELOAD */

      state[1][lid] = c[0][gid];

      for (i = 0; i < MT_N - M; ++i)
      {
        state[0][lid] = state[1][lid];  // +++ p[0] = c[0][gid]
        state[1][lid] = c[i+1][gid];
        state[2][lid] = c[i+M][gid];
        c[i][gid] = twist(state[2][lid], state[0][lid], state[1][lid]);
      }

      for (i = MT_N - M; i < MT_N-1; ++i)
      {
        state[0][lid] = state[1][lid];  // +++ p[0] = c[0][gid]
        state[1][lid] = c[i+1][gid];
        state[2][lid] = c[i-MT_N+M][gid];
        c[i][gid] = twist(state[2][lid], state[0][lid], state[1][lid]);
      }
      state[0][lid] = state[1][lid];  // +++ p[0] = c[0][gid]
      state[1][lid] = c[0][gid];
      state[2][lid] = c[i-MT_N+M][gid];
      c[i][gid] = twist(state[2][lid], state[0][lid], state[1][lid]);
      return;
/*

      s = state[lid];
      r = s;

      for (i = MT_N - M; i--; ++r)
        *r = twist(r[M], r[0], r[1]);

      for (i = M; --i; ++r)
        *r = twist(r[M-MT_N], r[0], r[1]);

      *r = twist(r[M-MT_N], r[0], s[0]);

      next = s;
*/
      /* END PHP_MT_RELOAD */

      /* PHP_MT_RAND */
/*
      for (i = 0; i < 8; ++i)
      {
          s1 = c[i][gid];
          s1 ^= (s1 >> 11);
          s1 ^= (s1 <<  7) & 0x9d2c5680U;
          s1 ^= (s1 << 15) & 0xefc60000U;
          s1 ^= (s1 >> 18);
          c[i][gid] = s1 >> 1;
      }
*/
      /* END PHP_MT_RAND */

/*
      for(i=0; i< MT_N; ++i)
      {
        c[gid][i] = state[i][lid];
      }
*/

      c[0][gid] = gid;//gid*2;
      c[1][gid] = lid;
      return;
    }
    """).build()
z = cl.enqueue_marker(queue_instruction)


instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(0), MT_state_buf)#, g_times_l=True)
data_event = cl.enqueue_copy(queue_data, MT_state_result, MT_state_buf, wait_for=[instr_event,])

#for i in xrange(10):
#    instr_event = prg.mt_brute(queue_instruction, (SIZE, ), (STATE_SIZE, ), np.uint32(i*SIZE), MT_state_buf, wait_for=[data_event,])#, g_times_l=True)
#    data_event = cl.enqueue_copy(queue_data, MT_state_result, MT_state_buf, wait_for=[instr_event,])


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

exit()

#MT_state_result = np.empty_like(a)
#print MT_state_result
for row in  MT_state_result:
    print row[0]
#print la.norm(MT_state_result - (a+b))
exit()

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


a = np.random.rand(256 ** 3).astype(np.int32)

ctx = cl.create_some_context()
queue_instruction = cl.CommandQueue(ctx)

mf = cl.mem_flags

seed_gpu = 0#cl_array.to_device(queue_instruction, np.arange(0, n).astype(np.uint32))
MT_state = np.zeros((n, MT_N, ), np.uint32)
MT_state_buf = cl.Buffer(ctx, mf.READ_WRITE, hostbuf=MT_state)

#seed_gpu = cl_array.zeros(queue_instruction, (1, ), np.uint32)


#MT_state_buf =


gen_state = cl.Program(ctx,
    """
        #define N {0}
    """.format(MT_N) + ""
                       """
                       void php_mt_initialize(unsigned int seed, unsigned int *state)
                       {
                           register unsigned int *s = state;
                           register unsigned int *r = state;
                           int i = 1;

                           *s++ = seed & 0xffffffffU;
                           for( ; i < N; ++i ) {
                               *s++ = ( 1812433253U * ( *r ^ (*r >> 30) ) + i ) & 0xffffffffU;
                               r++;
                           }
                       }
                       """
                       "__kernel void mt_srand("
    #"   unsigned int seed_start"
    + "  __global      unsigned int states[][{0}]".format(MT_N) + ")"
                                                                  """
                                                                  {
                                                                      int seed_start = 0;
                                                                      int gid = get_group_id(0);
                                                                      int gls = get_local_size(0);
                                                                      int lid = get_local_id(0);
                                                                      states[1][1] = 123;
                                                                      //php_mt_initialize(seed_start+(lid+gls*gid), (unsigned int*)states[lid+gls*gid]);
                                                                  }
                                                                  """
).build().mt_srand

gen_state(queue_instruction, (1024,), (1, 1), MT_state_buf)

result = np.empty_like(MT_state)
#cl.enqueue_read_buffer()
cl.enqueue_read_buffer(queue_instruction, MT_state_buf, result).wait()
print result

#print MT_state.get()
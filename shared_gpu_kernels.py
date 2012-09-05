#coding: utf8

__author__ = 'gifts'

def gen_kernel(MT_N, STATE_SIZE, M, SIZE, SIGNIFICANT_LENGTH):
    return ("""
    #define MT_N {0}
    #define STATES_SIZE {1}
    #define M {2}
    #define ALL_SIZE {3}
    #define RESULT_LEN {4}
    """.format(MT_N, STATE_SIZE, M, SIZE, SIGNIFICANT_LENGTH) +
    """

    #define hiBit(u)      ((u) & 0x80000000U)  /* mask all but highest   bit of u */
    #define loBit(u)      ((u) & 0x00000001U)  /* mask all but lowest    bit of u */
    #define loBits(u)     ((u) & 0x7FFFFFFFU)  /* mask     the highest   bit of u */
    #define mixBits(u, v) (hiBit(u)|loBits(v)) /* move hi bit of u to hi bit of v */
    #define twist(m,u,v)  (m ^ (mixBits(u,v)>>1) ^ ((unsigned int)(-(unsigned int)(loBit(u))) & 0x9908b0dfU))


    __kernel void mt_brute(         unsigned int seed_start,
        __global unsigned int *c,
        __global unsigned int *res
    )
    {
      __private int gid = get_global_id(0);
      __private int lid = get_local_id(0);


      /* PHP_MT_VARIABLES*/

      __private unsigned int i;
      __private unsigned int s1;
      __local unsigned int state[3*STATES_SIZE];
      __private unsigned int s2;
      __private unsigned int r2;

      /* END PHP_MT_VARIABLES*/


      /* PHP_MT_INITIALIZE */

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

      for (i = MT_N - M; i < MT_N-1; ++i)
      {
        state[                lid] = state[STATES_SIZE + lid];  // +++ p[0] = c[0][gid]
        state[STATES_SIZE   + lid] = c[(i+1) * ALL_SIZE + gid];
        state[STATES_SIZE*2 + lid] = c[(i-(MT_N - M)) * ALL_SIZE + gid];
        c[ALL_SIZE*i + gid] = twist(state[2*STATES_SIZE + lid], state[lid], state[STATES_SIZE + lid]);
      }
      state[                lid] = state[STATES_SIZE + lid];  // +++ p[0] = c[0][gid]
      state[STATES_SIZE   + lid] = c[gid];
      state[STATES_SIZE*2 + lid] = c[(i-(MT_N - M)) * ALL_SIZE + gid];
      c[ALL_SIZE*i + gid] = twist(state[2*STATES_SIZE + lid], state[lid], state[STATES_SIZE + lid]);

      /* END PHP_MT_RELOAD */


      /* PHP_MT_RAND */

      for (i = 0; i < RESULT_LEN; ++i)
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
    """)

def transform_to_opencl(code):
    return code

def transform_to_cuda(code):
    code = (code
            .replace('__private', '')
            .replace('__local', '__shared__')
            .replace('__global', '')
            .replace('__kernel', '__global__')
            .replace('get_local_id(0)', 'threadIdx.x')
            .replace('get_global_id(0)', 'blockIdx.x*blockDim.x+threadIdx.x')
    )
    return code


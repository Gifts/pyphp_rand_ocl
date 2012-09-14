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
      __private unsigned int s2;
      __private unsigned int x;

      /* END PHP_MT_VARIABLES*/


      /* PHP_MT_INITIALIZE */

      s2 = (seed_start + gid) & 0xffffffffU;

      x = 1812433253U * (s2 ^ (s2 >> 30)) + 1;
      for (i = 2; i <= M; i++)
      x = 1812433253U * (x ^ (x >> 30)) + i;

      x ^= ((s2 & 0x80000000U) | (r2 & 0x7fffffffU)) >> 1;
      x ^= (s2 & 1) * 0x9908b0dfU ;

      x ^= x >> 11;
      x ^= (x << 7) & 0x9d2c5680U;
      x ^= (x << 15) & 0xefc60000U;
      x ^= x >> 18;

      /* END PHP_MT_INITIALIZE */

      res[gid] = (long)(x >> 1);

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


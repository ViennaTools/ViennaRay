#ifndef RT_PRE_COMPILE_MACROS
#define RT_PRE_COMPILE_MACROS

#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X86
#include <immintrin.h>
#endif

#endif
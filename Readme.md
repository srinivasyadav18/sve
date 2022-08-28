## sve::experimental::simd header only library for SVE vectorization on A64FX

This library contains SIMD API for SVE vectorization that is built on top of `std::experimental::simd` API. It has most of the funcationalities of std::experimental::simd, except a few such as `where()` clause, `operator[]` for indexing the vector registers.

The library is tested on Ookami A64FX nodes with SVE 512 and AWS Gravition 3 cloud instances with SVE 256.


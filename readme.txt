Some code to create images using Alan Turing's reaction diffusion model

Nothing serious just easy to program and the patterns are an example
of complex behaviour due to the interactions of a few simple things

Can be numerically intensive so the heavy lifting is in Cython rather
than Python and will be parallelised using Cython's fairly easy way
of running threads across arrays in nested for loops


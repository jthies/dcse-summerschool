# Lab 1: Dense LU decomposition with OpenMP tasks

(Introductory text)

## Your tasks

- Compile and run the sequential code for different problem sizes. What is the computational complexity according to the measured runtimes?
- Insert OpenMP pragma's to achieve task-parallel factorization of each tile. Measure the scalability.
- Can you achieve task-based parallelization across tiles? Measure the scalability again
- Measure key performnace metrics on 12, 24 and 48 cores

# Lab 4: Performance optimization of communication-avoiding algorithms

(introductory text)

## What you will practice here

- measuring performance characteristics using likwid
- applying roofline analysis
- selecting orthogonalization strategies for performance and stability


## Your tasks

- using likwid, measure data traffic for the matrix powers kernel in the Trilinos implementation of s-step GMRES, plot it against s.
- measure the computational intensity of the block orthogonalization, plot against s.
- compare orthogonalization variants: MGS vs. CGS, SVQB/CholQR vs. TSQR
- (if time permits): implement your own task-based Q-less TSQR

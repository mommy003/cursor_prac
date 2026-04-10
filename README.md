# GCTB (quantized eigen / low-rank fork)

GCTB fork with quantized eigen LD blocks: per-SNP scaled **Q** (`.eigen.q*qc.bin`) and **U** transpose (`.eigen.u*utc.bin`) formats for SBayesR low-rank MCMC.

Build (adjust include paths in `Makefile` for Eigen and Boost on your system):

```bash
make
```

See `--help` for flags; eigen matrix options include `--ldm-eigen-q8-qc`, `--ldm-eigen-u8-utc`, etc.

# GCTB (quantized eigen / low-rank fork)

GCTB fork with quantized eigen LD blocks: per-SNP scaled **Q** (`.eigen.q*qc.bin`) and **U** transpose (`.eigen.u*utc.bin`) formats for SBayesR low-rank MCMC.

Build (adjust include paths in `Makefile` for Eigen and Boost on your system):

```bash
make
```

See `--help` for flags; eigen matrix options include `--ldm-eigen-q8-qc`, `--ldm-eigen-u8-utc`, etc.

For `.eigen.u*utc.bin` scale interpretation (quantized U transpose), the loader supports:

- legacy/default: `eigenScales` treated as U-column scales (`max|U[:,j]|`)
- auto-detect: infer whether scales already include `sqrt(lambda_j)`
- forced Q-scale: treat `eigenScales` as `max|sqrt(lambda_j) * U[:,j]|`

Set via environment variable:

```bash
GCTB_UUTC_SCALE_MODE=legacy   # default
GCTB_UUTC_SCALE_MODE=auto
GCTB_UUTC_SCALE_MODE=qscale
```

# Issue #7857 evidence figures

Evidence for [NVIDIA/Megatron-LM#7857](https://github.com/NVIDIA/Megatron-LM/issues/7857)
("Low-precision temperature scaling changes TorchSampling output at temperature=0.1 and 0").

| File | What it shows |
| --- | --- |
| `fig1_precision.png` | The two failures in the issue: the BF16 temperature divisor moves the top-p nucleus boundary (Nemotron-H-8B, step 29), and FP16 scaling overflows to `inf` at low temperature. |
| `fig2_disagreement.png` | How often the pre-fix path emits a token the FP32 reference excludes, over the (temperature, logit-gap) plane. |
| `make_figures.py` | Regenerates both PNGs. Run inside the Megatron environment: `python make_figures.py --out-dir .` |

The numbers come from the issue's own reproducer (Nemotron-H-8B-Base-8K, generated-token
index 29, seed 123, temperature=0.1, top_p=0.9) and from the pre-fix
`TorchSampling.filter_logits`, which divided the logits in the input dtype.

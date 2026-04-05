# Fig 3 Ablation Study -- LaTeX Report

Fig.~3 presents the ablation study on Qwen3-235B (FP8, batch size 1) across three parallelism strategies (TP16, HMP, and RO\_new) evaluated at sequence lengths of 8K, 256K, and 1M on a 4$\times$4 Mesh2D with 16 NPUs.

As shown in Fig.~3(a), RO\_new consistently achieves the highest end-to-end latency speedup over the TP16 baseline, reaching 1.1$\times$ at 8K, 1.5$\times$ at 256K, and 1.6$\times$ at 1M. HMP also outperforms TP16 but with slightly lower gains (1.1$\times$, 1.4$\times$, and 1.5$\times$), as it still incurs an additional AllReduce plus AllGather pair for the output projection that RO\_new eliminates through row-sharded $W_o$ with intra-group ReduceScatter.

The advantage becomes more pronounced in communication efficiency (Fig.~3(b)). RO\_new delivers 2.7$\times$, 17.7$\times$, and 65.4$\times$ communication speedup at 8K, 256K, and 1M respectively, compared to 1.3$\times$, 10.0$\times$, and 36.8$\times$ for HMP. This widening gap stems from TP16's attention AllReduce message size growing proportionally with sequence length (65\,KB at 8K to 16\,MB at 1M), while RO\_new maintains a constant 263\,ns communication overhead, making it particularly advantageous for long-context inference.

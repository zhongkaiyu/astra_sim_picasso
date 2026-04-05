# Fig 4 Batch Exploration -- Pareto Tradeoff LaTeX Report

Fig.~4 characterizes the throughput versus responsiveness tradeoff on Qwen3-235B (FP8, seq=64K, 4$\times$4 Mesh2D). As batch size increases from 1 to 64, total system throughput improves from 0.223 to 0.478\,tok/$\mu$s (2.14$\times$), while TPS/User decreases from 223K to 7.5K\,tokens/s, a 30$\times$ reduction, because each decoding step must process more tokens before any single user receives a response.

The Pareto frontier in Fig.~4(c) reveals a clear ``knee'' near bs=4 to 8: throughput nearly doubles (0.223$\to$0.437\,tok/$\mu$s, 1.96$\times$) with a moderate 4$\times$ TPS/User reduction, whereas further increasing bs beyond 8 yields only 10\% additional throughput at the cost of continued responsiveness degradation. This saturation arises because attention computation, which scales linearly with batch size and cannot amortize KV cache reads, grows from 44\% to 94\% of total wall time.

This analysis suggests bs=4 to 8 as the optimal operating region. For latency-critical interactive serving, bs=1 maximizes responsiveness at 223K\,TPS/User; for throughput-oriented offline inference, bs=16 captures 98\% of peak throughput. Across all operating points, our architecture strictly dominates all baselines on the Pareto frontier, achieving 14 to 16$\times$ higher throughput than H100 at comparable or lower power.

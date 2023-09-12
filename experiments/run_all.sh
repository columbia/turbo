#!/bin/sh

export LOGURU_LEVEL=ERROR

echo "Running Figures 7.a and 7.b (static-non-partitioned covid zipfs 0/1).."
python experiments/runner.cli.caching.py --exp caching_monoblock --dataset covid19

echo "Running Figure 7.c (static-non-partitioned citibike zipf 0).."
python experiments/runner.cli.caching.py --exp caching_monoblock --dataset citibike

echo "Running Figure 7.d (static-non-partitioned covid empirical convergence).."
python experiments/runner.cli.caching.py --exp convergence --dataset covid19

echo "Running Figure 8.a (static-non-partitioned covid heuristics).."
python experiments/runner.cli.caching.py --exp caching_monoblock_heuristics --dataset covid19

echo "Running Figure 8.b (static-non-partitioned covid learning rate).."
python experiments/runner.cli.caching.py --exp caching_monoblock_learning_rates --dataset covid19

echo "Running Figure 9.a and 9.b (static-partitioned covid zipfs 0/1).."
python experiments/runner.cli.caching.py --exp caching_static_multiblock_laplace_vs_hybrid --dataset covid19

echo "Running Figure 9.c (static-partitioned citibike zipf 0).."
python experiments/runner.cli.caching.py --exp caching_static_multiblock_laplace_vs_hybrid --dataset citibike

echo "Running Figure 10.a and 10.b (streaming-partitioned covid zipfs 0/1).."
python experiments/runner.cli.caching.py --exp caching_streaming_multiblock_laplace_vs_hybrid --dataset covid19

echo "Running Figure 10.c (streaming-partitioned citibike zipf 0).."
python experiments/runner.cli.caching.py --exp caching_streaming_multiblock_laplace_vs_hybrid --dataset citibike

echo "Running Figure 10.d (system runtime evaluation covid).."
python turbo/run_simulation.py --omegaconf "turbo/config/turbo_system_eval_monoblock_covid.json"

echo "Clean up Redis/Postgres after the experiment.."
python packaging/storage_utils.py --omegaconf turbo/config/turbo_system_eval_monoblock_covid.json --storage "*" --function delete-all --database covid

echo "Running Figure 10.d (system runtime evaluation citibike).."
python turbo/run_simulation.py --omegaconf turbo/config/turbo_system_eval_monoblock_citibike.json

echo "Clean up Redis/Postgres after the experiment.."
python packaging/storage_utils.py --omegaconf turbo/config/turbo_system_eval_monoblock_citibike.json --storage "*" --function delete-all --database citibike

mkdir -p logs/figures
python experiments/utils.py --function analyze_runtime --experiment-path system_runtime_covid > logs/figures/figure_10d.txt
python experiments/utils.py --function analyze_runtime --experiment-path system_runtime_citibike >> logs/figures/figure_10d.txt

# Map generated plot to corresponding figures
cp logs/ray/covid19/monoblock/laplace_vs_hybrid/cumulative_budget_utilization.png logs/figures/figures_7a_7b.png
cp logs/ray/citibike/monoblock/laplace_vs_hybrid/cumulative_budget_utilization.png logs/figures/figure_7c.png
cp logs/ray/covid19/monoblock/convergence/empirical_convergence.png logs/figures/figure_7d.png

cp logs/ray/covid19/monoblock/heuristics/cumulative_budget_utilization.png logs/figures/figure_8a.png
cp logs/ray/covid19/monoblock/learning_rates/cumulative_budget_utilization.png logs/figures/figure_8b.png

cp logs/ray/covid19/static_multiblock/laplace_vs_hybrid/cumulative_budget_utilization.png logs/figures/figures_9a_9b.png
cp logs/ray/citibike/static_multiblock/laplace_vs_hybrid*/cumulative_budget_utilization.png logs/figures/figures_9c.png

cp logs/ray/covid19/streaming_multiblock/laplace_vs_hybrid/cumulative_budget_utilization.png logs/figures/figures_10a_10b.png
cp logs/ray/citibike/streaming_multiblock/laplace_vs_hybrid/cumulative_budget_utilization.png logs/figures/figures_10c.png
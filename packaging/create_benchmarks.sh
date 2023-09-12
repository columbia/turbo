#!/bin/sh

# Create datasets for covid and citibike
python data/covid19/covid19_data/generate.py --num-blocks-cutoff 50
python data/citibike/citibike_data/generate.py
rm -rf data/citibike/citibike_data/months

# Create query pools for covid and citibike
python data/covid19/covid19_queries/queries.py
python data/citibike/citibike_queries/queries.py

# Create workloads for covid and citibike
python data/workload_generator.py --utility 0.05 \
                                  --utility-beta 0.001 \
                                  --queries "data/covid19/covid19_queries/all.queries.json" \
                                  --workload-dir "data/covid19/covid19_workload" \
                                  --blocks-metadata-path "data/covid19/covid19_data/blocks/metadata.json"

python data/workload_generator.py --utility 0.05 \
                                  --utility-beta 0.001 \
                                  --queries "data/citibike/citibike_queries/stories.queries.json" \
                                  --workload-dir "data/citibike/citibike_workload" \
                                  --blocks-metadata-path "data/citibike/citibike_data/blocks/metadata.json"


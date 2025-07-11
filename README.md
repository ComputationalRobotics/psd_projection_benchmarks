# PSD Projection Benchmarks
Benchmarks for the [`psd_projection`](https://github.com/ComputationalRobotics/psd_projection) library.

## Datasets
Datasets are generated using the [Matrix Depot](https://matrix-depot.readthedocs.io/en/latest/) Julia package. The datasets are stored in the `data` directory, and can be generated using the `data/generate.jl` script. The datasets were generated in parallel, so you may not get the same results if you generate them yourself.

### Generating the datasets
Note that this approach requires around 130 GB of disk space, can take a long time, and might not give the same results as the datasets we used in our experiments. We recommend using the pre-generated datasets instead.

To generate the datasets, you need to have [Julia](https://julialang.org/) installed. Then, you can run the following command in the terminal:
```bash
datasets=("cauchy" "chebspec") # see run.sh for the full list of datasets
instance_sizes=(10 20) # we used 5000, 10000, 20000 in our experiments
julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}"
```

### Downloading the datasets
This part is coming soon.

## Running the benchmarks
We provide several scripts to run the benchmarks:
- `run.sh`: runs the benchmarks using the `psd_projection` library; requires the library to cloned, as well as CMake;
- `run_full.sh`: runs the benchmarks by calling `benchmark_full.cu`, a self-contained CUDA program that does not require the `psd_projection` library;
- `run_final_1.sh`: same as `run_full.sh`, but runs the first half of the datasets;
- `run_final_2.sh`: same as `run_full.sh`, but runs the second half of the datasets.

We used `run_final_1.sh` and `run_final_2.sh` for our experiments. All scripts assume that the datasets are already generated (see above).

Results are stored in the `results` directory, in a CSV file named `psd_results.csv`. The scripts also generate an `output.txt` file with the output of the benchmark and useful debugging information.
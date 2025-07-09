import pandas as pd

error_file = "results/benchmark_error.tex"
time_file = "results/benchmark_time.tex"

GPU = "B200"

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("results/saved/final/" + GPU + ".csv")
    output_prefix = "results/saved/paper/" + GPU

    # renames
    df["method"] = df["method"].replace("newton TF16", "Newton-Schutz FP16")
    df["method"] = df["method"].replace("newton FP32", "Newton-Schutz FP32")
    df["method"] = df["method"].replace("composite TF16", "composite FP16")
    df["method"] = df["method"].replace("polarexpress TF16", "Polar Express FP16")
    df["method"] = df["method"].replace("polarexpress FP32", "Polar Express FP32")
    df["method"] = df["method"].replace("composite FP32", "Composite FP32")
    df["method"] = df["method"].replace("composite FP32 emulated", "Composite FP32 (emulated)")
    df["method"] = df["method"].replace("composite FP16", "Composite FP16")

    methods = ["cuSOLVER FP64", "cuSOLVER FP32", "Polar Express FP32", "Polar Express FP16", "Newton-Schutz FP32", "Newton-Schutz FP16", "Composite FP32", "Composite FP32 (emulated)", "Composite FP16"]

    # change the column names
    df.columns = ["Dataset", "Size", "Method", "Time (s)", "Relative Error"]
    
    # open error_file for writing
    with open(error_file, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2e", caption="Relative error of the PSD projection methods on different datasets.", label="tab:benchmark_error"))

    # open time_file for writing
    with open(time_file, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2e", caption="Time taken by the PSD projection methods on different datasets.", label="tab:benchmark_time"))


    ### Statistics table
    for n in [5000, 10000, 20000]:
        stats = df.groupby("Method").agg({"Relative Error": ["mean", "median", "std"], "Time (s)": ["mean", "median", "std"]})
        stats = stats.reindex(methods)
        styler = stats.style
        # styler = stats.style.highlight_min(axis=0, props='bfseries:;')
        styler = styler.format(precision=2, formatter="\\num{{{:.2e}}}".format)
        stats.columns = stats.columns.set_levels(
            ["\quad Mean", "\quad Median", "\quad Std."], level=1
        )
        
        with open(f"{output_prefix}_stats_{n}.tex", "w") as f:
            f.write(styler.to_latex(
                caption=f"Results of the PSD projection methods on datasets of size {n} for " + GPU + " GPU.",
                label=f"tab:benchmark_stats_{n}_" + GPU,
                hrules=True,
                column_format="lllllllll",
                multicol_align="c"
            ))
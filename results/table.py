import pandas as pd

GPU = "B200"

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("results/saved/final/" + GPU + ".csv")
    output_prefix = "results/saved/paper/" + GPU

    # renames
    df["method"] = df["method"].replace("newton TF16", "Newton-Schulz FP16")
    df["method"] = df["method"].replace("newton FP32", "Newton-Schulz FP32")
    df["method"] = df["method"].replace("composite TF16", "composite FP16")
    df["method"] = df["method"].replace("polarexpress TF16", "Polar Express FP16")
    df["method"] = df["method"].replace("polarexpress FP32", "Polar Express FP32")
    df["method"] = df["method"].replace("composite FP32", "Composite FP32")
    df["method"] = df["method"].replace("composite FP32 emulated", "Composite FP32 (emulated)")
    df["method"] = df["method"].replace("composite FP16", "Composite FP16")

    methods = ["cuSOLVER FP64", "cuSOLVER FP32", "Polar Express FP32", "Polar Express FP16", "Newton-Schulz FP32", "Newton-Schulz FP16", "Composite FP32", "Composite FP32 (emulated)", "Composite FP16"]

    # change the column names
    df.columns = ["Dataset", "Size", "Method", "Time (s)", "Relative Error"]

    ### Statistics table
    for n in [5000, 10000, 20000]:
        data = df[df["Size"] == n]
        remove = "Composite FP32" if GPU == "B200" else "Composite FP32 (emulated)"
        data = data[data["Method"] != remove]
        stats = data.groupby("Method").agg({"Relative Error": ["mean", "median", "std"], "Time (s)": ["mean", "median", "std"]})
        methods_no_fp32 = methods.copy()
        methods_no_fp32.remove(remove)
        stats = stats.reindex(methods_no_fp32)
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
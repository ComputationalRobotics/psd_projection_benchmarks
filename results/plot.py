import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

datasets = ["cauchy", "chebspec", "chow", "circul", "clement", "companion", "dingdong", "fiedler", "forsythe", "frank", "golub", "grcar", "hankel", "hilb", "kahan", "kms", "lehmer", "lotkin", "magic", "minij", "moler", "oscillate", "parter", "pei", "prolate", "randcorr", "rando", "rohess", "sampling", "toeplitz", "tridiag", "triw", "wilkinson"]


if __name__ == "__main__":
    n_first_datasets = len(datasets)
    n = 5000

    # Load the data
    # df = pd.read_csv("results/saved/psd_results-B200.csv")
    df = pd.read_csv("results/psd_results.csv")
    n_datasets = len(pd.unique(df["dataset"]))
    aspect = min(n_first_datasets, n_datasets) / 6

    # print total execution time
    total_time = df["time"].sum()
    print(f"\nTotal execution time for all methods: {total_time:.2f} s")

    df = df[df["n"] == n]
    df = df[df["dataset"].isin(datasets[:n_first_datasets])]

    sns.catplot(
        data=df[df["method"] != "cuSOLVER FP64"],
        x="dataset",
        y="relative_error",
        hue="method",
        # col="dataset",
        kind="bar",
        # height=3,
        aspect=aspect,
        palette=sns.color_palette()[1:6],
    )
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.xlabel("Dataset")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.title(f"$n={n}$")
    plt.savefig("results/benchmark_error.pdf", dpi=300, bbox_inches="tight")

    sns.catplot(
        data=df,
        x="dataset",
        y="time",
        hue="method",
        # col="dataset",
        kind="bar",
        # height=3,
        aspect=aspect,
        palette=sns.color_palette()[:6],
    )
    plt.xlabel("Dataset")
    plt.ylabel("Time (s)")
    plt.title(f"$n={n}$")
    plt.savefig("results/benchmark_time.pdf", dpi=300, bbox_inches="tight")

    with pd.option_context('display.float_format', '{:.2e}'.format):
        print(f"\nStats for n = {n}, {min(n_first_datasets, n_datasets)} datasets:")

        method_order = [
            "cuSOLVER FP64", 
            "cuSOLVER FP32", 
            "polarexpress FP32",
            "polarexpress TF16",
            "newton FP32",
            "newton TF16",
            "composite FP32", 
            "composite FP32 emulated", 
            "composite TF16"
        ]

        stats = df[df["dataset"] != "triw"].groupby("method").agg({"relative_error": ["mean", "median", "max"], "time": ["mean", "median"]})
        stats = stats.reindex(method_order)
        print(stats)

        # find the dataset with the maximum relative error of composite FP32
        df_composite_fp32 = df[df["method"] == "composite FP32"]
        max_error_dataset = df_composite_fp32.loc[df_composite_fp32["relative_error"].idxmax(), "dataset"]
        print(f"\nDataset with maximum relative error for composite FP32: {max_error_dataset}")

        # find the dataset with the maximum relative error of composite TF16
        df_composite_tf16 = df[df["method"] == "composite TF16"]
        max_error_dataset = df_composite_tf16.loc[df_composite_tf16["relative_error"].idxmax(), "dataset"]
        print(f"Dataset with maximum relative error for composite TF16: {max_error_dataset}")
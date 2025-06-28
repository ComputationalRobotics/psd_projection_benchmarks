import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

datasets = ["cauchy", "chebspec", "chow", "circul", "clement", "companion", "dingdong", "fiedler", "forsythe", "frank", "golub", "grcar", "hankel", "hilb", "kahan", "kms", "lehmer", "lotkin", "magic", "minij", "moler", "oscillate", "parter", "pei", "prolate", "randcorr", "rando", "randsvd", "rohess", "sampling", "toeplitz", "tridiag", "triw", "wilkinson"]


if __name__ == "__main__":
    n_first_datasets = len(datasets)
    aspect = n_first_datasets / 6
    n = 5000

    # Load the data
    # df = pd.read_csv("results/saved/psd_results_non_deflate_4.csv")
    df = pd.read_csv("results/psd_results.csv")
    df = df[df["n"] == n]
    # df = df[df["relative_error"] < 1e-3]
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
        palette=sns.color_palette()[1:],
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
        palette=sns.color_palette(),
    )
    plt.xlabel("Dataset")
    plt.ylabel("Time (s)")
    plt.title(f"$n={n}$")
    plt.savefig("results/benchmark_time.pdf", dpi=300, bbox_inches="tight")
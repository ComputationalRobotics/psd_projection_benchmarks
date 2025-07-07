import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

datasets = ["cauchy", "chebspec", "chow", "circul", "clement", "companion", "dingdong", "fiedler", "forsythe", "frank", "golub", "grcar", "hankel", "hilb", "kahan", "kms", "lehmer", "lotkin", "magic", "minij", "moler", "oscillate", "parter", "pei", "prolate", "randcorr", "rando", "rohess", "sampling", "toeplitz", "tridiag", "triw", "wilkinson"]

if __name__ == "__main__":
    n_first_datasets = len(datasets)

    # Load the data
    df = pd.read_csv("results/saved/B200/psd_final.csv")
    n_datasets = len(pd.unique(df["dataset"]))

    # renames
    df["method"] = df["method"].replace("newton TF16", "Newton-Schutz FP16")
    df["method"] = df["method"].replace("newton FP32", "Newton-Schutz FP32")
    df["method"] = df["method"].replace("composite TF16", "composite FP16")
    df["method"] = df["method"].replace("polarexpress TF16", "Polar Express FP16")
    df["method"] = df["method"].replace("polarexpress FP32", "Polar Express FP32")
    df["method"] = df["method"].replace("composite FP32 emulated", "Composite FP32 (emulated)")
    df["method"] = df["method"].replace("composite FP16", "Composite FP16")


    # print total execution time
    total_time = df["time"].sum()
    print(f"\nTotal execution time for all methods: {total_time:.2f} s")

    # df = df[df["n"] == n]
    # df = df[df["dataset"].isin(datasets[:n_first_datasets])]

    ### Full error plot
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 8.3))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = ["cuSOLVER FP64", "composite FP32"]
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]
        sns.barplot(
            ax=axs[i],
            data=data,
            x="dataset",
            y="relative_error",
            hue="method",
            palette=sns.color_palette()[1:8],
        )
        axs[i].set_ylabel("Relative Error")
        axs[i].set_yscale("log")
        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_xlabel("Dataset" if i == 2 else "")

        axs[i].legend_.remove()

        for label in axs[i].get_xticklabels():
            label.set_ha('right')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.,
        title="Method"
    )
    fig.tight_layout()
    plt.savefig("results/benchmark_error.pdf", dpi=300, bbox_inches="tight")

    ### Full time plot
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 8.3))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = ["composite FP32"]
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]
        sns.barplot(
            ax=axs[i],
            data=data,
            x="dataset",
            y="time",
            hue="method",
            palette=sns.color_palette()[0:8],
        )
        axs[i].set_ylabel("Time (s)")
        axs[i].set_yscale("log")
        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_xlabel("Dataset" if i == 2 else "")

        axs[i].legend_.remove()

        for label in axs[i].get_xticklabels():
            label.set_ha('right')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.,
        title="Method"
    )
    fig.tight_layout()
    plt.savefig("results/benchmark_time.pdf", dpi=300, bbox_inches="tight")

    # with pd.option_context('display.float_format', '{:.2e}'.format):
    #     print(f"\nStats for n = {n}, {min(n_first_datasets, n_datasets)} datasets:")

    #     method_order = [
    #         "cuSOLVER FP64", 
    #         "cuSOLVER FP32", 
    #         "polarexpress FP32",
    #         "polarexpress TF16",
    #         "newton FP32",
    #         "newton TF16",
    #         "composite FP32", 
    #         "composite FP32 emulated", 
    #         "composite TF16"
    #     ]

    ### Average time plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = ["composite FP32"]
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]

        sns.boxplot(
            ax=axs[i],
            data=data,
            x="method",
            y="time",
            hue="method",
            palette=sns.color_palette()[:8],
            showfliers=False,
            whis=[10, 90],
        )
        axs[i].set_ylabel("Time (s)" if i == 0 else "")
        axs[i].set_yscale("log")
        axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=30))
        axs[i].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[1,2,3,4,5,6,7,8,9], numticks=100))
        axs[i].tick_params(axis='y', which='minor', labelsize=8)
        axs[i].tick_params(axis='y', which='major', labelsize=10)
        # Show minor tick labels (can be verbose):
        axs[i].yaxis.set_minor_formatter(FormatStrFormatter('%.0e'))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        # axs[i].tick_params(axis='y', which='minor', labelleft=True, left=True)
        # axs[i].tick_params(axis='y', which='major', labelleft=True, left=True)

        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[i].set_xlabel("")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7, which='both')
        axs[i].grid(axis='x', linestyle='-', alpha=0.7, which='both')


    methods = df[~df["method"].isin(remove_methods)]["method"].unique()
    palette = sns.color_palette()[:len(methods)]

    handles = [Patch(facecolor=palette[i], label=method) for i, method in enumerate(methods)]

    fig.legend(
        handles, methods,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(methods)//2,
        frameon=True,
        title="Method"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("results/benchmark_time_avg.pdf", dpi=300, bbox_inches="tight")



    ### Average error plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = ["cuSOLVER FP64", "composite FP32"]
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]

        sns.boxplot(
            ax=axs[i],
            data=data,
            x="method",
            y="relative_error",
            hue="method",
            palette=sns.color_palette()[1:1+7],
            showfliers=False,
            whis=[10, 90],
        )
        axs[i].set_ylabel("Relative Error" if i == 0 else "")
        axs[i].set_yscale("log")
        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[i].set_xlabel("")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7, which='both')

        # for label in axs[i].get_xticklabels():
        #     label.set_ha('right')

    methods = df[~df["method"].isin(remove_methods)]["method"].unique()
    palette = sns.color_palette()[1:1+len(methods)]

    handles = [Patch(facecolor=palette[i], label=method) for i, method in enumerate(methods)]

    fig.legend(
        handles, methods,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(methods)//2+len(methods)%2,
        frameon=True,
        title="Method"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("results/benchmark_error_avg.pdf", dpi=300, bbox_inches="tight")


        # stats = df[df["dataset"] != "triw"].groupby("method").agg({"relative_error": ["mean", "median", "max"], "time": ["mean", "median"]})
        # stats = stats.reindex(method_order)
    #     print(stats)

    #     # find the dataset with the maximum relative error of composite FP32
    #     df_composite_fp32 = df[df["method"] == "composite FP32"]
    #     max_error_dataset = df_composite_fp32.loc[df_composite_fp32["relative_error"].idxmax(), "dataset"]
    #     print(f"\nDataset with maximum relative error for composite FP32: {max_error_dataset}")

    #     # find the dataset with the maximum relative error of composite TF16
    #     df_composite_tf16 = df[df["method"] == "composite TF16"]
    #     max_error_dataset = df_composite_tf16.loc[df_composite_tf16["relative_error"].idxmax(), "dataset"]
    #     print(f"Dataset with maximum relative error for composite TF16: {max_error_dataset}")
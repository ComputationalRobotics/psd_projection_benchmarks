import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import LogFormatterSciNotation

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

datasets = ["cauchy", "chebspec", "chow", "circul", "clement", "companion", "dingdong", "fiedler", "forsythe", "frank", "golub", "grcar", "hankel", "hilb", "kahan", "kms", "lehmer", "lotkin", "magic", "minij", "moler", "oscillate", "parter", "pei", "prolate", "randcorr", "rando", "rohess", "sampling", "toeplitz", "tridiag", "triw", "wilkinson"]

import matplotlib.ticker as mtick

# ------------------------------------------------------------------
#  put this inside the loop where you configure each Axes (ax = axs[i])
# ------------------------------------------------------------------
# sci_fmt = mtick.ScalarFormatter(useMathText=True)  # 5×10^{1} style
# sci_fmt.set_scientific(True)       # force scientific notation
# sci_fmt.set_powerlimits((0, 0))    # ... for *all* numbers
# sci_fmt.set_useOffset(False)       # no offset like "×10³" in the corner

def sci_fmt(x, _pos):
    if x == 0:
        return "0"
    exp   = int(np.floor(np.log10(x)))
    coeff = x / 10**exp
    if np.isclose(coeff, 1):
        return rf"$10^{{{exp}}}$"                #   10^{−2}
    return rf"${coeff:g}\times10^{{{exp}}}$"     # 2×10^{−2}

if __name__ == "__main__":
    n_first_datasets = len(datasets)

    # Load the data
    df = pd.read_csv("results/saved/final/B200.csv")
    output_prefix = "results/saved/paper/B200"
    n_datasets = len(pd.unique(df["dataset"]))

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
    method_colors = {m: color for m, color in zip(methods, sns.color_palette("tab10", len(methods)))}
    
    # print total execution time
    total_time = df["time"].sum()
    print(f"\nTotal execution time for all methods: {total_time:.2f} s")

    # if emulated is available, remove it non-emulated FP32
    if "Composite FP32 (emulated)" in df["method"].unique():
        df = df[~(df["method"] == "Composite FP32")]

    # df = df[df["n"] == n]
    # df = df[df["dataset"].isin(datasets[:n_first_datasets])]

    ### Full error plot
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 8.3))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = ["cuSOLVER FP64"]
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]
        methods_in_plot = data["method"].unique()
        palette = {m: method_colors[m] for m in methods_in_plot}
        sns.barplot(
            ax=axs[i],
            data=data,
            x="dataset",
            y="relative_error",
            hue="method",
            palette=palette,
        )
        axs[i].set_ylabel("Relative Error")
        axs[i].set_yscale("log")
        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_xlabel("Dataset" if i == 2 else "")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7, which='both')
        axs[i].set_axisbelow(True)

        axs[i].legend_.remove()

        for label in axs[i].get_xticklabels():
            label.set_ha('right')

    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(
    #     handles, labels,
    #     loc='center left',
    #     bbox_to_anchor=(1.01, 0.5),
    #     borderaxespad=0.,
    #     title="Method"
    # )
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),   # slightly above the plot; tweak as needed
        ncol=len(labels),             # This puts all labels in one row
        borderaxespad=0.,
        # title="Method",
        fontsize='small'  # or an integer like 8
    )
    fig.tight_layout()
    plt.savefig(output_prefix + "-error.pdf", dpi=300, bbox_inches="tight")

    ### Full time plot
    fig, axs = plt.subplots(3, 1, figsize=(11.7, 8.3))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = []
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]
        methods_in_plot = data["method"].unique()
        palette = {m: method_colors[m] for m in methods_in_plot}
        sns.barplot(
            ax=axs[i],
            data=data,
            x="dataset",
            y="time",
            hue="method",
            palette=palette,
        )
        axs[i].set_ylabel("Time (s)")
        axs[i].set_yscale("log")
        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_xlabel("Dataset" if i == 2 else "")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7, which='both')
        axs[i].set_axisbelow(True)

        axs[i].legend_.remove()

        for label in axs[i].get_xticklabels():
            label.set_ha('right')

    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(
    #     handles, labels,
    #     loc='center left',
    #     bbox_to_anchor=(1.01, 0.5),
    #     borderaxespad=0.,
    #     title="Method"
    # )
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),   # slightly above the plot; tweak as needed
        ncol=len(labels),             # This puts all labels in one row
        borderaxespad=0.,
        # title="Method",
        fontsize='small'  # or an integer like 8
    )
    fig.tight_layout()
    plt.savefig(output_prefix + "-time.pdf", dpi=300, bbox_inches="tight")

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
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = []
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]
        methods_in_plot = data["method"].unique()
        palette = {m: method_colors[m] for m in methods_in_plot}

        sns.boxplot(
            ax=axs[i],
            data=data,
            x="method",
            y="time",
            hue="method",
            palette=palette,
            # showfliers=False,
            # whis=[10, 90],
            showfliers=True,
            whis=1.5,
        )

        # Change default outlier marker from 'o' to '+' --------------------
        for line in axs[i].lines:
            if line.get_marker() == 'o':
                line.set_marker('+')
                line.set_markersize(6)
                line.set_markeredgewidth(1.0)

        axs[i].set_ylabel("Time (s)" if i == 0 else "")
        axs[i].set_yscale("log")
        axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        axs[i].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 4, 6, 8], numticks=100))
        axs[i].tick_params(axis='y', which='minor', labelsize=8)
        axs[i].tick_params(axis='y', which='major', labelsize=10)
        axs[i].yaxis.set_minor_formatter(sci_fmt)
        axs[i].yaxis.set_major_formatter(sci_fmt)

        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[i].set_xlabel("")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7, which='both')
        axs[i].grid(axis='x', linestyle='-', alpha=0.7, which='both')


    methods = df[~df["method"].isin(remove_methods)]["method"].unique()
    palette = {m: method_colors[m] for m in methods}

    handles = [Patch(facecolor=palette[method], label=method) for method in methods]

    fig.legend(
        handles, methods,
        loc='center right',
        bbox_to_anchor=(1.22, 0.5),  # Slightly outside the plot on the right
        borderaxespad=0.,
        ncol=1,
        frameon=True,
        # title="Method"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_prefix + "-time-avg.pdf", dpi=300, bbox_inches="tight")

    ### Average error plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    for i, n in enumerate([5000, 10000, 20000]):
        remove_methods = ["cuSOLVER FP64"]
        data = df[~(df["method"].isin(remove_methods)) & (df["n"] == n)]
        methods_in_plot = data["method"].unique()
        palette = {m: method_colors[m] for m in methods_in_plot}

        sns.boxplot(
            ax=axs[i],
            data=data,
            x="method",
            y="relative_error",
            hue="method",
            palette=palette,
            # showfliers=False,
            # whis=[10, 90],
            showfliers=True,
            whis=1.5,
        )

        # Change default outlier marker from 'o' to '+' --------------------
        for line in axs[i].lines:
            if line.get_marker() == 'o':
                line.set_marker('+')
                line.set_markersize(6)
                line.set_markeredgewidth(1.0)
        
        # axs[i].yaxis.set_major_locator(LogLocator(base=10.0))
        # axs[i].yaxis.set_minor_locator(LogLocator(base=10.0,
        #                                       subs=np.arange(1, 10) * 0.1))

        

        axs[i].set_ylabel("Relative Error" if i == 0 else "")
        axs[i].set_yscale("log")
        axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        axs[i].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 4, 6, 8], numticks=100))
        axs[i].tick_params(axis='y', which='minor', labelsize=8)
        axs[i].tick_params(axis='y', which='major', labelsize=10)
        axs[i].set_title(f"$n={n}$")
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[i].set_xlabel("")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7, which='both')

        # for label in axs[i].get_xticklabels():
        #     label.set_ha('right')

    methods = df[~df["method"].isin(remove_methods)]["method"].unique()
    palette = {m: method_colors[m] for m in methods}

    handles = [Patch(facecolor=palette[method], label=method) for method in methods]

    fig.legend(
        handles, methods,
        loc='center right',
        bbox_to_anchor=(1.22, 0.5),  # Slightly outside the plot on the right
        borderaxespad=0.,
        ncol=1,
        frameon=True,
        # title="Method"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_prefix + "-error-avg.pdf", dpi=300, bbox_inches="tight")


    # for n in [5000, 10000, 20000]:
    #     stats = df.groupby("method").agg({"relative_error": ["mean", "median", "max", "std"], "time": ["mean", "median", "std"]})
    #     stats = stats.reindex(methods)
    #     print(stats)

        # find the dataset with the maximum relative error of composite FP32
        # df_composite_fp32 = df[df["method"] == "composite FP32"]
        # max_error_dataset = df_composite_fp32.loc[df_composite_fp32["relative_error"].idxmax(), "dataset"]
        # print(f"\nDataset with maximum relative error for composite FP32: {max_error_dataset}")

        # # find the dataset with the maximum relative error of composite TF16
        # df_composite_tf16 = df[df["method"] == "composite TF16"]
        # max_error_dataset = df_composite_tf16.loc[df_composite_tf16["relative_error"].idxmax(), "dataset"]
        # print(f"Dataset with maximum relative error for composite TF16: {max_error_dataset}")
import pandas as pd

error_file = "results/benchmark_error.tex"
time_file = "results/benchmark_time.tex"

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("results/psd_results.csv")

    # change the column names
    df.columns = ["Dataset", "Size", "Method", "Time (s)", "Relative Error"]
    
    # open error_file for writing
    with open(error_file, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2e", caption="Relative error of the PSD projection methods on different datasets.", label="tab:benchmark_error"))

    # open time_file for writing
    with open(time_file, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2e", caption="Time taken by the PSD projection methods on different datasets.", label="tab:benchmark_time"))
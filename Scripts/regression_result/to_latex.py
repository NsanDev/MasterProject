import pandas as pd


def print_table(file):
    df = pd.read_pickle(file)
    df = df[df.columns[:4]]
    df.columns = ["b_S", "b_V", "alpha", "SE"]
    df["test"] = (df["alpha"] - 1) / df["SE"]
    latex = df.to_latex(bold_rows=True, longtable=True, index=False)
    print(latex)


print_table("regression_delta_intensity.pkl")
print_table("regression_delta_S.pkl")
print_table("regression_cva_hull.pkl")

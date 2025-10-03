"""
EBSD manuscript tables (final, resolution-invariant components in caption):

1) Accuracy table (DI + PCA; spherical excluded), grouped by resolution with dict size.
2) KNN throughput table only:
     - Rows: FP32 / FP16 / INT8 per resolution group
     - Columns: DI, PCA-<N1>, PCA-<N2>, ...
   Caption lists single (resolution-invariant) values for:
     - Dictionary Interpolation (formerly "simulation")
     - PCA
     - Projection per PCA-N
     - Spherical FFT (L vs pps)
"""

from pathlib import Path
import numpy as np
import pandas as pd


# --------------------------- IO ---------------------------


def load_results(results_file="benchmark_results/benchmark_dictionary.npy"):
    data = np.load(results_file, allow_pickle=True).item()
    return data


# --------------------------- Helpers ---------------------------


def _fmt_int(x):
    try:
        if x is None:
            return "—"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "—"
        return f"{int(x):,}"
    except Exception:
        return "—"


def _fmt_rate(x):
    try:
        if x is None:
            return "—"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "—"
        return f"{x:,.0f}"
    except Exception:
        return "—"


def _fmt_float(x, digits=2):
    try:
        if x is None:
            return "—"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    except Exception:
        return "—"


def _header_with_dict_size(res_deg, dict_size):
    return f"{res_deg}$^\\circ$ ({_fmt_int(dict_size)})"


def _sorted_unique(series):
    vals = pd.unique(series)
    try:
        return sorted(v for v in vals if pd.notna(v))
    except Exception:
        return list(vals)


# --------------------------- Tables ---------------------------


def create_accuracy_table(data):
    df = pd.DataFrame(
        {
            "dataset_id": data["dataset_id"],
            "method": data["method"],
            "dict_resolution": data["dict_resolution"],
            "dict_size": data.get("dict_size", [np.nan] * len(data["dataset_id"])),
            "pca_components": data["pca_components"],
            "dtype": data["dtype"],
            "mean_disorientation": data["mean_disorientation"],
            "fraction_above_3deg": data["fraction_above_3deg"],
        }
    )

    # Exclude Spherical
    df = df[df["method"] != "Spherical"]

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{EBSD Indexing Accuracy Comparison}")
    print("\\label{tab:ebsd_accuracy}")
    print("\\small")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{\\textbf{Method}} & "
        "\\multicolumn{3}{c}{\\textbf{Mean Disorientation (°)}} & "
        "\\multicolumn{3}{c}{\\textbf{Angles $>$ 3° (\\%)}} \\\\"
    )
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    print("& Low & Med & High & Low & Med & High \\\\")
    print("\\midrule")

    for res in _sorted_unique(df["dict_resolution"]):
        df_res = df[df["dict_resolution"] == res]
        ds = df_res["dict_size"].dropna()
        dict_size = int(ds.iloc[0]) if len(ds) else None
        print(
            f"\\multicolumn{{7}}{{l}}{{\\textbf{{{_header_with_dict_size(res, dict_size)}}}}} \\\\"
        )

        # DI then PCA-N
        if (df_res["method"] == "DI").any():
            _print_accuracy_row("Dictionary", df_res[df_res["method"] == "DI"])

        for n in _sorted_unique(
            df_res.loc[df_res["method"] == "PCA", "pca_components"]
        ):
            if n and n > 0:
                sub = df_res[
                    (df_res["method"] == "PCA") & (df_res["pca_components"] == n)
                ]
                _print_accuracy_row(f"PCA-{int(n)}", sub)

        if res != _sorted_unique(df["dict_resolution"])[-1]:
            print("\\addlinespace[0.2em]")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        "\\caption*{Mean disorientation (left block) and percentage of disorientations over $>$3° (right block) are shown as triplets: FP32 FP16 INT8. Columns indicate which scan (Low/Med/High) was tested. The values show that indexing accuracy is not significantly modified by using FP16 or INT8 quantization.}"
    )
    print("\\end{table}")


def _print_accuracy_row(label, df_cfg):
    row = [label]
    for metric, dig in [("mean_disorientation", 2), ("fraction_above_3deg", 1)]:
        for noise_id in [1, 5, 10]:
            triplet = []
            for dt in ["FP32", "FP16", "INT8"]:
                v = df_cfg.loc[
                    (df_cfg["dataset_id"] == noise_id) & (df_cfg["dtype"] == dt), metric
                ]
                if len(v):
                    val = float(v.iloc[0])
                    if metric == "fraction_above_3deg":
                        val *= 100.0
                    triplet.append(_fmt_float(val, dig))
                else:
                    triplet.append("—")
            row.append(" ".join(triplet))
    print(" & ".join(row) + " \\\\")


def create_knn_table_with_resolution_invariant_caption(data):
    """
    Show only KNN throughputs in the table.
    Caption lists single, resolution-invariant values for:
      - Dictionary Interpolation (dict_sim_pps)
      - PCA (pca_pps)
      - Projection per PCA-N (dict_proj_pps)
      - Spherical FFT (L vs pps)
    """
    df = pd.DataFrame(
        {
            "dataset_id": data["dataset_id"],
            "method": data["method"],
            "dict_resolution": data["dict_resolution"],
            "dict_size": data.get("dict_size", [np.nan] * len(data["dataset_id"])),
            "pca_components": data["pca_components"],
            "dtype": data["dtype"],
            "dict_sim_pps": data["dict_sim_pps"],
            "pca_pps": data["pca_pps"],
            "dict_proj_pps": data["dict_proj_pps"],
            "knn_pps": data["knn_pps"],
            "L_value": data["L_value"],
            "spherical_fft_pps": data["spherical_fft_pps"],
        }
    )

    df_np = df[df["method"] != "Spherical"]

    # Find resolution-invariant component values (take the first non-null across dataset)
    def first_non_null(series):
        s = series.dropna()
        return s.iloc[0] if len(s) else None

    dict_interp_val = first_non_null(df_np["dict_sim_pps"])
    pca_val = first_non_null(df_np["pca_pps"])

    # PCA-N set actually present (>0)
    pca_ns = [
        int(n)
        for n in _sorted_unique(df_np.loc[df_np["method"] == "PCA", "pca_components"])
        if n and n > 0
    ]

    # One projection value per PCA-N (resolution-invariant)
    proj_vals = {}
    for n in pca_ns:
        sub = df_np[(df_np["method"] == "PCA") & (df_np["pca_components"] == n)]
        proj_vals[n] = first_non_null(sub["dict_proj_pps"])

    # Spherical FFT list (L vs pps)
    df_sph = df[df["method"] == "Spherical"]
    spherical_pairs = []
    if len(df_sph):
        for L in _sorted_unique(df_sph["L_value"]):
            if L and L > 0:
                pps = df_sph.loc[df_sph["L_value"] == L, "spherical_fft_pps"]
                spherical_pairs.append((int(L), first_non_null(pps)))

    # Build caption text (single, invariant values)
    proj_txt = (
        ", ".join([f"PCA-{n} Proj={_fmt_rate(proj_vals[n])}" for n in pca_ns])
        if pca_ns
        else "—"
    )
    sph_txt = (
        "; ".join([f"L={L}: {_fmt_rate(pps)}" for (L, pps) in spherical_pairs])
        if spherical_pairs
        else "—"
    )

    caption_text = (
        "\\textbf{Components (resolution-invariant):} "
        f"Dictionary Interpolation={_fmt_rate(dict_interp_val)}, "
        f"PCA={_fmt_rate(pca_val)}, "
        f"{proj_txt}. "
        "\\textbf{Spherical FFT (pps):} " + sph_txt + "."
    )

    # ---------- KNN table ----------
    # Column spec: l then one r for DI and one r per PCA-N
    knn_cols_spec = " ".join(["r"] * (1 + len(pca_ns)))  # DI + PCA-Ns

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{KNN Throughputs (patterns/second). " + caption_text + "}")
    print("\\label{tab:ebsd_knn_only}")
    print("\\small")
    print(f"\\begin{{tabular}}{{l {knn_cols_spec}}}")
    print("\\toprule")
    # Header row: blank left cell, then DI + PCA-N headers
    method_headers = ["\\textbf{DI}"] + [f"\\textbf{{PCA-{n}}}" for n in pca_ns]
    print(" & " + " & ".join(method_headers) + " \\\\")
    print("\\midrule")

    for res in _sorted_unique(df_np["dict_resolution"]):
        df_res = df_np[df_np["dict_resolution"] == res]
        ds = df_res["dict_size"].dropna()
        dict_size = int(ds.iloc[0]) if len(ds) else None
        print(
            f"\\multicolumn{{{2+len(pca_ns)}}}{{l}}{{\\textit{{{_header_with_dict_size(res, dict_size)}}}}} \\\\"
        )

        for dt in ["FP32", "FP16", "INT8"]:
            # DI value
            di_sub = df_res[(df_res["method"] == "DI") & (df_res["dtype"] == dt)]
            di_val = _fmt_rate(di_sub["knn_pps"].mean()) if len(di_sub) else "—"
            # PCA-N values
            pvals = []
            for n in pca_ns:
                sub = df_res[
                    (df_res["method"] == "PCA")
                    & (df_res["pca_components"] == n)
                    & (df_res["dtype"] == dt)
                ]
                pvals.append(_fmt_rate(sub["knn_pps"].mean()) if len(sub) else "—")
            print(f"{dt} & {di_val} & " + " & ".join(pvals) + " \\\\")
        if res != _sorted_unique(df_np["dict_resolution"])[-1]:
            print("\\addlinespace[0.25em]")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_summary_statistics(data):
    df = pd.DataFrame(
        {
            "method": data["method"],
            "dict_resolution": data["dict_resolution"],
            "pca_components": data["pca_components"],
            "dtype": data["dtype"],
            "knn_pps": data["knn_pps"],
            "mean_disorientation": data["mean_disorientation"],
        }
    )
    df = df[df["method"] != "Spherical"]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if len(df):
        print("\nKNN Throughput Ranges:")
        for method in ["DI", "PCA"]:
            dd = df[df["method"] == method]
            if len(dd):
                print(
                    f"  {method}: {dd['knn_pps'].min():,.0f} - {dd['knn_pps'].max():,.0f} pps"
                )

        print("\nAccuracy Ranges (Mean Disorientation):")
        print(
            f"  All methods: {df['mean_disorientation'].min():.2f}° - {df['mean_disorientation'].max():.2f}°"
        )

        best_knn = df.loc[df["knn_pps"].idxmax()]
        label_knn = (
            f"{best_knn['method']}-{int(best_knn['pca_components'])}"
            if best_knn["method"] == "PCA" and best_knn["pca_components"] > 0
            else best_knn["method"]
        )
        print(
            f"\nBest KNN: {label_knn} {best_knn['dtype']} @ {best_knn['dict_resolution']}° → {best_knn['knn_pps']:,.0f} pps"
        )

        best_acc = df.loc[df["mean_disorientation"].idxmin()]
        label_acc = (
            f"{best_acc['method']}-{int(best_acc['pca_components'])}"
            if best_acc["method"] == "PCA" and best_acc["pca_components"] > 0
            else best_acc["method"]
        )
        print(
            f"Best accuracy: {label_acc} {best_acc['dtype']} @ {best_acc['dict_resolution']}° → {best_acc['mean_disorientation']:.2f}°"
        )
    else:
        print("No DI/PCA rows available.")


# --------------------------- Main ---------------------------


def main():
    results_file = Path("benchmark_results/benchmark_dictionary.npy")

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run the benchmark script first.")
        return

    print("\n" + "=" * 80)
    print("LOADING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"File: {results_file}")

    data = load_results(results_file)

    print(f"Total entries: {len(data['dataset_id'])}")
    print(f"Methods: {set(data['method'])}")
    print(f"Resolutions: {set(data['dict_resolution'])}")
    print(f"Data types: {set(data['dtype'])}")

    # Console summary sanity check
    print_summary_statistics(data)

    # LaTeX
    print("\n" + "=" * 80)
    print("LATEX TABLES")
    print("=" * 80)

    print("\n% Table 1: Accuracy Comparison")
    create_accuracy_table(data)

    print(
        "\n\n% Table 2: KNN Throughput (components moved to caption; resolution-invariant)"
    )
    create_knn_table_with_resolution_invariant_caption(data)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

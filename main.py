import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pythermalcomfort.models import pmv_ppd, set_tmp, two_nodes
from pythermalcomfort.utilities import v_relative, clo_dynamic
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import warnings
from scipy import stats
import matplotlib.style
import matplotlib as mpl
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
import psychrolib
import matplotlib.patches as patches

warnings.filterwarnings("ignore")


psychrolib.SetUnitSystem(psychrolib.SI)


def save_var_latex(key, value):
    import csv

    dict_var = {}

    file_path = "Manuscript/Variables/results.csv"

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")


def preprocess_comfort_db_data(read_csv):

    if read_csv:
        # fixme convert to pickle
        return pd.read_csv(r"./Data/DBII_preprocessed.csv")

    # import DB II data
    df_db2 = pd.read_csv(r"./Data/DatabaseI&II_20180703.csv", encoding="unicode_escape")

    rename_cols = {
        "Thermal sensation": "TSV",
        "Air temperature (°C)": "Ta",
        "Radiant temperature (°C)": "Tr",
        "Relative humidity (%)": "Rh",
        "Air velocity (m/s)": "V",
        "Operative temperature (°C)": "To",
    }
    df_db2.rename(columns=rename_cols, inplace=True)

    # remove entries which do not have all the data
    tot_entries = df_db2.shape[0]

    # I am keeping entries without Tr otherwise I would lose too many data points
    df_db2 = (
        df_db2.dropna(subset=["Ta", "V", "Rh", "Met", "Clo", "TSV"])
        .reset_index(drop=True)
        .copy()
    )

    def entries_removed(initial, new_df):
        print(
            f"Entries: {new_df.shape[0]} -- Entries removed: "
            f"{initial - new_df.shape[0]}"
        )

    entries_removed(tot_entries, df_db2)

    df_db2 = df_db2.reset_index(drop=True).copy()

    # calculate PMV ISO, PMV SET and PMV*
    results = []
    for ix, row in df_db2.iterrows():

        # remove entries outside applicability limits
        if row["Tr"] != row["Tr"]:
            if row["To"] != row["To"]:
                row["Tr"] = row["Ta"]
            else:
                row["Tr"] = 2 * row["To"] - row["Ta"]

        vr = v_relative(v=row["V"], met=row["Met"])
        clo_d = clo_dynamic(clo=row["Clo"], met=row["Met"])
        pmv_ashrae = pmv_ppd(
            tdb=row["Ta"],
            tr=row["Tr"],
            vr=vr,
            rh=row["Rh"],
            met=row["Met"],
            clo=clo_d,
            standard="ashrae",
        )["pmv"]
        pmv_iso = pmv_ppd(
            tdb=row["Ta"],
            tr=row["Tr"],
            vr=vr,
            rh=row["Rh"],
            met=row["Met"],
            clo=clo_d,
            standard="iso",
        )["pmv"]
        # pmv_set = two_nodes(
        #     tdb=row["Ta"],
        #     tr=row["Tr"],
        #     v=row["V"],
        #     rh=row["Rh"],
        #     met=row["Met"],
        #     clo=row["Clo"],
        # )["pmv_set"]
        # set_py = set_tmp(
        #     tdb=row["Ta"],
        #     tr=row["Tr"],
        #     v=row["V"],
        #     rh=row["Rh"],
        #     met=row["Met"],
        #     clo=row["Clo"],
        # )
        results.append(
            {
                "index": ix,
                "pmv_ashrae": pmv_ashrae,
                "pmv_iso": pmv_iso,
                "Tr_est": row["Tr"]
                # "pmv_set": pmv_set,
                # "set_py": set_py,
            }
        )

    df_ = pd.DataFrame(results)

    df_ = pd.concat([df_db2, df_], axis=1, sort=False)

    df_.to_csv(r"./Data/DBII_preprocessed.csv", index=False)

    return df_


def filter_data(df_):

    # remove entries outside the Standards' applicability limits
    df_ = df_[
        (df_["Ta"] >= applicability_limits["Ta"][0])
        & (df_["Ta"] <= applicability_limits["Ta"][1])
    ]
    df_ = df_[
        (df_["Tr_est"] >= applicability_limits["Tr"][0])
        & (df_["Tr_est"] <= applicability_limits["Tr"][1])
    ]
    df_ = df_[
        (df_["V"] >= applicability_limits["V"][0])
        & (df_["V"] <= applicability_limits["V"][1])
    ]
    df_ = df_[
        (df_["Clo"] >= applicability_limits["Clo"][0])
        & (df_["Clo"] <= applicability_limits["Clo"][1])
    ]
    df_ = df_[
        (df_["Met"] >= applicability_limits["Met"][0])
        & (df_["Met"] <= applicability_limits["Met"][1])
    ]
    df_ = df_[
        (df_["pmv_iso"] > applicability_limits["PMV"][0])
        & (df_["pmv_iso"] < applicability_limits["PMV"][1])
    ]
    df_ = df_[
        (df_["pmv_ashrae"] > applicability_limits["PMV"][0])
        & (df_["pmv_ashrae"] < applicability_limits["PMV"][1])
    ]
    df_ = df_[
        (df_["TSV"] > applicability_limits["TSV"][0])
        & (df_["TSV"] < applicability_limits["TSV"][1])
    ]
    pa_arr = []
    for i, row in df_.iterrows():
        pa_arr.append(psychrolib.GetVapPresFromRelHum(row["Ta"], row["Rh"] / 100))
    df_["Pa"] = pa_arr
    df_ = df_[
        (df_["Pa"] > applicability_limits["Pa"][0])
        & (df_["Pa"] < applicability_limits["Pa"][1])
    ]

    return df_


def calculate_new_indices(df_):

    df_["pmv_iso_round"] = df_["pmv_iso"].round()
    df_["pmv_ashrae_round"] = df_["pmv_ashrae"].round()
    df_["PMV_round"] = df_["PMV"].round()
    df_["TSV_round"] = df_["TSV"].round()
    df_["diff_iso"] = df_[["TSV", "pmv_iso"]].diff(axis=1)["pmv_iso"]
    df_["diff_ash"] = df_[["TSV", "pmv_ashrae"]].diff(axis=1)["pmv_ashrae"]

    # check delta between the PMV I calculated and the one Toby did
    df_["PMV - pmv_iso"] = df_["PMV"] - df_["pmv_iso"]
    df_["PMV - pmv_ashrae"] = df_["PMV"] - df_["pmv_ashrae"]
    # df_["SET - set_py"] = df_["SET"] - df_["set_py"]
    df_["TSV_round - pmv_ashrae_round"] = df_["TSV"] - df_["pmv_ashrae_round"]

    return df_


def bar_chart(data, ind="tsv", show_per=True, figletter=False):
    if data.V.min() != 0:
        f, axs = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(8.0, 4.1)
        )
    else:
        f, axs = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(8.0, 4)
        )

    for ix, model in enumerate(["pmv_iso_round", "pmv_ashrae_round"]):
        if ind == "pmv":
            _df = data.groupby(["TSV_round", model])[model].count().unstack("TSV_round")
            x = model
            x_label = "PMV"
            axs[ix].set(xlabel=map_model_name[model], ylabel="Percentage [%]")
            # conside the special case I am only including data with TSV = 0
            if _df.columns == [0.0]:
                for index in _df.index:
                    if index in _df.columns:
                        continue
                    _df[index] = 0
                _df = _df[_df.index.sort_values()]

        else:
            _df = data.groupby(["TSV_round", model])["TSV_round"].count().unstack(model)
            x = "TSV_round"
            x_label = "TSV"
            axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
            if data.V.min() == 0:
                axs[ix].set_title(map_model_name[model], y=0.9)
        df_total = _df.sum(axis=1)
        df_rel = _df.div(df_total, 0) * 100
        colors = [
            (33 / 255, 102 / 255, 172 / 255),
            (103 / 255, 169 / 255, 207 / 255),
            (209 / 255, 229 / 255, 240 / 255),
            (153 / 255, 213 / 255, 148 / 255),
            (253 / 255, 219 / 255, 199 / 255),
            (239 / 255, 138 / 255, 98 / 255),
            (178 / 255, 24 / 255, 43 / 255),
        ]
        df_plot = df_rel.reset_index()
        df_plot[x] = pd.to_numeric(df_plot[x], downcast="integer")
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        hist = df_plot.plot(
            x=x,
            kind="bar",
            stacked=True,
            mark_right=True,
            width=0.95,
            rot=0,
            legend=False,
            ax=axs[ix],
            colormap=cmap1,
        )
        if ind == "pmv":
            axs[ix].set(xlabel=map_model_name[model], ylabel="Percentage [%]")
        else:
            axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
            if data.V.min() == 0:
                axs[ix].set_title(map_model_name[model], y=1.1)
                axs[ix].set_xticklabels("")
                axs[ix].set_xlabel("")
            else:
                axs[ix].set_xticklabels(
                    [
                        "Cold",
                        "Cool",
                        "Sl. Cool",
                        "Neutral",
                        "Sl. Warm",
                        "Warm",
                        "Hot",
                    ],
                    Fontsize=9,
                )
        sns.despine(ax=axs[ix], left=True, bottom=True)

        # show accuracy
        df_acc = df_rel[df_rel.index.isin(df_rel.columns)]
        df_acc = df_acc[df_acc.index]
        diagonal = pd.Series(np.diag(df_acc), index=df_acc.index)

        axs[ix].grid(axis="x")

        for ix_s, value in enumerate(diagonal):
            if value != value:
                value = 0
            axs[ix].text(
                ix_s, 110, f"{value:.0f}%", va="center", ha="center", fontsize=9
            )

        # show surveys counts
        values = data.groupby([x])[x].count()
        for ix_s, value in enumerate(values):
            axs[ix].text(
                ix_s, 105, f"{value:.0f}", va="center", ha="center", fontsize=9
            )

        # add percentages
        if show_per:
            for index, row in df_rel.fillna(0).reset_index(drop=True).iterrows():
                cum_sum, el = 0, 0
                for ixe, el in enumerate(row):
                    if el > 7:
                        axs[ix].text(
                            index,
                            cum_sum + el / 2,
                            f"{el:.0f}%",
                            va="center",
                            ha="center",
                        )
                    cum_sum += el

    # if data.V.min() != 0:
    # sm = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(vmin=-3.5, vmax=+3.5))
    # cmap = mpl.cm.rainbow
    # bounds = np.linspace(-3.5, 3.5, 8)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # sm = plt.cm.get_cmap("rainbow", 5)
    # cbar = plt.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap1),
    #     ticks=np.linspace(-3, 3, 7),
    #     ax=axs,
    #     orientation="horizontal",
    #     aspect=70,
    # )
    # cbar.ax.set_xticklabels(
    #     [
    #         "-3",
    #         "-2",
    #         "-1",
    #         "0",
    #         "1",
    #         "2",
    #         "3",
    #     ]
    # )
    # cbar.outline.set_visible(False)
    # cbar.set_label("PMV")

    if figletter:
        plt.gcf().text(0.05, 0.95, f"{figletter})", weight="bold")

    plt.savefig(f"./Manuscript/Figures/bar_plot_{ind}_Vmin_{data.V.min()}.png", dpi=300)


def legend_pmv():
    f, ax = plt.subplots()

    colors = [
        (33 / 255, 102 / 255, 172 / 255),
        (103 / 255, 169 / 255, 207 / 255),
        (209 / 255, 229 / 255, 240 / 255),
        (153 / 255, 213 / 255, 148 / 255),
        (253 / 255, 219 / 255, 199 / 255),
        (239 / 255, 138 / 255, 98 / 255),
        (178 / 255, 24 / 255, 43 / 255),
    ]

    ax.legend(
        handles=[
            patches.Patch(color=colors[ix], label=str(x))
            for ix, x in enumerate(range(-3, 4))
        ],
        frameon=False,
        mode="expand",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        ncol=7,
    )
    f.text(0.05, 0.95, "PMV = ", va="center", ha="left")
    ax.grid(False)


def distributions_pmv(v_lower=False):
    # check difference in distribution for V > 0.1
    variables = ["pmv_iso", "pmv_ashrae", "TSV"]
    f, axs = plt.subplots(len(variables), 1, sharex=True)
    if v_lower:
        data = df[df["V"] > 0.1]
    else:
        data = df.copy()
    for ix, var in enumerate(variables):
        sns.histplot(data=data, x=var, kde=True, stat="density", ax=axs[ix])
        axs[ix].set(title=var)
    plt.tight_layout()


def scatter_plot_flip_x(data):
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    sns.regplot(
        data=data,
        x="TSV",
        y="pmv_iso",
        ax=axs[0],
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=data["TSV"], y=data["pmv_iso"]
    )
    print("TSV x-axis:", slope, intercept, r_value, p_value, std_err)
    sns.regplot(
        data=data,
        y="TSV",
        x="pmv_iso",
        ax=axs[1],
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=data["pmv_iso"], y=data["TSV"]
    )
    print("ISO x-axis:", slope, intercept, r_value, p_value, std_err)


def scatter_plot(data, ind="tsv", x_jitter=0):
    f, axs = plt.subplots(1, 2, constrained_layout=True)

    for ix, model in enumerate(["pmv_iso", "pmv_ashrae"]):
        if ind == "pmv":
            sns.regplot(data=df, x=df[model], y="TSV", ax=axs[ix], x_jitter=0.1)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                y=df["TSV"], x=df[model]
            )
        else:
            axs[ix].scatter(
                # data=data,
                y=data[model],
                x=data["TSV"],
                alpha=0.5,
                s=5,
                c="lightgray",
            )
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x=data["TSV"], y=data[model]
            )

        # mean absolute error
        r2 = r2_score(data["TSV"], data[model])
        mae = mean_absolute_error(data["TSV"], data[model])

        axs[ix].set(ylim=(-3.5, 3.5))

        axs[ix].text(
            0.5,
            0.85,
            f"{map_model_name[model]}={slope:.2}*TSV{intercept:.2}\n"
            + r"R$^2$"
            + f"={r_value**2:.2}, MAE={mae:.2}",
            transform=axs[ix].transAxes,
            ha="center",
            va="center",
        )

        color = "#FDB515"
        if model == "pmv_iso":
            color = "#3B7EA1"

        axs[ix].plot(data["TSV"], intercept + data["TSV"] * slope, color=color)
        axs[ix].set(ylabel=map_model_name[model])
        axs[ix].set_xticks(
            np.arange(-3, 4, step=1),
        )
        axs[ix].set_xticklabels(
            [
                "Cold",
                "Cool",
                "Sl. Cool",
                "Neutral",
                "Sl. Warm",
                "Warm",
                "Hot",
            ],
            Fontsize=9,
        )
        sns.despine(bottom=True, left=True)
        axs[ix].grid(axis="x")

    plt.tight_layout()
    plt.savefig("./Manuscript/Figures/scatter_tsv_pmv.png", dpi=300)


def plot_error_prediction(data):
    f, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(8.0, 6))

    _df = (
        data[["TSV_round", "diff_iso", "diff_ash"]]
        .set_index("TSV_round")
        .stack()
        .reset_index()
    )
    _df.columns = ["TSV", "model", "delta"]
    _df["model"] = _df["model"].map({"diff_iso": "PMV", "diff_ash": r"PMV$_{CE}$"})
    _df["TSV"] = pd.to_numeric(_df["TSV"], downcast="integer")
    sns.violinplot(
        data=_df,
        x="TSV",
        y="delta",
        size="data",
        split=True,
        hue="model",
        inner="quartile",
        palette=["#3B7EA1", "#FDB515"],
    )
    acceptable_error = 0.5
    axs.fill_between(
        [-0.5, len(_df["TSV"].unique()) - 0.5],
        acceptable_error,
        -acceptable_error,
        color="k",
        alpha=0.3,
        edgecolor="b",
        linewidth=0.0,
    )
    axs.set(yticks=([-6, -4, -2, -0.5, 0, 0.5, 2, 4]))

    axs.set_xticklabels(
        [
            "Cold",
            "Cool",
            "Sl. Cool",
            "Neutral",
            "Sl. Warm",
            "Warm",
            "Hot",
        ],
    )
    for label in axs.xaxis.get_majorticklabels():
        label.set_y(+0.05)

    axs.set(ylabel="PMV - TSV")
    sns.despine(bottom=True, left=True)
    # plt.legend(frameon=False, loc=1)
    # leg = axs.legend()
    # leg.get_frame().set_edgecolor("b")
    # leg.get_frame().set_linewidth(0.0)
    axs.legend(
        handles=[
            patches.Patch(color="#3B7EA1", label="PMV"),
            patches.Patch(color="#FDB515", label="PMV$_{CE}$"),
        ],
        frameon=False,
        loc=1,
    )

    # t-test
    for ix, tsv_vote in enumerate(_df["TSV"].sort_values().unique()):
        sample_1 = _df[(_df["TSV"] == tsv_vote) & (_df["model"] == "PMV")]["delta"]
        sample_2 = _df[(_df["TSV"] == tsv_vote) & (_df["model"] == "PMV$_{CE}$")][
            "delta"
        ]
        p = round(stats.ttest_ind(sample_1, sample_2).pvalue, 3)
        if p < 0.01:
            text_p = r"$p$ < 0.01"
        elif p <= 0.05:
            text_p = r"$p$ = " + str(p)
        else:
            text_p = r"$p$ = " + str(round(p, 1))
        if ix < 3:
            axs.text(ix, -2.5 - ix / 2, text_p, ha="center", va="center")
        else:
            axs.text(ix, 5.2 - ix / 2, text_p, ha="center", va="center")
        perc_1 = round(
            (sample_1.abs() <= acceptable_error).sum() / sample_1.shape[0] * 100
        )
        perc_2 = round(
            (sample_2.abs() <= acceptable_error).sum() / sample_1.shape[0] * 100
        )
        perc_1_1 = round((sample_1.abs() <= 1).sum() / sample_1.shape[0] * 100)
        perc_2_2 = round((sample_2.abs() <= 1).sum() / sample_1.shape[0] * 100)
        y_text = 1.5
        rad = 0.5
        if ix < 2 or ix > 3:
            axs.text(ix + 0.15, 0, f"{perc_2}%", va="center")
            axs.text(ix - 0.15, 0, f"{perc_1}%", ha="right", va="center")
        if ix == 2 or ix == 3:
            if ix == 2:
                y_text = -1.5
                rad = -0.5
            # axs.text(ix + 0.1, y_text, f"{perc_2}%", va="center")
            # axs.text(ix - 0.1, y_text, f"{perc_1}%", ha="right", va="center")
            axs.annotate(
                f"{perc_1}%",
                xy=(ix - 0.3, 0),
                xycoords="data",
                textcoords="data",
                xytext=(ix - 0.1, y_text),
                va="center",
                ha="right",
                arrowprops=dict(
                    arrowstyle="->", connectionstyle=f"arc3,rad={rad}", fc="k", ec="k"
                ),
            )
            axs.annotate(
                f"{perc_2}%",
                xy=(ix + 0.3, 0),
                xycoords="data",
                textcoords="data",
                xytext=(ix + 0.1, y_text),
                va="center",
                arrowprops=dict(
                    arrowstyle="->", connectionstyle=f"arc3,rad={-rad}", fc="k", ec="k"
                ),
            )

    plt.savefig(
        f"./Manuscript/Figures/prediction_error_Vmin_{data.V.min()}.png", dpi=300
    )


def plot_distribution_variable():
    f, axs = plt.subplots(1, 6, constrained_layout=True, figsize=(8, 3))

    for ix, var in enumerate(["Ta", "Tr", "V", "Clo", "Met", "Rh"]):
        sns.boxenplot(y=var, data=df, ax=axs[ix], color="lightgray")
        axs[ix].set(
            ylabel="",
            xlabel=f"{var_names[var]} ({var_units[var]})",
            ylim=(applicability_limits[var][0], applicability_limits[var][1]),
        )
        if var == "Ta":
            axs[ix].set(
                ylim=(applicability_limits["Tr"][0], applicability_limits["Tr"][1]),
            )
        if var == "Clo":
            axs[ix].set(yticks=(np.arange(0, 1.8, 0.3)))
    sns.despine(bottom=True, left=True)
    plt.savefig("./Manuscript/Figures/dist_input_data.png", dpi=300)


if __name__ == "__main__":

    plt.close("all")
    sns.set_context("paper")
    mpl.rcParams["figure.figsize"] = [8.0, 3.5]
    sns.set_theme(style="whitegrid")
    map_model_name = {
        "pmv_iso": r"PMV",
        "pmv_iso_round": r"PMV",
        "pmv_ashrae_round": r"PMV$_{CE}$",
        "pmv_ashrae": r"PMV$_{CE}$",
    }
    applicability_limits = {
        "Ta": [10, 30],
        "Tr": [10, 40],
        "V": [0, 1],
        "Clo": [0, 1.5],
        "Met": [1, 4],
        "PMV": [-3.5, 3.5],
        "TSV": [-3.5, 3.5],
        "Rh": [0, 100],
        "Pa": [0, 2700],
    }

    var_names = {
        "Ta": r"$t_{db}$",
        "Tr": r"$\overline{t_{r}}$",
        "V": r"$V$",
        "Rh": r"$RH$",
        "Clo": r"$I_{cl}$",
        "Met": r"$M$",
    }

    var_units = {
        "Ta": r"$^{\circ}$C",
        "Tr": r"$^{\circ}$C",
        "V": r"m/s",
        "Rh": r"%",
        "Clo": r"clo",
        "Met": r"met",
    }

    # import data
    df = preprocess_comfort_db_data(read_csv=True)

    df = filter_data(df_=df)
    df = calculate_new_indices(df_=df)

    save_var_latex("Tot usable surveys", df.shape[0])
    save_var_latex("Tot surveys V higher 0.1", df[df.V > 0.1].shape[0])

    # accuracies calculation
    for limit in [3, 2, 1]:
        data = df[df["TSV_round"].abs() <= limit]
        data_iso = data[df["pmv_iso_round"].abs() <= limit]
        data_ash = data[df["pmv_ashrae_round"].abs() <= limit]

        # check
        print(data_iso["pmv_iso_round"].sort_values().unique())
        print(data_ash["pmv_ashrae_round"].sort_values().unique())
        print(data["TSV_round"].sort_values().unique())

        acc_iso = (
            data_iso[data_iso["TSV_round"] == data_iso["pmv_iso_round"]].shape[0]
            / data_iso.shape[0]
        )
        save_var_latex(f"Overall PMV ISO accuracy - limit {limit}", int(acc_iso * 100))
        acc_ash = (
            data_ash[data_ash["TSV_round"] == data_ash["pmv_ashrae_round"]].shape[0]
            / data_ash.shape[0]
        )
        save_var_latex(
            f"Overall PMV ASHRAE accuracy - limit {limit}", int(acc_ash * 100)
        )

    def accuracy_varying_v(v):
        data = df[df["V"] > v]
        data = data[data["TSV_round"] == 0]
        print(f"{v=}")
        print(round(data[data["pmv_iso_round"] == 0].shape[0] / data.shape[0] * 100))
        print(round(data[data["pmv_ashrae_round"] == 0].shape[0] / data.shape[0] * 100))
        print(data.shape[0])

    accuracy_varying_v(0.1)
    accuracy_varying_v(0.2)
    accuracy_varying_v(0.4)
    accuracy_varying_v(0.6)

    print(df[~df.TSV.isin(range(-3, 3))].shape[0])
    df_tpv = df.dropna(subset=["Thermal preference"])
    print(
        df_tpv[
            (df_tpv.TSV_round.isin([-1, 1]))
            & (df_tpv["Thermal preference"] == "no change")
        ].shape[0]
    )
    print(df_tpv.shape[0])

    # Figure 1
    plot_distribution_variable()

    # Figure 2
    # bar_chart(ind="pmv")
    bar_chart(data=df, ind="tsv", show_per=False, figletter="a")
    # bar_chart(data=df[df.TSV == 0], ind="pmv")
    bar_chart(data=df[df.V > 0.1], ind="tsv", show_per=False, figletter="b")
    # bar_chart(data=df[(df.V > 0.2)], ind="tsv")
    # bar_chart(data=df[(df.V > 0.4)], ind="tsv")
    # bar_chart(data=df[(df.TSV == 0) & (df.V > 0.3)], ind="tsv")
    # bar_chart(data=df[(df.TSV == 0) & (df.V > 0.6)], ind="tsv")
    # bar_chart(data=df[(df.TSV == 0) & (df.V > 0.9)], ind="tsv")
    legend_pmv()

    # Figure 3
    plot_error_prediction(data=df[df.V > 0.1])

    # Figure 4
    scatter_plot(data=df[df.V > 0.1], ind="tsv")
    # scatter_plot(data=df, ind="pmv")
    # scatter_plot(data=df, ind="tsv")

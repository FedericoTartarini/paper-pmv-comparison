import matplotlib as mpl

# mpl.use("Qt5Agg")  # or can use 'TkAgg', whatever you have/prefer

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import matplotlib as mpl
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
import psychrolib
import matplotlib.patches as patches
from pythermalcomfort.models import two_nodes, athb, pmv, pmv_ppd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit, logit
from statsmodels.tools import add_constant
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import f1_score
import math

warnings.filterwarnings("ignore")


psychrolib.SetUnitSystem(psychrolib.SI)


palette_tp = [
    "#06A6EE",
    "#31CAA8",
    "#FF412C",
]
palette_tp = [
    "#4F96FF",
    "#60E693",
    "#FF362B",
]
palette_tsv = [
    "#2C45FE",
    "#4F96FF",
    "#6CDFFF",
    "#60E693",
    "#FFDF6B",
    "#FFB36B",
    "#FF362B",
]

# sns.palplot(palette_tsv)

palette_primary = [
    "#FFBA22",
    "#FF2380",
    "#6600AB",
    "#FF7BB3",
    "#003CAB",
    "#83DFCB",
    "#C20800",
    "#C57FFF",
    "#FF8D80",
    "#6ACAF5",
    "#006B46",
    "#FF6756",
    "#9F29FF",
    "#8001D8",
    "#FF4F99",
    "#006BDB",
    "#5AD5B9",
    "#E8281E",
    "#B254FF",
    "#149678",
    "#FFE146",
]


def save_var_latex(key, value, units=False, round_var=False):
    import csv

    dict_var = {}

    file_path = "Manuscript/src/mydata.dat"

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    if round_var:
        value = round(value, round_var)

    if units:
        dict_var[key] = f"\\qty{{{value}}}{{{units}}}"
    else:
        dict_var[key] = f"\\num{{{value}}}"

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")


def importing_filtering_processing(load_preprocessed=False):

    df_ = pd.read_csv(r"./Data/db_measurements_v2.1.0.csv.gz", compression="gzip")

    save_var_latex("entries_db_all", df_.shape[0])

    # entries without ta, rh, v, clo, met
    df_valid_input = df_.dropna(subset=["ta", "vel", "rh", "met", "clo"])
    df_valid_input_no_tr = df_valid_input.dropna(subset=["tr"]).shape[0]
    save_var_latex(
        "entries_db_valid",
        int(100 - df_valid_input.shape[0] / df_.shape[0] * 100),
        "\\percent",
    )
    save_var_latex(
        "entries_db_valid_no_tr",
        int(100 - df_valid_input_no_tr / df_valid_input.shape[0] * 100),
        "\\percent",
    )

    if load_preprocessed:
        return pd.read_pickle(r"./Data/db_analysis.pkl.gz", compression="gzip")

    pa_arr = []
    for i, row in df_.iterrows():
        pa_arr.append(psychrolib.GetVapPresFromRelHum(row["ta"], row["rh"] / 100))

    df_["pa"] = pa_arr

    # remove entries outside the Standards' applicability limits
    for key in applicability_limits.keys():
        if "pmv" in key:
            continue
        df_ = df_[
            (df_[key] >= applicability_limits[key][0])
            & (df_[key] <= applicability_limits[key][1])
        ]

    two_nodes_results = two_nodes(
        tdb=df_.ta,
        tr=df_.tr,
        v=df_.vel,
        rh=df_.rh,
        met=df_.met,
        clo=df_.clo,
    )

    df_["pmv_gagge"] = two_nodes_results["pmv_gagge"]
    df_["pmv_set"] = two_nodes_results["pmv_set"]

    # estimate thermal sensation using toby's model
    df_["pmv_toby"] = list(
        pd.cut(
            df_["ta"],
            [-90, 15, 18, 20, 25, 27, 30, 90],
            labels=[-3, -2, -1, 0, 1, 2, 3],
        )
    )

    df_["athb"] = athb(
        tdb=df_.ta,
        tr=df_.tr,
        vr=df_.vel_r,
        rh=df_.rh,
        met=df_.met,
        t_running_mean=df_.t_mot_isd,
    )

    for key in applicability_limits.keys():
        if "pmv" in key:
            df_ = df_[
                (df_[key] >= applicability_limits[key][0])
                & (df_[key] <= applicability_limits[key][1])
            ]

    # calculate rounded variables and differences
    for model in models_to_test:
        rounded_col = f"{model}_round"
        diff_col = f"diff_ts_{model}"
        df_[rounded_col] = df_[model].round()
        df_[diff_col] = df_[["thermal_sensation", model]].diff(axis=1)[model]
        # calculate the heat balance value
        if model != "athb":
            df_[f"{model}_hb"] = df_[model] / (
                0.303 * np.exp(-0.036 * df_["met"] * 58.15) + 0.028
            )
        else:
            met_adapted = df_["met"] - (0.234 * df_["t_mot_isd"]) / 58.2
            df_[f"{model}_hb"] = df_[model] / (
                0.303 * np.exp(-0.036 * met_adapted * 58.15) + 0.028
            )

    df_["thermal_sensation_round"] = df_["thermal_sensation"].round()
    df_["thermal_sensation_round - pmv_ce_round"] = (
        df_["thermal_sensation"] - df_["pmv_ce_round"]
    )

    # estimate thermal sensation as a function of heat balance, met and clothing
    for model in models_to_test[:-1]:
        df_reg = df_[
            [f"{model}_hb", "met", "clo", "thermal_sensation_round", "record_id"]
        ].dropna()
        clf = LogisticRegression(random_state=0, class_weight="balanced").fit(
            df_reg[[f"{model}_hb", "met", "clo"]], df_reg["thermal_sensation_round"]
        )
        df_reg[f"lr_hb_{model}"] = clf.predict(df_reg[[f"{model}_hb", "met", "clo"]])
        df_ = df_.merge(
            df_reg[[f"lr_hb_{model}", "record_id"]], on="record_id", how="left"
        )
        df_[f"lr_hb_{model}_round"] = df_[f"lr_hb_{model}"].round()
        df_[f"diff_ts_lr_hb_{model}"] = df_[
            ["thermal_sensation", f"lr_hb_{model}"]
        ].diff(axis=1)[f"lr_hb_{model}"]

    save_var_latex("entries_db_used", df_.shape[0])
    save_var_latex("entries_db_used_v_01", df_[df_.vel > 0.1].shape[0])

    df_.to_pickle(r"./Data/db_analysis.pkl.gz", compression="gzip")

    return df_


def bar_chart(
    data,
    ind="tsv",
    show_per=True,
    figletter=False,
    variables=["pmv_round", "pmv_ce_round"],
):
    if data.vel.min() != 0:
        f, axs = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(8.0, 4.1)
        )
    else:
        f, axs = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(8.0, 4)
        )

    for ix, model in enumerate(variables):
        if ind == "pmv":
            _df = (
                data.groupby(["thermal_sensation_round", model])[model]
                .count()
                .unstack("thermal_sensation_round")
            )
            x = model
            x_label = "PMV"
            axs[ix].set(xlabel=var_names[model], ylabel="Percentage [%]")
            # conside the special case I am only including data with thermal_sensation = 0
            if _df.columns == [0.0]:
                for index in _df.index:
                    if index in _df.columns:
                        continue
                    _df[index] = 0
                _df = _df[_df.index.sort_values()]

        else:
            _df = (
                data.groupby(["thermal_sensation_round", model])[
                    "thermal_sensation_round"
                ]
                .count()
                .unstack(model)
            )
            x = "thermal_sensation_round"
            x_label = "thermal_sensation"
            axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
            if data.vel.min() == 0:
                axs[ix].set_title(var_names[model], y=0.9)
        df_total = _df.sum(axis=1)
        df_rel = _df.div(df_total, 0) * 100
        for col in df_rel.index:
            if col in df_rel.columns:
                continue
            else:
                df_rel[col] = 0
        df_rel = df_rel.reindex(sorted(df_rel.columns), axis=1)
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
            axs[ix].set(xlabel=var_names[model], ylabel="Percentage [%]")
        else:
            axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
            if data.vel.min() == 0:
                axs[ix].set_title(var_names[model], y=1.1)
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

    # if data.vel.min() != 0:
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

    plt.savefig(
        f"./Manuscript/src/figures/bar_plot_{ind}_Vmin_{data.vel.min()}.png", dpi=300
    )


def scatter_plot_flip_x(data):
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    sns.regplot(
        data=data,
        x="thermal_sensation",
        y="pmv",
        ax=axs[0],
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=data["thermal_sensation"], y=data["pmv"]
    )
    print("thermal_sensation x-axis:", slope, intercept, r_value, p_value, std_err)
    sns.regplot(
        data=data,
        y="thermal_sensation",
        x="pmv",
        ax=axs[1],
        scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    )
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=data["pmv"], y=data["thermal_sensation"]
    )
    print("ISO x-axis:", slope, intercept, r_value, p_value, std_err)


def scatter_plot(data, ind="tsv", x_jitter=0):
    f, axs = plt.subplots(1, 2, constrained_layout=True)

    for ix, model in enumerate(["pmv", "pmv_ce"]):
        if ind == "pmv":
            sns.regplot(
                data=df, x=df[model], y="thermal_sensation", ax=axs[ix], x_jitter=0.1
            )
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                y=df["thermal_sensation"], x=df[model]
            )
        else:
            axs[ix].scatter(
                # data=data,
                y=data[model],
                x=data["thermal_sensation"],
                alpha=0.5,
                s=5,
                c="lightgray",
            )
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x=data["thermal_sensation"], y=data[model]
            )

        # mean absolute error
        r2 = r2_score(data["thermal_sensation"], data[model])
        mae = mean_absolute_error(data["TSV"], data[model])

        axs[ix].set(ylim=(-3.5, 3.5))

        axs[ix].text(
            0.5,
            0.85,
            f"{var_names[model]}={slope:.2}*TSV{intercept:.2}\n"
            + r"R$^2$"
            + f"={r_value**2:.2}, MAE={mae:.2}",
            transform=axs[ix].transAxes,
            ha="center",
            va="center",
        )

        color = "#FDB515"
        if model == "pmv":
            color = "#3B7EA1"

        axs[ix].plot(data["TSV"], intercept + data["TSV"] * slope, color=color)
        axs[ix].set(ylabel=var_names[model])
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
    plt.savefig("./Manuscript/src/figures/scatter_tsv_pmv.png", dpi=300)


def plot_error_prediction(data):
    f, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(8.0, 6))

    _df = (
        data[["thermal_sensation_round", "diff_ts_pmv", "diff_ts_pmv_ce"]]
        .set_index("thermal_sensation_round")
        .stack()
        .reset_index()
    )
    _df.columns = ["TSV", "model", "delta"]
    _df["model"] = _df["model"].map(
        {"diff_ts_pmv": "PMV", "diff_ts_pmv_ce": r"PMV$_{CE}$"}
    )
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
        f"./Manuscript/src/figures/prediction_error_Vmin_{data.vel.min()}.png", dpi=300
    )


def plot_distribution_variable():
    f, axs = plt.subplots(1, 6, constrained_layout=True, figsize=(8, 3))

    for ix, var in enumerate(["ta", "tr", "rh", "vel", "clo", "met"]):
        sns.boxenplot(y=var, data=df, ax=axs[ix], color="lightgray")
        axs[ix].set(
            ylabel="",
            xlabel=f"{var_names[var]} ({var_units[var]})",
            ylim=(applicability_limits[var][0], applicability_limits[var][1]),
        )
        desc = df[var].describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975])
        if var == "ta":
            axs[ix].set(
                ylim=(applicability_limits["tr"][0], applicability_limits["tr"][1]),
            )
        if var == "clo":
            axs[ix].set(yticks=(np.arange(0, 1.8, 0.3)))

        desc = desc.round(2)
        if var in ["ta", "tr", "rh"]:
            desc = desc.round(1)

        axs[ix].text(0.5, desc["2.5%"], desc["2.5%"], size="small", c="b", va="center")
        axs[ix].text(
            0.5,
            desc["97.5%"],
            desc["97.5%"],
            size="small",
            c="b",
            va="center",
        )
        axs[ix].text(0.5, desc["50%"], desc["50%"], size="small", c="b", va="center")
    sns.despine(bottom=True, left=True)
    plt.savefig("./Manuscript/src/figures/dist_input_data.png", dpi=300)
    plt.show()

    desc = df[["ta", "tr", "rh", "vel", "clo", "met"]].describe(
        percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]
    )

    # save_var_latex("rh_95_perc_min", desc["ta"]["2.5%"], "\\celsius", round_var=1)
    save_var_latex("rh_95_perc_max", desc["rh"]["97.5%"], "\\percent", round_var=2)
    save_var_latex("v_95_perc_max", desc["vel"]["97.5%"], "\\m\\per\\sec", round_var=2)

    r2 = r2_score(df.ta, df.tr)

    df["const"] = 1
    f, axs = plt.subplots(1, 4, figsize=(8, 3))
    for ix, var in enumerate(["age", "ht", "wt", "t_mot_isd"]):
        sns.violinplot(
            y=var,
            x="const",
            data=df,
            ax=axs[ix],
            hue="gender",
            split=True,
            inner="quartile",
            palette="viridis",
        )
        axs[ix].get_legend().remove()
        axs[ix].set(xlabel=var_names[var], xticks=[], ylabel="")
        desc = df[var].describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).round(1)

        axs[ix].text(0.4, desc["2.5%"], desc["2.5%"], size="small", c="b", va="center")
        axs[ix].text(
            0.4,
            desc["97.5%"],
            desc["97.5%"],
            size="small",
            c="b",
            va="center",
        )
        axs[ix].text(0.4, desc["50%"], desc["50%"], size="small", c="b", va="center")

        if var == "t_mot_isd":
            axs[ix].set(yticks=(np.arange(-30, 50, 10)))
    sns.despine(bottom=True, left=True)
    handles, labels = axs[ix].get_legend_handles_labels()
    f.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0.5, 1.03),
        loc="upper center",
        # borderaxespad=0,
        frameon=False,
        ncol=3,
    )
    plt.tight_layout()
    plt.savefig("./Manuscript/src/figures/dist_other_data.png", dpi=300)
    plt.show()


def plot_bubble_models_vs_tsv():
    # Scatter thermal_sensation vs pmv prediction
    f, axs = plt.subplots(
        1, len(models_to_test), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, model in enumerate(models_to_test):
        # sns.regplot(x="thermal_sensation", y=pmv, data=df,ax=axs[ix], scatter_kws={"s":2, "alpha":0.3}, line_kws={"color":"k"})
        df_plot = df.copy()
        df_plot["ts_binned"] = pd.cut(
            df["thermal_sensation"],
            np.arange(-3.75, 4.25, 0.5),
        )

        df_plot["y_binned"] = pd.cut(df[model], np.arange(-3.75, 4.25, 0.5))
        df_plot = df_plot.groupby(["ts_binned", "y_binned"]).size()
        axs[ix].scatter(
            pd.IntervalIndex(df_plot.index.get_level_values("ts_binned")).mid,
            pd.IntervalIndex(df_plot.index.get_level_values("y_binned")).mid,
            s=df_plot / 20,
            alpha=0.5,
            c="darkgray",
        )
        sns.regplot(
            x="thermal_sensation",
            y=model,
            data=df,
            ax=axs[ix],
            # robust=True,
            ci=None,
            line_kws={"color": "k"},
            scatter_kws={"s": 1},
            scatter=False,
            lowess=True,
        )
        axs[ix].axvline(0, c="darkgray", ls="--")
        axs[ix].axhline(0, c="darkgray", ls="--")
        axs[ix].set(title=var_names[model], ylabel="", xlabel="")
    f.supxlabel(var_names["thermal_sensation"])

    plt.savefig(f"./Manuscript/src/figures/bubble_models_vs_tsv.png", dpi=300)


def plot_bar_tp_by_ts():
    x_var, y_var = "thermal_sensation_round", "thermal_preference"
    save_var_latex(
        f"entries_with_tp",
        df["thermal_preference"].value_counts().sum(),
    )

    df_count = df.groupby(x_var)[[y_var]].count()

    save_var_latex(
        f"perc_tsv_neutral",
        int(
            df[x_var]
            .value_counts(normalize=True)
            .to_frame()
            .query("index == 0")
            .values[0][0]
            * 100
        ),
        "\\percent",
    )

    save_var_latex(
        f"perc_tsv_hot",
        int(
            df[x_var]
            .value_counts(normalize=True)
            .to_frame()
            .query("index == 3")
            .values[0][0]
            * 100
        ),
        "\\percent",
    )
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :-1])
    df_plot = df.groupby(x_var)[y_var].value_counts(normalize=True) * 100
    df_plot.unstack(y_var).plot.barh(stacked=True, color=palette_tp, ax=ax1)
    ax1.set(xlabel="Percentage (%)", ylabel=var_names[x_var])
    ax1.legend(
        bbox_to_anchor=(0.5, 1.04),
        loc="lower center",
        borderaxespad=0,
        frameon=False,
        ncol=3,
    )
    ax1.grid(axis="y", ls="--")
    for ix, row in df_count.reset_index().iterrows():
        ax1.text(112, ix, int(row[y_var]), va="center", ha="right")

    ax2 = fig.add_subplot(gs[0, -1])
    df.groupby(x_var)[x_var].count().plot.bar(color=palette_tsv, ax=ax2)
    ax2.set(ylabel="", xlabel=var_names[x_var], title="Number of votes")
    ax2.yaxis.tick_right()
    ax2.grid(axis="x", ls="--")

    plt.savefig(f"./Manuscript/src/figures/bar_plot_tp_by_ts.png", dpi=300)


def plot_stacked_bar_predictions_ts(hb_models=False):
    plt.close("all")

    fig_name = "bar_stacked_model_accuracy"
    models = models_to_test
    if hb_models:
        models = [f"lr_hb_{x}" for x in models_to_test[:-1]]
        fig_name = "bar_stacked_model_accuracy_hb"

    # Stacked boxplot
    f, axs = plt.subplots(
        1, len(models), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, pmv in enumerate(models):
        var = f"{pmv}_round"
        df_plot = (
            df.groupby("thermal_sensation_round")[var]
            .value_counts(normalize=True)
            .unstack(var)
        )
        if len(df_plot.columns) != 7:
            for x in range(-3, 4):
                if x in df_plot.columns:
                    continue
                else:
                    df_plot[x] = np.nan
        df_plot = df_plot[df_plot.columns.sort_values()]
        df_plot.plot.bar(stacked=True, color=palette_tsv, ax=axs[ix], rot=0)
        accuracy = round(
            accuracy_score(df[var].fillna(999), df["thermal_sensation_round"]) * 100
        )
        axs[ix].set(xlabel="")
        axs[ix].set_title(f"{var_names[pmv]} {accuracy}%", y=1.025)
        handles, labels = axs[ix].get_legend_handles_labels()
        axs[ix].get_legend().remove()
        axs[ix].grid(False)
        df_match = df_plot.stack().reset_index()
        df_match = df_match[df_match["thermal_sensation_round"] == df_match[var]]
        for x in axs[ix].get_xticklabels():
            try:
                match = df_match[df_match[var] == float(x._text)][0].values[0]
                axs[ix].text(
                    x._x,
                    0.5,
                    f"{int(match * 100)}%",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )
            except IndexError:
                axs[ix].text(
                    x._x,
                    0.5,
                    f"0 %",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )

    plt.subplots_adjust(left=0.05, right=1, bottom=0.2, top=0.85)
    cax = plt.axes([0, 0.92, 1, 0.05])
    cax.axis("off")

    cax.legend(
        handles,
        labels,
        frameon=False,
        # mode="expand",
        # bbox_to_anchor=(0, 1.1, 1, 0.2),
        loc="upper center",
        ncol=7,
    )
    f.supxlabel(var_names["thermal_sensation"])
    plt.savefig(f"./Manuscript/src/figures/{fig_name}.png", dpi=300)


def plot_stacked_bar_predictions_model():
    plt.close("all")

    # Stacked boxplot
    f, axs = plt.subplots(
        1, len(models_to_test), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, pmv in enumerate(models_to_test):
        var = f"{pmv}_round"
        df_plot = (
            df.groupby(var)["thermal_sensation_round"]
            .value_counts(normalize=True)
            .unstack("thermal_sensation_round")
        )
        df_counts = df.groupby(var)["thermal_sensation_round"].count()
        if len(df_plot.index) != 7:
            for x in range(-3, 4):
                if x in df_plot.index:
                    continue
                else:
                    df_plot = pd.concat(
                        [
                            df_plot,
                            pd.DataFrame(np.nan, index=[x], columns=df_plot.columns),
                        ],
                    )
        if len(df_plot.columns) != 7:
            for x in range(-3, 4):
                if x in df_plot.columns:
                    continue
                else:
                    df_plot[x] = np.nan
        df_plot = df_plot[df_plot.columns.sort_values()]
        df_plot = df_plot.sort_index()
        df_plot.plot.bar(stacked=True, color=palette_tsv, ax=axs[ix], rot=0)
        axs[ix].set(title=var_names[pmv], xlabel="")
        handles, labels = axs[ix].get_legend_handles_labels()
        axs[ix].get_legend().remove()
        df_match = df_plot.stack()
        df_match = df_match[
            df_match.index.get_level_values(0) == df_match.index.get_level_values(1)
        ]
        df_match = df_match.to_frame().reset_index()
        for x in axs[ix].get_xticklabels():
            try:
                match = df_match[df_match["thermal_sensation_round"] == float(x._text)][
                    0
                ].values[0]
                count = df_counts[df_counts.index == float(x._text)].values[0]
                axs[ix].text(
                    x._x,
                    0.5,
                    f"{int(match * 100)}% - #{count}",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )
            except IndexError:
                axs[ix].text(
                    x._x,
                    0.5,
                    f"0 %",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )

    plt.subplots_adjust(left=0.05, right=1, bottom=0.2, top=0.85)
    cax = plt.axes([0, 0.95, 1, 0.05])
    cax.axis("off")

    cax.legend(
        handles,
        labels,
        frameon=False,
        # mode="expand",
        # bbox_to_anchor=(0, 1.1, 1, 0.2),
        loc="upper center",
        ncol=7,
    )
    f.supxlabel("Model prediction")
    plt.savefig(
        f"./Manuscript/src/figures/bar_stacked_model_accuracy_model.png", dpi=300
    )


def plot_stacked_bar_predictions_tp():
    plt.close("all")

    # Stacked boxplot
    f, axs = plt.subplots(
        1, len(models_to_test), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, model in enumerate(models_to_test):
        df_plot = df.copy()
        df_plot[model] = pd.cut(
            df_plot[model],
            [-3.5, -0.5, 0.5, 3.5],
            labels=["warmer", "no change", "cooler"],
        )
        df_plot = (
            df_plot.groupby("thermal_preference")[model]
            .value_counts(normalize=True)
            .unstack()
        )
        # if len(df_plot.columns) != 7:
        #     for x in range(-3, 4):
        #         if x in df_plot.columns:
        #             continue
        #         else:
        #             df_plot[x] = np.nan
        df_plot = df_plot[df_plot.columns.sort_values(ascending=False)]
        df_plot.plot.bar(stacked=True, color=palette_tp, ax=axs[ix])
        axs[ix].set(title=var_names[model], xlabel="")
        handles, labels = axs[ix].get_legend_handles_labels()
        axs[ix].get_legend().remove()
        df_match = df_plot.stack().reset_index()
        df_match = df_match[df_match["thermal_preference"] == df_match["level_1"]]
        for x in axs[ix].get_xticklabels():
            try:
                match = df_match[df_match["level_1"] == x._text][0].values[0]
                axs[ix].text(
                    x._x,
                    0.5,
                    f"{int(match * 100)}%",
                    va="center",
                    ha="center",
                    size=10,
                    rotation=90,
                )
            except IndexError:
                pass

    plt.subplots_adjust(left=0.05, right=1, bottom=0.2, top=0.85)
    cax = plt.axes([0, 0.95, 1, 0.05])
    cax.axis("off")

    cax.legend(
        handles,
        labels,
        frameon=False,
        # mode="expand",
        # bbox_to_anchor=(0, 1.1, 1, 0.2),
        loc="upper center",
        ncol=7,
    )
    f.supxlabel(var_names["thermal_preference"])
    plt.savefig(f"./Manuscript/src/figures/bar_stacked_model_accuracy_tp.png", dpi=300)


def plot_bias_distribution_whole_db(hb_models=False):

    fig_name = "hist_discrepancies"
    models = models_to_test
    if hb_models:
        models = [f"lr_hb_{x}" for x in models_to_test[:-1]]
        fig_name = f"{fig_name}_hb"

    # plot bias distribution
    f, axs = plt.subplots(
        1, len(models), sharex=True, sharey=True, constrained_layout=True
    )
    axs = axs.flatten()

    for ix, model in enumerate(models):
        df_plot = df[f"diff_ts_{model}"]
        interval = 0.5
        bins_plot = np.arange(-3, 3, interval)
        axs[ix].hist(df_plot, bins=bins_plot, color="gray")
        axs[ix].hist(
            df_plot[(df_plot >= -interval) & (df_plot < interval)],
            bins=bins_plot,
            color="r",
        )
        axs[ix].set(title=var_names[model], ylabel="", xlabel="")
        mean, std = df_plot.mean().round(2), df_plot.std().round(2)
        axs[ix].text(0, 9800, f"{mean} ({std})", va="center", ha="center")

    f.supxlabel("Delta between PMV and TSV")
    plt.savefig(f"./Manuscript/src/figures/{fig_name}.png", dpi=300)


def plot_bias_distribution_by(variable="building_id"):
    # plot bias by building
    plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(1, 1, constrained_layout=True)
        sns.violinplot(
            x=variable,
            y=f"diff_ts_{model}",
            data=df,
            ax=axs,
            color=color,
            scale="count",
        )
        counts = df.groupby(variable)["const"].count()
        axs.axhline(-0.5, c="r")
        axs.axhline(+0.5, c="r")

        labels = [x._text for x in axs.get_xticklabels()]
        for i, label in enumerate(labels):
            axs.text(i - 0.25, -2, label[:20], va="center", ha="center", rotation=90)
            axs.text(i - 0.25, 2, counts[label], va="center", ha="center", rotation=90)

        axs.set(
            ylabel=variable,
            ylim=(-2, 2),
            xlabel="",
            xticklabels="",
        )
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_by_{variable}_{model}.png", dpi=300)


def plot_bias_distribution_by_building():
    # plot bias by building
    plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(1, 1, constrained_layout=True)
        sns.violinplot(
            x="building_id",
            y=f"diff_ts_{model}",
            data=df,
            ax=axs,
            color=color,
            scale="count",
        )
        if model == "pmv":
            good_buildings = df.groupby("building_id")[f"diff_ts_{model}"].median()
            good_buildings = good_buildings[good_buildings.between(-0.5, 0.5)].index
        axs.axhline(-0.5, c="r")
        axs.axhline(+0.5, c="r")

        axs.set(
            ylabel="building id",
            ylim=(-2, 2),
            xlabel="",
            xticklabels="",
        )
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_buildings.png", dpi=300)


def plot_bias_distribution_by_contributor():
    # plot bias by contributor
    plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(1, 1, constrained_layout=True)
        sns.violinplot(
            x="contributor",
            y=f"diff_ts_{model}",
            data=df,
            ax=axs,
            color=color,
            scale="count",
        )
        if model == "pmv":
            good_buildings = df.groupby("building_id")[f"diff_ts_{model}"].median()
            good_buildings = good_buildings[good_buildings.between(-0.5, 0.5)].index
        axs.axhline(-0.5, c="r")
        axs.axhline(+0.5, c="r")

        contributors = [x._text.split(" ")[1] for x in axs.get_xticklabels()]
        for i, contributor in enumerate(contributors):
            axs.text(i - 0.25, -2, contributor, va="center", ha="center", rotation=90)

        axs.set(
            ylabel="building id",
            ylim=(-2, 2),
            xlabel="",
            xticklabels="",
        )
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_contributors.png", dpi=300)


def plot_bias_distribution_by_variable():
    variables = [
        "ta",
        "tr",
        "top",
        # "t_mot_isd",
        "vel",
        "rh",  # todo report pa rather than RH
        "clo",
        "met",
        "thermal_sensation",
        "thermal_preference",
        "pmv",
    ]

    # filter_good_buildings = False
    plt.close("all")
    for ix, model in enumerate(models_to_test):
        color = palette_primary[ix]
        f, axs = plt.subplots(5, 2, constrained_layout=True, figsize=(8, 10))
        axs = axs.flatten()
        for i, var in enumerate(variables):
            # plot bias distribution
            ax = axs[i]
            df_plot = df[[var, f"diff_ts_{model}", "building_id"]].copy().dropna()
            # if filter_good_buildings:
            #     df_plot = df_plot[df_plot["building_id"].isin(good_buildings)]
            if ("thermal" not in var) and ("pmv" not in var):
                df_plot[var] = pd.cut(df_plot[var], bins=10)
            if ("thermal_sensation" == var) or ("pmv" == var):
                df_plot[var] = pd.cut(df_plot[var], bins=np.arange(-3.5, 4.5, 1))
            # elif "" == var:
            #     pass
            sns.violinplot(
                x=var,
                y=f"diff_ts_{model}",
                data=df_plot,
                ax=ax,
                color=color,
                scale="count",
            )
            ax.axhline(-0.5, c="r")
            ax.axhline(+0.5, c="r")

            if "preference" not in var:
                x_labels = [
                    round(x, 1)
                    for x in pd.IntervalIndex(
                        sorted(df_plot[var].cat.categories.unique())
                    ).mid
                ]
                ax.set(
                    xticklabels=x_labels,
                )
            ax.set(ylabel=var_names[var].split(" ")[-1], ylim=(-2, 2), xlabel="")
        plt.suptitle(model)
        plt.savefig(f"./Manuscript/src/figures/bias_{model}.png", dpi=300)


def table_f1_scores():
    results_f1 = {}
    for model in models_to_test:
        df_analysis = df[[f"{model}_round", "thermal_sensation_round"]].copy().dropna()
        x = df_analysis[f"{model}_round"]
        y = df_analysis[f"thermal_sensation_round"]
        results_f1[model] = {}
        for type in ["micro", "macro", "weighted"]:
            results_f1[model][type] = f1_score(y, x, average=type)
    df_f1 = pd.DataFrame.from_dict(results_f1)
    print(df_f1.to_markdown())
    df_f1.round(2).to_latex("./Manuscript/src/tables/f1.tex")

    results_f1 = {}
    for model in [f"lr_hb_{x}" for x in models_to_test[:-1]]:
        df_analysis = df[[f"{model}_round", "thermal_sensation_round"]].copy().dropna()
        x = df_analysis[f"{model}_round"]
        y = df_analysis[f"thermal_sensation_round"]
        results_f1[model] = {}
        for type in ["micro", "macro", "weighted"]:
            results_f1[model][type] = f1_score(y, x, average=type)
    df_f1 = pd.DataFrame.from_dict(results_f1)
    print(df_f1.to_markdown())


if __name__ == "__main__":

    plt.close("all")

    sns.set_context("paper")
    mpl.rcParams["figure.figsize"] = [8.0, 3.5]
    sns.set_style(
        "whitegrid",
        {
            "grid.color": ".85",
            "grid.linewidth": "1",
            "grid.linestyle": "--",
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        },
    )

    plt.rc("axes.spines", top=False, right=False, left=False)
    plt.rcParams["font.family"] = "sans-serif"

    applicability_limits = {
        "ta": [10, 30],
        "tr": [10, 40],
        "vel": [0, 1],
        "clo": [0, 1.5],
        "met": [1, 4],
        "thermal_sensation": [-3.5, 3.5],
        "pmv_ce": [-3.49999, 3.5],
        "pmv_set": [-3.49999, 3.5],
        "pmv_gagge": [-3.49999, 3.5],
        "pmv_toby": [-3.49999, 3.5],
        "rh": [0, 100],
        "pa": [0, 2700],
    }

    var_names = {
        "ta": r"$t_{db}$",
        "tr": r"$\overline{t_{r}}$",
        "top": r"$t_{o}$",
        "vel": r"$V$",
        "rh": r"RH",
        "clo": r"$I_{cl}$",
        "met": r"$M$",
        "thermal_sensation": "Thermal Sensation Vote (TSV)",
        "thermal_sensation_round": "Thermal Sensation Vote (TSV)",
        "thermal_preference": "Thermal Preference Vote (TPV)",
        "age": "Age (years)",
        "ht": "Height (m)",
        "wt": "Weight (kg)",
        "t_mot_isd": r"$t_{ormt}$",
        "pmv": r"PMV",
        "pmv_round": r"PMV",
        "lr_hb_pmv": r"PMV$_{hb}$",
        "lr_hb_pmv_ce": r"PMV$_{CE,hb}$",
        "lr_hb_pmv_set": r"PMV$_{SET,hb}$",
        "lr_hb_pmv_gagge": r"PMV$_{Gagge,hb}$",
        "lr_hb_athb": r"ATHB$_{hb}$",
        "pmv_ce_round": r"PMV$_{CE}$",
        "pmv_ce": r"PMV$_{CE}$",
        "pmv_set": r"PMV$_{SET}$",
        "pmv_set_round": r"PMV$_{SET}$",
        "pmv_gagge": r"PMV$_{Gagge}$",
        "pmv_gagge_round": r"PMV$_{Gagge}$",
        "pmv_toby": r"PMV$_{Toby}$",
        "athb_round": r"ATHB",
        "athb": r"ATHB",
    }

    var_units = {
        "ta": r"$^{\circ}$C",
        "tr": r"$^{\circ}$C",
        "vel": r"m/s",
        "rh": r"%",
        "clo": r"clo",
        "met": r"met",
    }

    models_to_test = [
        "pmv",
        "pmv_ce",
    ]  # ["pmv", "pmv_ce", "pmv_set", "pmv_gagge", "athb", "pmv_toby"]

    # filter data outside standard applicability limits
    df = importing_filtering_processing(load_preprocessed=True)

    df_meta = pd.read_csv("./Data/db_metadata.csv")
    df = pd.merge(df, df_meta, on="building_id", how="left")


if __name__ == "__plot__":

    # def pmv_jiayu_fed(tdb, tr, vr, rh, met, clo, wme=0):
    #
    #     pa = rh * 10 * math.exp(16.6536 - 4030.183 / (tdb + 235))
    #
    #     icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    #     m = met * 58.15  # metabolic rate in W/M2
    #     w = wme * 58.15  # external work in W/M2
    #     mw = m - w  # internal heat production in the human body
    #     # calculation of the clothing area factor
    #     if icl <= 0.078:
    #         f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
    #     else:
    #         f_cl = 1.05 + (0.645 * icl)
    #
    #     # heat transfer coefficient by forced convection
    #     hcf = 12.1 * math.sqrt(vr)
    #     hc = hcf  # initialize variable
    #     taa = tdb + 273
    #     tra = tr + 273
    #     t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)
    #
    #     p1 = icl * f_cl
    #     p2 = p1 * 3.96
    #     p3 = p1 * 100
    #     p4 = p1 * taa
    #     p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
    #     xn = t_cla / 100
    #     xf = t_cla / 50
    #     eps = 0.00015
    #
    #     n = 0
    #     while abs(xn - xf) > eps:
    #         xf = (xf + xn) / 2
    #         hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
    #         if hcf > hcn:
    #             hc = hcf
    #         else:
    #             hc = hcn
    #         xn = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
    #         n += 1
    #         if n > 150:
    #             raise StopIteration("Max iterations exceeded")
    #
    #     tcl = 100 * xn - 273
    #
    #     # heat loss diff. through skin
    #     hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    #     # heat loss by sweating
    #     if mw > 58.15:
    #         hl2 = 0.42 * (mw - 58.15)
    #     else:
    #         hl2 = 0
    #     # latent respiration heat loss
    #     hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    #     # dry respiration heat loss
    #     hl4 = 0.0014 * m * (34 - tdb)
    #     # heat loss by radiation
    #     hl5 = 3.96 * f_cl * (xn ** 4 - (tra / 100.0) ** 4)
    #     # heat loss by convection
    #     hl6 = f_cl * hc * (tcl - tdb)
    #
    #     ts = 0.303 * math.exp(-0.036 * m) + 0.028
    #     return mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6
    #
    # plt.close("all")
    # y_array = []
    # l_array = []
    # for t in np.arange(20, 40, 0.1):
    #     y = -8.471 + 0.33 * t
    #     met = 0.98
    #     l = pmv_jiayu_fed(t, t, 0.1, 50, met, 0.6)
    #     y_array.append(y)
    #     l_array.append(l)
    #     # print(t, l)
    # plt.plot(l_array, y_array, c="g")
    # y_array = []
    # l_array = []
    # for t in np.arange(20, 40, 0.1):
    #     y = -3.643 + 0.175 * t
    #     met = 1.56
    #     l = pmv_jiayu_fed(t, t, 0.2, 50, met, 0.6)
    #     pmv = l * (0.303 * math.exp(-0.036 * met * 58.12) + 0.028)
    #     pmv = l * (0.31 * math.exp(-0.04 * met * 58.12) + 0.028)
    #     y_array.append(y)
    #     l_array.append(l)
    #     # print(t, l)
    # plt.plot(l_array, y_array, c="gray")
    # # plt.plot([-30, 30], [-2, +2])
    #
    # [x for x in df.columns if "_hb" in x]
    # plt.scatter(y=df["thermal_sensation"], x=df["pmv_hb"], c="gray")
    # plt.figure()
    # sns.regplot(
    #     y=df["pmv_hb"],
    #     x=df["thermal_sensation"],
    #     data=df,
    #     scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
    #     lowess=True,
    # )
    # plt.tight_layout()
    #
    # # logistic regression
    # from sklearn.linear_model import LogisticRegression
    #
    # df_reg = df[["pmv_hb", "met", "clo", "thermal_sensation_round"]].dropna()
    # clf = LogisticRegression(random_state=0).fit(
    #     df_reg[["pmv_hb", "met", "clo"]], df_reg["thermal_sensation_round"]
    # )
    # clf.predict([[0, 1, 0.6]])

    # Figure 1 and 2
    plot_distribution_variable()

    # Figure 3
    plot_bar_tp_by_ts()

    # plot model results vs TSV todo add regression lines info
    plot_bubble_models_vs_tsv()

    # plot model accuracy using bar chart
    plot_stacked_bar_predictions_ts()
    # plot_stacked_bar_predictions_ts(hb_models=True)
    # plot_stacked_bar_predictions_tp()

    # plot bias distribution
    plot_bias_distribution_whole_db()
    # plot_bias_distribution_whole_db(hb_models=True)

    # # plot bias by building
    # plot_bias_distribution_by_building()
    # plot_bias_distribution_by(variable="building_id")
    # plot_bias_distribution_by(variable="contributor")
    # plot_bias_distribution_by(variable="region")
    # plot_bias_distribution_by(variable="climate")
    # plot_bias_distribution_by(variable="building_type")
    # plot_bias_distribution_by(variable="cooling_type")
    # plot_bias_distribution_by(variable="country")
    #
    # # plot bias by contributor
    # plot_bias_distribution_by_contributor()

    # plot bias by each variable
    plot_bias_distribution_by_variable()

    # print Markdown table of f1-scores
    table_f1_scores()

    plt.close("all")
    f, axs = plt.subplots(1, 5, constrained_layout=True, sharey=True, sharex=True)
    for ix, model in enumerate(models_to_test):
        sns.regplot(
            x=model,
            y="set",
            data=df,
            scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
            # lowess=True,
            ax=axs[ix],
        )
    plt.savefig(f"./Manuscript/src/figures/scatter_set_vs_models.png", dpi=300)

    plt.close("all")
    f, axs = plt.subplots(1, 5, constrained_layout=True, sharey=True, sharex=True)
    for ix, model in enumerate(models_to_test):
        sns.regplot(
            x=f"{model}_hb",
            y="thermal_sensation",
            data=df,
            scatter_kws={"s": 5, "alpha": 0.5, "color": "lightgray"},
            ax=axs[ix],
            # lowess=True,
            # y_partial="met",
            ci=None,
        )
    plt.savefig(f"./Manuscript/src/figures/scatter_tsv_vs_hb.png", dpi=300)


if __name__ == "__old_code__":
    # accuracies calculation
    for limit in [3, 2, 1]:
        data = df[df["thermal_sensation_round"].abs() <= limit]
        data_iso = data[df["pmv_round"].abs() <= limit]
        data_ash = data[df["pmv_ce_round"].abs() <= limit]

        acc_iso = (
            data_iso[
                data_iso["thermal_sensation_round"] == data_iso["pmv_round"]
            ].shape[0]
            / data_iso.shape[0]
        )
        save_var_latex(f"Overall PMV ISO accuracy - limit {limit}", int(acc_iso * 100))
        acc_ash = (
            data_ash[
                data_ash["thermal_sensation_round"] == data_ash["pmv_ce_round"]
            ].shape[0]
            / data_ash.shape[0]
        )
        save_var_latex(
            f"Overall PMV ASHRAE accuracy - limit {limit}", int(acc_ash * 100)
        )

    # logistic regression models
    plt.figure()
    sns.boxenplot(df.pmv_gagge_hb)
    print(df.groupby("thermal_preference")["ta"].count())

    # gagge heat balance vs thermal sensation
    clf = LogisticRegression(random_state=0).fit(
        df["pmv_gagge_hb"].values.reshape(-1, 1), df["thermal_sensation_round"]
    )
    set_range = np.arange(-60, 60, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df["thermal_sensation_round"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    # gagge heat balance vs thermal preference
    df_dropna = df[["pmv_set_hb", "thermal_preference"]].dropna().sample(frac=1)
    df_log = pd.DataFrame()
    for preference in df_dropna.thermal_preference.unique():
        _df = df_dropna.query("thermal_preference == @preference").head(10000)
        df_log = pd.concat([df_log, _df])
    print(df_log.groupby("thermal_preference")["pmv_set_hb"].count())
    clf = LogisticRegression(
        random_state=0,
    ).fit(df_log["pmv_set_hb"].values.reshape(-1, 1), df_log["thermal_preference"])
    set_range = np.arange(-60, 60, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df["tp_fede"] = clf.predict(df["pmv_set_hb"].values.reshape(-1, 1))
    print(
        df.groupby(["thermal_preference", "tp_fede"])["ta"]
        .count()
        .unstack("thermal_preference")
    )
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    # gagge heat balance vs thermal preference
    df_dropna = (
        df[["pmv_set_hb", "met", "clo", "t_mot_isd", "thermal_preference"]]
        .dropna()
        .sample(frac=1)
    )
    df_log = pd.DataFrame()
    for preference in df_dropna.thermal_preference.unique():
        _df = df_dropna.query("thermal_preference == @preference").head(3000)
        df_log = pd.concat([df_log, _df])
    print(df_log.groupby("thermal_preference")["pmv_set_hb"].count())
    clf = LogisticRegression(random_state=0,).fit(
        df_log[["pmv_set_hb", "met", "clo", "t_mot_isd"]].values,
        df_log["thermal_preference"],
    )
    set_range = np.arange(-60, 60, 0.5)
    prob = clf.predict_proba([[x[0], 1.2, 0.6, 15] for x in set_range.reshape(-1, 1)])
    df_dropna["tp_fede"] = clf.predict(
        df_dropna[["pmv_set_hb", "met", "clo", "t_mot_isd"]].values
    )
    print(
        df_dropna.groupby(["thermal_preference", "tp_fede"])["tp_fede"]
        .count()
        .unstack("thermal_preference")
    )
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob["no change"], label="no change")
    plt.plot(set_range, df_prob[["cooler", "warmer"]].sum(axis=1), label="change")
    plt.tight_layout()
    plt.legend()

    clf = LogisticRegression(random_state=0).fit(
        df["set"].values.reshape(-1, 1), df["thermal_sensation_round"]
    )
    set_range = np.arange(5, 40, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df["thermal_sensation_round"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob[[0]].sum(axis=1), label="neutral")
    plt.plot(set_range, df_prob[[-3, -2, -1, 1, 3, 2]].sum(axis=1), label="hot or cold")
    plt.tight_layout()
    plt.legend()

    df_log = df[["set", "thermal_preference"]].dropna()
    clf = LogisticRegression(random_state=0).fit(
        df_log["set"].values.reshape(-1, 1), df_log["thermal_preference"]
    )
    set_range = np.arange(5, 40, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob["no change"], label="no change")
    plt.plot(set_range, df_prob[["cooler", "warmer"]].sum(axis=1), label="change")
    plt.tight_layout()
    plt.legend()

    df_log = df[["ta", "thermal_preference"]].dropna()
    clf = LogisticRegression(random_state=0).fit(
        df_log["ta"].values.reshape(-1, 1), df_log["thermal_preference"]
    )
    set_range = np.arange(5, 40, 0.5)
    prob = clf.predict_proba(set_range.reshape(-1, 1))
    df_prob = pd.DataFrame(prob, columns=sorted(df_log["thermal_preference"].unique()))
    plt.figure()
    for col in df_prob.columns:
        plt.plot(set_range, df_prob[col], label=col)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.plot(set_range, df_prob["no change"], label="no change")
    plt.plot(set_range, df_prob[["cooler", "warmer"]].sum(axis=1), label="change")
    plt.tight_layout()
    plt.legend()

    # # Old Figure 2
    # bar_chart(data=df, ind="tsv", show_per=False, figletter="a")
    # bar_chart(
    #     data=df,
    #     ind="tsv",
    #     show_per=False,
    #     figletter="a",
    #     variables=["pmv_gagge_round", "pmv_set_round"],
    # )
    # bar_chart(
    #     data=df,
    #     ind="tsv",
    #     show_per=False,
    #     figletter="a",
    #     variables=["pmv_round", "athb_round"],
    # )
    # legend_pmv()

    # Figure 3
    plot_error_prediction(data=df[df.vel > 0.1])
    plot_error_prediction(data=df)

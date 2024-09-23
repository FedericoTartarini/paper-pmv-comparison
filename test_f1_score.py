import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import numpy as np
from pythermalcomfort.models import two_nodes, pmv
from pprint import pprint

pprint(two_nodes(tdb=35, tr=35, v=0.1, rh=50, met=1.1, clo=0.5))
pprint(pmv(tdb=35, tr=35, vr=0.1, rh=50, met=1.1, clo=0.5, standard="ASHRAE"))
pprint(
    pmv(
        tdb=35,
        tr=35,
        vr=0.1,
        rh=50,
        met=1.1,
        clo=0.5,
        standard="ISO",
        limit_inputs=False,
    )
)


f, ax = plt.subplots(2, 2, constrained_layout=True)
ax = ax.flatten()
for clo in [0.5, 1]:
    for met in [1.1, 2]:
        results = []
        for t in np.arange(20, 50, 0.1):
            r = two_nodes(tdb=t, tr=t, v=0.1, rh=50, met=met, clo=clo)
            r["tdb"] = t
            r["clo"] = clo
            results.append(r)

        df = pd.DataFrame(results)
        ax[0].plot(df["_set"], df["w"], label=f"{clo=}-{met=}")
        ax[2].plot(df["tdb"], df["w"], label=f"{clo=}-{met=}")
        ax[1].plot(df["_set"], df["pmv_gagge"], label=f"{clo=}-{met=}")
        ax[3].plot(df["tdb"], df["pmv_gagge"], label=f"{clo=}-{met=}")
        ax[0].legend()
        ax[1].legend()
        ax[3].legend()
        t_pmv_3 = df[df["pmv_gagge"] < 3]["_set"].max()
        ax[1].axvline(x=t_pmv_3)
        ax[0].axvline(x=t_pmv_3)
        t_pmv_3 = df[df["pmv_gagge"] < 3]["tdb"].max()
        ax[3].axvline(x=t_pmv_3)
        ax[1].axhline(y=3)
        ax[3].axhline(y=3)
ax[0].set(ylabel="w", xlim=(20, 40), xlabel="set")
ax[2].set(ylabel="w", xlim=(20, 40), xlabel="tdb")
ax[1].set(ylabel="pmv", xlim=(20, 40), xlabel="set")
ax[3].set(ylabel="pmv", xlim=(20, 40), xlabel="tdb")
plt.show()
plt.savefig("./Manuscript/src/figures/PMV and w as a function of SET.pdf")

# SMOTE example https://towardsdatascience.com/smote-fdce2f605729
# F1 score example https://towardsdatascience.com/the-f1-score-bec2bbc38aa6

data = pd.read_csv(
    "https://raw.githubusercontent.com/JoosKorstanje/datasets/main/sales_data.csv"
)
data.head()

plt.figure()
data.pivot_table(index="buy", aggfunc="size").plot(
    kind="bar", title="Class distribution"
)
plt.show()

train, test = train_test_split(data, test_size=0.3, stratify=data.buy)

plt.figure()
train.pivot_table(index="buy", aggfunc="size").plot(
    kind="bar", title="Verify that class distribution in train is same as input data"
)
plt.show()


# Instantiate the Logistic Regression with only default settings
my_log_reg = LogisticRegression()

# Fit the logistic regression on the independent variables of the train data with buy as dependent variable
my_log_reg.fit(
    train[["time_on_page", "pages_viewed", "interest_ski", "interest_climb"]],
    train["buy"],
)

# Make a prediction using our model on the test set
preds = my_log_reg.predict(
    test[["time_on_page", "pages_viewed", "interest_ski", "interest_climb"]]
)

tn, fp, fn, tp = confusion_matrix(test["buy"], preds).ravel()
print(
    "True negatives: ",
    tn,
    "\nFalse positives: ",
    fp,
    "\nFalse negatives: ",
    fn,
    "\nTrue Positives: ",
    tp,
)

print("Accuracy is: ", accuracy_score(test.buy, preds))
print("Precision is: ", precision_score(test.buy, preds))
print("Recall is: ", recall_score(test.buy, preds))
print("F1 is: ", f1_score(test.buy, preds))

print(classification_report(test["buy"], preds))

from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE().fit_resample(
    train[["time_on_page", "pages_viewed", "interest_ski", "interest_climb"]],
    train["buy"],
)

pd.Series(y_resampled).value_counts().plot(
    kind="bar", title="Class distribution after appying SMOTE", xlabel="buy"
)

# Instantiate the new Logistic Regression
log_reg_2 = LogisticRegression()

# Fit the model with the data that has been resampled with SMOTE
log_reg_2.fit(X_resampled, y_resampled)

# Predict on the test set (not resampled to obtain honest evaluation)
preds2 = log_reg_2.predict(
    test[["time_on_page", "pages_viewed", "interest_ski", "interest_climb"]]
)

print(classification_report(test["buy"], preds2))

# test with my data
plt.close("all")
df = pd.read_pickle(r"./Data/db_analysis.pkl.gz", compression="gzip")

independent_variable = "ta"
dependent_variable = "thermal_preference"

df = df.dropna(subset=dependent_variable)

plt.figure()
df.pivot_table(index=dependent_variable, aggfunc="size").plot(
    kind="bar", title="Class distribution"
)
plt.show()

train, test = train_test_split(df, test_size=0.3, stratify=df[dependent_variable])

plt.figure()
train.pivot_table(index=dependent_variable, aggfunc="size").plot(
    kind="bar", title="Verify that class distribution in train is same as input data"
)
plt.show()

my_log_reg = LogisticRegression(max_iter=1000)

# Fit the logistic regression on the independent variables of the train data with buy as dependent variable
my_log_reg.fit(
    train[[independent_variable]],
    train[dependent_variable],
)

preds = my_log_reg.predict(test[[independent_variable]])

print(classification_report(test[dependent_variable], preds))

preds = my_log_reg.predict(np.arange(0, 50, 0.5).reshape(-1, 1))
plt.figure()
plt.plot(np.arange(0, 50, 0.5), preds)
plt.show()

X_resampled, y_resampled = SMOTE().fit_resample(
    train[[independent_variable]],
    train[dependent_variable],
)

pd.Series(y_resampled).value_counts().plot(
    kind="bar",
    title="Class distribution after applying SMOTE",
    xlabel=dependent_variable,
)

# Instantiate the new Logistic Regression
log_reg_2 = LogisticRegression(max_iter=1000)

# Fit the model with the data that has been resampled with SMOTE
log_reg_2.fit(X_resampled, y_resampled)

# Predict on the test set (not resampled to obtain honest evaluation)
preds2 = log_reg_2.predict(test[[independent_variable]])

print(classification_report(test[dependent_variable], preds2))

preds = log_reg_2.predict(np.arange(0, 50, 0.5).reshape(-1, 1))
plt.figure()
plt.plot(np.arange(0, 50, 0.5), preds)
plt.show()


from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic

tdb = 25
tr = 25
rh = 50
v = 0.1
met = 1.4
clo = 0.5
# calculate relative air speed
v_r = v_relative(v=v, met=met)
# calculate dynamic clothing
clo_d = clo_dynamic(clo=clo, met=met, standard="ashrae")
results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d, standard="ashrae")
print(results)
print(results["pmv"])

tdb = 19.6
tr = 19.6
rh = 86
v = 0.1
met = 1.1
clo = 1
# calculate relative air speed
v_r = v_relative(v=v, met=met)
# calculate dynamic clothing
clo_d = clo_dynamic(clo=clo, met=met, standard="ashrae")
results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d, standard="ashrae")
print(results)
print(results["pmv"])

tdb = 19.6
tr = 19.6
rh = 86
v = 0.2
met = 1.1
clo = 1
# calculate relative air speed
v_r = v_relative(v=v, met=met)
# calculate dynamic clothing
clo_d = clo_dynamic(clo=clo, met=met, standard="ashrae")
results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d, standard="ashrae")
print(results)
print(results["pmv"])

tdb = 29.4
tr = 29.4
rh = 61
v = 0.1
met = 1.9
clo = 0.5
# calculate relative air speed
v_r = v_relative(v=v, met=met)
# calculate dynamic clothing
clo_d = clo_dynamic(clo=clo, met=met, standard="ashrae")
results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d, standard="ashrae")
print(results)
print(results["pmv"])

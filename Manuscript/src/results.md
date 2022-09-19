# Summary results

## Comfort DB overview

The dataset contains mainly:
* temperature data between 20 and 27C.
* RH between 30 and 70 %.
* V lower than 0.2 m/s
* Met lower than 1.5 met

![](./figures/dist_input_data.png)

* Age is not normally distributed with more votes from younger adults
* Running mean temperature is also skewed with mainly warm data

![](./figures/dist_other_data.png)

The dataset is not well-balanced, most of the people reported to be thermally `neutral`.

The great majority of the participants voted `no change` when thermally `neutral`. 
Thermal preference can be easily used to predict `hot` or `cold`.
The great majority of the people who reported to be `slighlty warm` or `slightly cold` wanted to be `cooler` and `warmer`, respectively.

![](./figures/bar_plot_tp_by_ts.png)

## Preliminary PMV model comparison

### Prediction accuracy
* All models can only accurately predict the thermal sensation of thermally `neutral` people.
All models failed to predict thermal discomfort.
* ATHB model performed best followed by the PMV Gagge.
> while both the above-mentioned models had a higher accuracy in detecting people who reported to be thermally `neutral` this results could be deceptive since a model who always predict `neutral` will lead to a higher accuracy. 

* The ATHB model which tent to predict `neutral` under every condition.
* The ATHB chart comprises a smaller number of data-points since not all the studies measured running mean outdoor temperature.

![](./figures/bar_stacked_model_accuracy.png)

This plot shows on the x-axis the predicted PVM value while the stacked bar plot shows the number of thermal sensation votes recorded for the PMV binned. We are also reporting the number of points per bin.
Even in this scenario only for PMV = 0 results are `satisfactory`.

![](./figures/bar_stacked_model_accuracy_model.png)

#### Thermal sensation f1-micro

**F1 score definition** The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while recall (also known as sensitivity) is the fraction of relevant instances that were retrieved.
- `micro` calculate metrics globally by counting the total true positives, false negatives and false positives.
- `macro` calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
- `weighted` calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

|          |      pmv |   pmv_ce |   pmv_set |   pmv_gagge |     athb |   pmv_toby |
|:---------|---------:|---------:|----------:|------------:|---------:|-----------:|
| micro    | 0.33595  | 0.34124  |  0.382199 |    0.391962 | 0.418915 |   0.375858 |
| macro    | 0.152728 | 0.155519 |  0.130147 |    0.130317 | 0.13544  |   0.17403  |
| weighted | 0.297244 | 0.29656  |  0.296015 |    0.299774 | 0.312856 |   0.312435 |

#### Predicted vs reported thermal preference
Thermal preference was calculated considering -0.5 < PMV < 0.5 satisfactory.
On the x-axis I am showing the reported thermal preference while on the y-axis the percentage of time the PMV correctly predicted thermal preference.

![](./figures/bar_stacked_model_accuracy_tp.png)

### Regression

The LOWESS line of some models do not pass through the center point.
The slope of the line is low.

![](./figures/bubble_models_vs_tsv.png)

### Overall bias

In red the values that are between -0.5 and 0.5. 
The text reports the mean and standard deviation of the distribution. 
Same analysis as Humphreys et al. (2002)

> PMV is free from serious bias, same conclusion as Humphreys et al. (2002)

![](./figures/hist_discrepancies.png)

### Bias per model vs each variable

The following Figures show the value of PMV - TSV binned by different variables.
The value on the x-axis is the middle point of the bin.

#### Temperature
* PMV Gagge, SET, or ATHB much better than PMV and PMV-CE for common air temperatures.
* PMV model lower applicability limit may need to be changed to 16 C.
* For t-r higher than 33 the model does not work well. Recommend to use same limits for t-db and t-r
* For low t-r results are mixed.
* PMV and PMV-CE t-o bias are very bad. PMV Gagge, SET, and ATHB are better for the `center` range. But still bad at the extremes
* Various results with running mean outdoor temperature

#### Airspeed
* PMV-CE perform worse than PMV at higher airspeeds
* For airspeed bias PMV SET or ATHB are better

#### RH
* RH quite bad results for PMV and PMV-CE. Other models perform better in the `central` range but worse in edge conditions.

#### Clo
* PMV and PMV had bad results for low and high clo values.

#### Met
* PMV very bad for met higher than 2.5 met

#### TS and TP
* All model are very bad aside for TSV = 0

#### PMV models comparison
* ATHB has less bias towards different PMV values, while other models are okay only for central categories or PMV between 0 and 1.

![](./figures/bias_pmv.png)

![](./figures/bias_pmv_ce.png)

![](./figures/bias_pmv_gagge.png)

![](./figures/bias_pmv_set.png)

![](./figures/bias_athb.png)

## Formulaic error

The following scatter plot illustrates the formulaic error between the SET and the various PMV models.
Since both the SET and PMV models use the same inputs, the delta is only caused by the difference in the formulation of both models.
Analysis from Humphreys et al. 2000.

![](./figures/scatter_set_vs_models.png

# Main issue with the PMV

From the above mentioned analysis it looks like the PMV is accurate enough to estimate thermal neutrality.
The issue arises when we move away from thermal neutrality.
The PMV tries to find a correlation between heat losses/gains from a cylinder to the environment and thermal sensation.
This is problematic since in most conditions of mild thermal discomfort the body it is still in equilibrium withe environment.
Albeit it is taking some actions to compensate for the instability (sweating, vasoconstriction, vasodilation).
Moreover the correlation between PMV and heat losses is based solely on a few datapoints collected by Nevis et al.
The PMV is based on a study which determined the correlation between TSV and T for four activity levels.
Clothing were kept constant.
A correlation between TSV and T was found and by substituting this new equation in the PMV model a correlation between TSV and L can also be found.
This correlation is assumed to be constant which is already the first main possible issue.
These data were then used to determine a correlation between PMV and L (heat losses) as a function of metabolic rate.

# TODO
* check with marcel the output of his model
* shall we analyse the data by building type?
* undersampling 500 data point or more, repeat test like dorn paper
* plot heat losses (L) vs thermal sensation and preference
* which model should we recommend

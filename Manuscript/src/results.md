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
The great majority of the participants voted `no change` when reported to be `neautral`. 
Thermal predeference can be easily used to predict `hot` or `cold`.
The great majority of the people who reported to be `slighlty warm` or `slightly cold` wanted to be `cooler` and `warmer`, respectively.

![](./figures/bar_plot_tp_by_ts.png)

## Preliminary PMV model comparison

### Prediction accuracy
All models can only be used to predict the thermal sensation of thermally `neutral` people.
All models failed to predict thermal discomfort.
ATHB model performed best followed by the PMV gagge.
It should be noted that while both the above-mentioned models had a higher accuracy in detecting people who reported to be thermally `neutral` this results could be deceptive since a model who always predict `neutral` will lead to a higher accuracy. 
This somewhat appears to be the case with the ATHB model which tent to predict `neutral` under every conditions.

![](./figures/bar_stacked_model_accuracy.png)

This plot shows on the x-axis the predicted PVM value while the stacked bar plot shows the number of thermal sensation votes recorded for the PMV binned. We are also reporting the number of points per bin.
Even in this scenario only for PMV = 0 results are `satisfactory`.

![](./figures/bar_stacked_model_accuracy_model.png)

#### Thermal sensation f1-micro
|          |      pmv |   pmv_ce |   pmv_set |   pmv_gagge |     athb |
|:---------|---------:|---------:|----------:|------------:|---------:|
| micro    | 0.33595  | 0.34124  |  0.382199 |    0.391962 | 0.418695 |
| macro    | 0.152728 | 0.155519 |  0.130147 |    0.130317 | 0.135975 |
| weighted | 0.297244 | 0.29656  |  0.296015 |    0.299774 | 0.313977 |

#### Predicted vs reported thermal preference
Thermal preference was calculated considering -0.5 < PMV < 0.5 satisfactory.

![](./figures/bar_stacked_model_accuracy_tp.png)

### Regression
![](./figures/bubble_models_vs_tsv.png)

### Overall bias
![](./figures/hist_discrepancies.png)

### Bias per model vs each variable

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

#### PMV
* ATHB has less bias towards different PMV values, while other models are okay only for centeral categores or PMV between 0 and 1.

![](./figures/bias_pmv.png)

![](./figures/bias_pmv_ce.png)

![](./figures/bias_pmv_gagge.png)

![](./figures/bias_pmv_set.png)

![](./figures/bias_athb.png)


"""Interpretations for the notebook, based on its executed outputs."""

INTERPRETATIONS = {
    "naive": r"""
There are two problems here: strict date parsing stops with an error, while the permissive conversions can quietly lose or misread information.

The strict date parser just refuses the file. That's actually fine, because at least it tells me something is wrong. The `mixed` mode is the dangerous one. It reads `12/11/2023` as **11 December** instead of 12 November, which creates a duplicate date and makes the calendar jump forward 30 days and then back 28. Later, when I build features like "yesterday's anxiety", the rows around that date would quietly be wrong, and nothing would warn me.

The number conversion throws away **15 values I actually wrote down**, 3 in every column. These are the ones written as words, as fractions, or with units. After the conversion they look exactly like the days I left blank, which breaks my codebook's rule that blank means "not recorded".

To be fair, 15 cells out of 6,844 is only about 0.2%, so the overall count is small. The bigger issue is the ordering: one misread date affects every feature that looks back in time.

So instead of trusting the defaults, I'm going to parse every cell myself, keep track of which rule handled it, and check the calendar after parsing.
""",

    "audit": r"""
All **6,900 measurement cells and 1,380 date cells** matched a rule, including the explicit missing-value rules, so nothing was left unparseable. Ten dates needed something other than the standard format, and **23 recorded values needed normalization or recovery**: 8 had extra spaces, 6 were number words, 5 were fractions, 3 had units, and 1 was the label "Fair".

I printed every fixed cell above so I could check each one by eye against the raw text. For example, `3/10` became 3 for anxiety and `0 session(s)` became 0 for exercise. The calendar check also passed, with 1,380 days in a row and no duplicates or gaps. That's an additional consistency check that `12/11/2023` means 12 November, because reading it the other way would have created a duplicate date and the check would have failed.

One limitation is that these rules only cover patterns that actually appear in this file. If a new one showed up, like "5 out of 10", it would be marked `unparseable` and the `assert` would stop the notebook. I did that on purpose. I'd rather the notebook crash than quietly accept something it doesn't understand.

Parsing tells me what a cell *says*. It doesn't tell me whether that value is even possible, so that's the next check.
""",

    "ranges": r"""
Four values parsed fine but are outside their scale, so they're now marked as missing:

- **Sleep adequacy of 6** on 2024-06-02, when the scale only goes from 1 to 5.
- **Sleep quality of 0** on 2023-01-15, when the scale starts at 1.
- **Exercise of −1** on 2025-11-28. You can't do a negative number of workouts.
- **Anxiety of 11** on 2023-06-11, when the scale goes from 0 to 10. It is outside the codebook. The other measurements cannot tell me what the intended anxiety rating was.

A range check can only catch values that are impossible. If I wrote a 3 when I meant a 7, nothing in the data can catch that.

I'm keeping all four as missing values and not dropping any rows. Missing predictors may be filled later inside each training fold. Missing target ratings will be excluded after alignment, not filled.
""",

    "summary": r"""
I recorded **6,844** values. The default conversion would have kept 6,829 of them, my parser keeps all 6,844, and the range check removes the 4 impossible ones, which leaves **6,840**.

Looking at whole days, **1,348 out of 1,380 (97.7%) have all five values**. Seven days are completely empty, in three short gaps of two or three days each. Anxiety, the thing I'm predicting, is missing on 8 days: those 7 plus the day with the 11.

The empty days bother me a little because they come in short streaks, which are exactly the days where "yesterday's value" has nothing to point to. I can't tell whether those were unusually good days or unusually bad ones, so I'm not going to pretend I know.

At the modeling stage, forecasts with a missing target rating will be excluded after alignment. Missing current anxiety or habits can remain as predictors and be handled using training-only information; they do not automatically make tomorrow's target unusable. The cleaned data is saved to `data/processed/anxiety_clean.csv`.
""",

    "timeseries": r"""
The first thing I notice is that my anxiety comes in waves, not random spikes. The rolling mean makes the rises and quieter stretches easier to see than the daily line alone. A sustained rise in the mean is different from an uninterrupted run of daily ratings above 6; I count those runs separately below. Honestly, I could match many of these changes to things happening in my life at the time.

**252 of 1,372 rated days (18.4%) were 6 or higher**, and they tend to come together. I count **60 "episodes"** of at least two high days in a row. Most of them last two days, and the longest lasted **10 days**.

The yearly table is going to matter later. 2024 and 2025 were calmer (averages of 3.34 and 3.37), but **2026 has been the hardest year so far**, with an average of 3.99 and a quarter of days at 6 or above. In the earlier years that was 14–19%.

(6 is a cutoff I picked myself. A different cutoff would change the number of episodes, but it wouldn't change the fact that high days cluster together.)

This changes two things in my plan:

1. Because I want a future forecast, the main split has to go in time order. Random splitting allows training on dates later than a validation forecast. I will measure any score difference, rather than assume shuffling must improve it.
2. My test period will be 2026, which has been harder than the years the model learns from. If the test error ends up worse than the validation error, that might reflect a real change in my life rather than a broken model.


I have already explored 2026 here, so I cannot call it an untouched holdout. I will freeze model choices using the earlier development periods before final evaluation. Missing ratings are excluded from the yearly high-day proportions, and they break high-anxiety episode runs.
""",

    "acf": r"""
Yesterday's anxiety has a correlation of **0.58** with today's. It drops to **0.42 after two days**, **0.33 after three**, and **0.08 after seven**.

What interests me is that the association is still visible beyond yesterday. That makes recent averages worth testing alongside today's rating. It does not prove a particular memory mechanism or tell me which window predicts best. Slow changes between years can also affect correlations calculated over the whole series.

After about a week the correlations are much smaller, roughly 0.02–0.06 at the displayed later lags. I have left off an independent-day significance band because the series is temporally dependent.

My first reference will be **persistence**, predicting that tomorrow equals today. In Section 11 I'll compare lookbacks of 1, 3, 7 and 14 days using development results, rather than deciding from this plot alone.
""",

    "distributions": r"""
- **Sleep adequacy and sleep quality** use the whole 1–5 scale and are centered around 3 (averages of 3.25 and 3.35), so a model has variation to work with; whether it improves forecasts still needs testing.
- **Exercise** is basically yes or no. Almost every day is 0 or 1 session, and I only did 2 sessions on 37 of 1,366 days (average 0.50).
- **Social encounters** go from 0 to 5 but are usually 1 or 2 (average 1.77). Days with 4 or 5 are rare.
- **Anxiety** is skewed to the right. Most days are between 2 and 5 (average 3.56, median 3), with a thin tail up to 10.

Two things follow from this. First, both zero and one exercise session are well represented, but the two-session category is small. That limits conclusions about higher frequency; it does not make exercise generally uninformative. Second, my worst days (7 to 10) are rare, so the model won't have many examples of them. A model that minimizes squared error tends to pull its predictions toward the middle, so I will check whether it underestimates my worst days. Because of that, I'll report the error on high-anxiety days separately, instead of only giving one overall number.
""",

    "levels": r"""
The sleep panels show the largest differences. At sleep adequacy 1 versus 5, same-evening means are **6.5 versus 1.4**, while next-evening means are **5.4 versus 2.2**. The gap is smaller for the next evening: about **3.2 points rather than 5.1**. Sleep quality has a similar pattern: **7.1 versus 1.5** on the same evening and **5.4 versus 2.6** the next evening.

Exercise and socializing show gentler differences. With no exercise versus one session, next-evening means are **3.7 versus 3.4**; two sessions also averages about 3.4. With zero versus 4+ social encounters, next-evening means are **4.0 versus 3.2**. These are group descriptions, not effects of changing a habit.

The counts matter. Sleep quality 1 has **53 same-evening ratings but 51 next-evening ratings**. Two exercise sessions has only **37 ratings for each evening**. A group mean based on fewer observations can be sensitive to a small number of days. No uncertainty intervals are estimated here.

The orange sleep lines suggest a next-day association worth investigating, but today's anxiety is also related to both sleep and tomorrow's rating. Step 2 checks how the descriptive association looks within groups with the same current anxiety.
""",

    "holdfixed": r"""
On the same retained days, sleep adequacy has pooled Spearman correlation **−0.399**, compared with a weighted within-rating correlation of **−0.070**. Sleep quality changes from **−0.285 to +0.086**. Exercise changes from **−0.069 to +0.017**, and socializing from **−0.099 to +0.013**. The within-group averages are all closer to zero, although sleep quality also changes sign.

For comparison with the earlier analysis, the table retains the correlations over all eligible pairs: **−0.418, −0.307, −0.076 and −0.117**, respectively. The plotted pooled values differ because both panels now exclude the same small groups. Each habit retains about **98.4% of its eligible pairs**. Ratings 9 and 10 today are excluded: they have only **16 and 6 complete pairs**, respectively. The within-group comparison therefore describes ratings 0–8 today, not the most extreme current ratings.

This makes anxiety history a useful reference for the modeling stage, but does not establish that the habits add no forecast value. The check holds only today's rating fixed, not earlier history, and an average can conceal different patterns across groups. A small rank correlation can also miss a nonlinear relationship. The positive sleep-quality average is not evidence that good sleep causes higher anxiety.

The next question is empirical: on the same future target dates, does adding lifestyle information to an anxiety-history model reduce forecast error? Only the chronological model comparison can answer that. These plots do not explain a percentage of a causal effect or supply a significance test.
""",

    "weekday": r"""
Weekends look a bit calmer. Saturday averages 3.29 and Sunday 3.45, while weekdays are between 3.58 and 3.70. The gap between the highest and lowest day is 0.41 anxiety points. I treat this as a descriptive comparison, not an independent-day significance test.

I don't want to read too much into this, though. I'm comparing seven days at once, so one of them looking different by chance isn't that surprising, and 0.41 is small next to my normal day-to-day spread (a standard deviation of 2.17). I use a single `target_is_weekend` flag for the day I am predicting. All later ablations keep this calendar control, so they do not isolate its added value. Its inclusion is a simple design choice suggested by this descriptive exploration, which already includes 2026.
""",

    "wrapup": r"""
1. **The data is in good shape, and I can explain every cleaning step.** The parser kept all 6,844 recorded values, including the 15 the default conversion loses. The cleaner removed four invalid values; 97.7% of days are complete, and the calendar covers 1,380 consecutive days.
2. **My recent anxiety is a useful reference to test.** Yesterday's rating has correlation 0.58 with today's. Persistence and recent averages will show how far anxiety history alone can take the forecast.
3. **Sleep has the strongest same-evening association among these habits.** The lowest and highest sleep levels differ by about five points in same-evening anxiety. Exercise and socializing show smaller differences.
4. **The next-evening associations are smaller within groups with the same current anxiety.** This is a descriptive result, not proof of a causal pathway or evidence that habits cannot improve a model.
5. **The rules for the next step:** keep the calendar intact, use only information available by evening $t$, exclude missing targets after alignment, and evaluate models chronologically. I have already explored 2026 and will disclose that exposure.
""",

    "shortfall_example": r"""
With $\lambda=0.8$, the half-life is **3.11 days**. In the constructed example, one adequacy rating of 1 starts the score at 2 and it then decays. Three successive ratings of 1 give **2, 3.6 and 4.88** on the first three days, because each new contribution is added to what remains. The figure illustrates that arithmetic; it does not show an estimated relationship with anxiety.

The small table uses $\lambda=0.5$ so I can check every step mentally. The first two ratings, 1 and 2, give scores 2 and 2. On the missing third rating, the observed score decays to 1 while missing-input weight rises to 1. After the fourth rating of 4, they decay to 0.5 and 0.5. A decaying observed score therefore does not mean the unrecorded night was good.

I'll keep both sleep columns as candidates and later compare forecasts with and without them. Neither this illustration nor the chosen decay is evidence that the feature helps.
""",

    "feature_checks": r"""
The hand calculations, target alignment and calendar checks passed. An entirely unobserved window stays missing with coverage zero. The missing-sleep bound also matches the scores obtained by filling the unknown input with each endpoint of its codebook; those fills are used only for this check, never for the modeling table.

At all three cutoffs, changing every future measurement left earlier features identical. Rebuilding from just the available prefix also reproduced them, including across the July 2023 missing-data run. Changing sleep adequacy on the issue date changed that day's input and left earlier rows alone.

These checks support the timing and arithmetic of this implementation. They do not establish forecast accuracy. No imputer, scaler or model has been fitted, so training-fold checks will still be needed when those steps are added.
""",

    "feature_rows": r"""
The seven-day configuration gives **1,365 eligible forecasts and 17 candidate features**. Starting with 1,380 issue dates, I exclude six without a full calendar lookback, one whose target is beyond the file, and eight whose target rating is missing. The resulting target dates run from **8 January 2023 to 11 October 2026**.

I retain **28 forecasts with at least one missing predictor**, including **four with missing current anxiety**. There are **41 missing predictor cells** in total. These are genuine gaps to handle inside training folds, not reasons to invent target ratings. Coverage and missingness indicators already describe the gaps without filling the underlying values.

The domain counts are **6 history, 6 sleep, 2 exercise, 2 socializing and 1 calendar feature**. The core nine columns match the equation; the other eight describe missingness and coverage. The feature dictionary records each column's domain so later ablations remove all related inputs together.

The modeling table is now aligned in memory, but it has not been split or used to fit anything. Next I will define ridge and its objective in Section 7, then establish chronological splits and baseline forecasts in Section 8. The window and feature set will be compared using development data. Decay 0.8 and shortfall threshold 3 remain fixed design assumptions; their alternatives are not tested here.
""",

    "ridge_setup_demo": r"""
**How to read the figure.** Each bar is one supplied pair of weights. The blue part is mean squared prediction error, the orange part is the weight penalty, and the full height is their sum, $J$. A and B have no visible blue segment because their MSE is exactly zero in this constructed example. Their orange bars are not forecast errors. A lower total is preferred by this particular training objective, at this particular alpha.

Choices A and B both predict the three artificial targets exactly, so both have zero training MSE. Sharing the contribution between two identical inputs changes the squared-weight sum from $2^2+0^2=4$ to $1^2+1^2=2$. With $\alpha=0.25$, the penalty drops from **1.000 to 0.500** without changing a prediction.

Choice C makes both weights smaller again. Its penalty falls to **0.125**, but its predictions become $2,3,4$ instead of $1,3,5$. The residuals are $-1,0,1$, so its MSE is $[(-1)^2+0^2+1^2]/3=2/3\approx\mathbf{0.667}$. Adding the penalty gives $J\approx\mathbf{0.792}$. C pays less for its weights, but its worse fit costs more than it saves relative to B.

B has the lowest objective **among these three supplied choices**. I have not searched over all possible weights, and B is not claimed to be the overall ridge solution. The intercept-shift check provides another useful check on the equation: adding 2 to every target and to the intercept leaves the residuals and the weight penalty unchanged. If that changed the objective, I would have accidentally penalized the intercept or computed the residuals incorrectly.

This is why shrinking every weight as far as possible is not the goal: the objective balances fit against the penalty. The journal's overlapping features are not identical like these toy columns, so their learned weights need not be shared equally. I will judge the actual tradeoff using future-date errors, not this illustration.
""",

    "split_results": r"""
The aligned data gives **899 training, 184 validation and 282 final-evaluation forecasts**, totaling 1,365. The first training target is 8 January 2023; the last is 30 June 2025. Validation targets cover July–December 2025, and final targets cover 1 January–11 October 2026.

These are the counts I will use from now on, rather than the earlier plan's raw-calendar counts. Seven target dates at the beginning cannot be forecast with the required seven-day history inside this file, and missing target ratings exclude additional examples. Missing predictors remain eligible and will be handled by training-fitted preprocessing. The boundary checks confirm that a fitting label is available by the first held-out issue evening.
""",

    "validation_baselines": r"""
**Persistence is the strongest of these three validation baselines.** Its MAE is **1.375 anxiety points** over 184 forecasts, compared with **1.815** for the training mean and **1.542** for the seven-day mean. Its RMSE is **1.804**, so the larger misses remain relevant even though the average absolute miss is smaller.

The seven-day mean has skill **−0.121**, meaning about **12.1% higher MAE** than persistence here. Smoothing the recent ratings did not help on these dates. That does not establish that rolling means are useless as model inputs: ridge can combine them with today's rating rather than replacing it outright. The training-mean baseline's negative $R^2$ also illustrates why $R^2$ and improvement over an operational baseline are different comparisons.

There are no missing current ratings or entirely missing seven-day windows in this validation block, so the fallback rules do not change these particular scores. They are still specified and checked for gaps elsewhere in the series. I now have a concrete benchmark for later validation forecasts: they should improve on persistence's **1.375 MAE** on the same dates. This table contains no 2026 scores.
""",

    "c3_results": r"""
**The shuffled result looks slightly better, but the measured difference is small.** On the same **745 forecasts**, with target dates from 12 June 2023 to 30 June 2025, shuffled CV gives ridge MAE **1.3803** and expanding CV gives **1.3984**. The gap is **0.0181 anxiety points**, about 1.3% of the temporal MAE. I should report that modest difference rather than turn the autocorrelation plot into a claim of dramatic score inflation.

The timing checks show why the shuffled protocol is unsuitable for my forecasting question regardless of the small score gap. For **99.9%** of these scored forecasts, its fit used at least one label observed after the issue evening. For **79.1%**, the held-out target rating also appears as current anxiety in a later training input row. Both rates are **0%** for the expanding folds. Keeping the scaler inside a pipeline prevents preprocessing leakage, but it cannot repair a split that lets the fitting process use the future.

The gap is not a clean estimate of the effect of autocorrelation alone. The methods use different training sizes and compositions, and conditions may change over time. Expanding fits grow from **154 to 750 rows**, whereas shuffled fits use **719 or 720**. Matching the scored dates and the mean-error alpha convention controls two differences, but not all of them. These five folds also do not establish statistical significance or guarantee the same gap on another period or model.

On the valid temporal CV predictions, persistence has MAE **1.5168**, the seven-day mean **1.5785**, and the training mean **1.6842**. Ridge's **1.3984** corresponds to **7.8% lower MAE than persistence** on those same forecasts. This is an encouraging development result for one fixed diagnostic alpha. It does not yet show that habits add value beyond anxiety history; that requires the feature ablation in Section 12. It is also not ridge's validation or final-evaluation score. In particular, I cannot compare this training-CV MAE directly with the **1.375** validation persistence MAE, because those refer to different dates.

The conclusion I will act on is methodological: use forward-in-time validation for model choices, and keep persistence in every comparable scorecard. I will tune ridge in Section 9 rather than treating this diagnostic fit as the chosen model.
""",

    "ridge_tuning": r"""
The lowest pooled CV MAE is **1.39774**, at **$\alpha=0.0316228$** in my mean-error convention. Fitting all 899 training rows therefore uses **$\alpha_{\mathrm{sklearn}}=28.4289$**. These are two conventions for the same penalty, not two separately selected settings.

The minimum is shallow. Near-zero alpha gives MAE about **1.39903**, only **0.00129 points** worse, while very strong regularization eventually raises it to **1.66723**. The data does not identify a sharply optimal penalty here; the grid gives me a reproducible development choice. The faint gray curves are the five individual folds, whose changing periods and sample sizes explain why their levels differ. They are not confidence bands.

I selected on MAE, not RMSE: the RMSE column need not have its minimum at the same alpha. This is also a selected CV score, so I will not treat its minimum as an unbiased estimate of the final model's error. The next step is the separate 2025 validation check, without changing alpha in response to that score.
""",

    "ridge_validation": r"""
On the **184 July–December 2025 forecasts**, ridge has **MAE 1.2955**, **RMSE 1.6293** and **$R^2=0.3445$**. Persistence has MAE **1.3750**, so ridge reduces the average absolute miss by **0.0795 anxiety points**, or **5.8%**, on exactly the same dates. It also beats the training-mean and seven-day-mean baselines in this block.

The skill calculation makes the relative gain explicit: $1-1.295514/1.375\approx0.0578$. An MAE of 1.2955 means the mean absolute distance between forecast and observed rating is about 1.30 points. It does not mean that each forecast lies within 1.30 points, or that 94.2% of forecasts are correct. RMSE is larger because squaring gives the bigger misses more influence. The positive $R^2$ says the model's squared errors are smaller than those from the validation outcomes' own mean, which is a different reference from persistence.

This is a modest improvement in average prediction, not a claim that anxiety is now predictable to a precise degree. No raw prediction falls outside 0–10, and I have not clipped the forecasts. The preprocessing-statistic checks confirm that medians and scales came from training rows.

The full feature set beat persistence here, but persistence is only one history-based rule. I still need a tuned history-only ridge comparison before attributing the gain to habits. Section 12 will make that comparison. Nor does a point-score improvement establish statistical significance for these dependent daily errors. The 2026 model evaluation remains reserved for Section 14.
""",

    "ridge_time_plot": r"""
The blue forecasts follow the broad changes but are less extreme than the observed ratings. This is visible around the September and November rises, where several high ratings extend above ridge's predictions, and during calm runs near zero, where ridge often stays above them. Persistence copies the preceding rating, so it can follow the level closely during a run while lagging a sudden change.

Those observations help me understand the score, but the overlay alone cannot tell me whether one rule is better. The aligned metric table measures that comparison. The next plot checks whether the average improvement hides larger errors on the days I most care about.
""",

    "ridge_level_errors": r"""
**The average improvement does not carry equally across anxiety levels.** For the **24 observed zero-rating days**, ridge overpredicts by **2.453 points on average**. For ratings **6, 7 and 8**, its mean biases are **−1.471, −2.702 and −3.037 points** respectively: it underpredicts the higher ratings. There are only **4 days rated 7** and **5 rated 8**, so I should not generalize those small groups too confidently. There are no validation targets rated 9 or 10.

This pattern is consistent with forecasts that are less extreme than the outcomes. It can reflect averaging under squared-error fitting, regularization and information absent from the inputs; this plot cannot isolate a single cause. It also groups by the outcome after it is observed, so it is a retrospective error analysis, not an early-warning rule I could apply tonight. Section 14 will examine high-anxiety detection explicitly.
""",

    "ridge_calibration": r"""
Most forecasts fall between **2 and 4**: this bin contains **117 days**, with mean prediction **2.916** and mean observed anxiety **2.581**. The **4–6** bin has **36 days**, with means **4.732** and **4.389**. Ridge overpredicts the bin average by about **0.34 points** in both groups. The lowest bin contains **29 days** and has a smaller average gap, **0.057 points**.

Only **two forecasts** fall in the 6–8 bin, and none reaches 8. I therefore have little evidence about calibration at high predicted levels. A point close to the diagonal with $n=2$ is not reassuring on its own.

This does not contradict the previous plot's underprediction of high observed ratings. Here I group by what the model predicted; there I grouped by what actually happened. A high-anxiety outcome that received a moderate forecast appears in a moderate forecast bin here. Bin averages can also conceal substantial individual errors, so I will read this check alongside MAE and the outcome-level error table.
""",

    "ridge_scratch_convergence": r"""
The numerical gradient check passed, and gradient descent reached its stopping tolerance after **556 updates**, using a step size of about **0.119452**. Its final largest absolute gradient component is **$9.8\times10^{-11}$**, below the specified $10^{-10}$ threshold.

The upper plot shows the objective falling to the direct solution's value. It looks flat early because later changes are small on that vertical scale. The lower plot makes those later steps visible: the gradient continues to shrink until it reaches the stopping threshold. The code also checks that the recorded loss does not increase beyond numerical tolerance.

The maximum difference between gradient-descent and scikit-learn validation predictions is only **$1.41\times10^{-10}$ points**. The direct solution agrees as well. These discrepancies are numerical, far below the precision of the self-ratings; both scratch solvers reproduce the fitted reference model.
""",

    "ridge_scratch_coefficients": r"""
The three coefficient columns agree to the displayed precision, and the `np.allclose` checks passed for coefficients, intercept and validation predictions. The scorecard is consequently the same for all three solvers: **MAE 1.295514** and **RMSE 1.629257**. Reimplementing ridge has verified the calculation, not created an additional performance improvement.

The intercept is **3.57842**, the training target mean after centering the fitted inputs. Current anxiety has weight **0.8223**, the three-day mean **0.5298**, and the seven-day mean **−0.1106**. Those weights belong to overlapping inputs; the negative seven-day coefficient does not mean that a higher recent average protects me from anxiety. It changes the prediction conditionally on the other included summaries.

For example, increasing standardized current anxiety by one while keeping the other 16 inputs fixed raises the fitted prediction by **0.8223 points**. On the raw input scale the change is $w_j\Delta v_j/s_j$, where $s_j$ is that input's training scale. A one-point change in the original rating is therefore not automatically a one-unit change in the standardized input. Also, changing current anxiety in a real daily record changes its rolling summaries too, so that one-coefficient calculation is a conditional reading of the equation, not the complete effect of changing a day's entry.

The intercept is the forecast at the vector of training feature means, where standardized inputs are all zero. That reference need not correspond to an actual observed evening, especially when some columns are binary flags. It sets the model's level; the weighted deviations determine each day's forecast.

The habit weights also need that care. Sleep adequacy is **−0.0280**, while sleep quality is **+0.1474** after the other inputs are included, despite the negative pooled sleep associations in Section 5. That is a different comparison: these are jointly fitted, regularized, standardized coefficients for tomorrow's target, not pooled same-evening correlations. Their signs do not establish that better sleep raises or lowers anxiety causally. Exercise and socializing have small positive weights here, **0.0215** and **0.0436**; their practical predictive contribution still needs the planned ablations.

Missingness and coverage coefficients describe the measurement pattern, not a habit intervention. For nonconstant binary indicators, a raw 0-to-1 change is not a one-standard-deviation change, as Section 7 explained. I will avoid ranking personal actions from coefficient magnitudes or signs, and examine uncertainty in Section 15 before making stronger coefficient claims.
""",

    "c4_forecasts": r"""
The controlled comparison uses **888 fitting targets, 17 January 2023–28 June 2025**, and the same **184 validation targets** for every configuration. The date audit confirms that even the three-day forecasts use only fitting labels available at their first issue time.

For **tomorrow**, $k=3$ has the lowest validation MAE, **1.2931**, compared with **1.2999, 1.2961 and 1.2959** for $k=1,7,14$. The difference between three and seven days is only about **0.003 anxiety points**. I would not call this compelling evidence that three days is a uniquely best window. All four improve on same-date persistence, whose MAE is **1.3750**.

At **three days ahead**, $k=7$ has the lowest MAE, **1.6121**, while the fourteen-day version is worse at **1.6695**. The horizon-matched persistence baseline has MAE **1.7880**, so the seven-day model has about **9.8% skill** relative to that harder baseline. Its absolute error is still larger than tomorrow's: a larger relative skill does not mean forecasting three days ahead is easier. The seven-day-mean baseline also changes with the issue date and has MAE **1.7345** at this horizon.

The main forecasting task remains $h=1$. I record $k=3$ as the lowest-scoring validation candidate and keep $k=7$ as the established reference for the next ablation. Keeping both roles explicit is more honest than declaring a strong window preference from a tiny difference or rewriting the earlier reference results. Alpha was retuned for each configuration; these scores compare complete learning procedures, not the isolated effect of adding one column at fixed weights.
""",

    "c4_same_day": r"""
The same-day lifestyle-and-calendar diagnostic has **11 inputs**, validation **MAE 1.0637**, **RMSE 1.3176** and **$R^2=0.5713$**. Its **41.5% skill is relative to the training mean**, not persistence. The check confirms that all four lookbacks produce identical input values on the common dates once anxiety-history columns are removed.

This supports the descriptive observation in Section 5 that same-evening habits move with anxiety. It does not establish causal effects or solve the forecasting question. Those measurements describe the same day as the outcome, and the timing of their recording does not establish that they were available before the anxiety rating. I will not carry this model into the tomorrow-forecast scorecard as a competing forecast.
""",

    "c1_results": r"""
**For this ridge procedure, adding the habit features does not improve on the history-and-calendar model.** On the same **745 outer-CV forecasts**, history plus calendar has MAE **1.3873**, compared with **1.4134** for the full model. Both beat persistence's **1.5168**, with skill about **8.5%** and **6.8%** respectively. Lifestyle plus calendar alone has MAE **1.5405**, slightly worse than persistence.

This is why the earlier full-model improvement over persistence was not enough to answer my central question. A more flexible history model can already explain an improvement over that simple rule. Here the full model's additional lifestyle inputs increase MAE by about **0.0261 points** relative to history plus calendar. That is a small observed difference, not a significance claim or a conclusion that habits do not matter in my life.

The domain-removal results point in the same direction: removing sleep, exercise or socializing gives MAE **1.4059, 1.3995 and 1.4112**, respectively. Their reduced-minus-full differences are **−0.0075, −0.0139 and −0.0022**. A negative bar means the reduced model has lower MAE; none of these removals hurts average accuracy in this comparison. The candidates were retuned separately, so this measures each domain's incremental value to the learning procedure, not a fixed coefficient being set to zero.

The socializing example also shows why counting “winning days” can mislead. The full model has smaller absolute error on about **54.2%** of dates than the model without socializing, yet its mean absolute error is slightly higher. The sizes of the remaining misses outweigh that majority. MAE uses the size of every error rather than only which model won the day.

Alpha varies across outer folds because each is selected from a different earlier period. The **1.4134** nested score is not expected to equal Section 9's **1.39774** selected training-CV minimum: that earlier minimum used the same CV targets to choose alpha, whereas each outer prediction here uses an alpha selected without its scored targets. Neither training-CV number should be compared directly with the **1.2955** validation MAE from July–December 2025, because those are different dates.

My answer so far is specific: **these recorded lifestyle features have not added predictive value beyond recent anxiety and calendar for this seven-day, one-day-ahead ridge setup.** I have not tested every possible representation or nonlinear interaction, and I have not estimated causal effects. The forest and its ablation in Section 13 will test whether a different model changes that predictive conclusion.
""",

    "forest_ablation": r"""
The **full forest** has the lowest validation MAE in this search, **1.3171**, using **depth 3, minimum leaf size 20, 80% of features per candidate subset, and 60 trees**. Its MAE is lower than history plus calendar's **1.3267**, but the difference is only about **0.0096 points**. Removing sleep, exercise or socializing gives **1.3196, 1.3216 and 1.3209**, respectively. The small differences provide only weak evidence of additional lifestyle value in this particular validation period.

This selected model uses relatively shallow trees: at depth 3, one tree has at most eight leaves. For the 17-column full input, the 0.8 setting initially considers $\lfloor0.8\times17\rfloor=13$ features at a node, extending the search if needed to find a valid partition. It is therefore a constrained nonlinear model, not an unrestricted tree memorizing every fitting day. The leaf-size rule and averaging further discourage very unstable predictions, although they can also smooth genuine peaks.

Lifestyle plus calendar alone performs worse, at **1.5049**, compared with persistence's **1.3750**. Recent anxiety remains useful even when the model can learn thresholds and interactions. The selected full forest has about **4.2% skill** relative to persistence, while Section 9's full ridge has validation MAE **1.2955**, slightly better than this forest. A more flexible model did not automatically improve the forecast.

This table does not overturn Section 12. That was a nested training-period ridge comparison; this is a separately tuned forest comparison on July–December validation. The different model, dates and selection procedure can produce different rankings. Selecting the smallest of six validation scores also creates selection optimism. I carry the full forest forward as a documented development choice, without treating these tiny removal effects as causal or statistically conclusive.
""",

    "forest_scratch": r"""
The deterministic tree and weighted-tree comparisons passed, as did the tests of seeded reproducibility, forest averaging and which trees contribute to each OOB prediction. On validation, the scratch forest has **MAE 1.3256**, compared with **1.3171** for scikit-learn. Their RMSE values are **1.6807** and **1.6686**. They behave similarly, but they are different random ensembles, so exact prediction equality would be the wrong test.

All **899 fitting rows** have OOB coverage here. The scratch OOB $R^2$ is **0.2942**, and scikit-learn's is **0.2887**. These are random-exclusion diagnostics within the training data, not chronological forecast scores. In particular, a higher OOB number is not a reason to switch implementations after seeing a preferred result. I keep the sklearn model as the main forest reference and report the scratch version separately.
""",

    "final_scorecard": r"""
The frozen comparison was refitted on **1,083 development targets** and evaluated on **282 observed 2026 targets**, from **1 January to 11 October**. Two missing target ratings are excluded from every score; their calendar positions remain gaps in the plots.

**Ridge has the lowest MAE among the reported procedures: 1.5373**, compared with **1.6206** for persistence. That is about **0.0833 anxiety points**, or **5.1%**, lower average absolute error. The sklearn forest has MAE **1.5460** and **4.6% skill**; the scratch forest has **1.5415** and **4.9% skill**. Ridge and the forests also improve RMSE over persistence, but their differences from each other are small. This scorecard does not establish a statistically reliable model-family ranking.

The arithmetic behind ridge's skill is $1-1.5373/1.6206\approx0.0514$. That is a reduction relative to a meaningful forecasting rule, not a 5.1-point change in the 0–10 rating and not a 5.1% chance of success. The gap between ridge and the sklearn forest is only about **0.0087 points**, much smaller than either model's typical miss. I would report that ordering without claiming it identifies a universally better model.

Both selected feature sets were full, based on the 2025 development comparisons. The history-plus-calendar ridge had validation MAE **1.3121**, slightly worse than the full ridge's **1.2955**, even though history won Section 12's nested training-CV comparison. That disagreement is evidence that the apparent contribution of habits is not stable across these comparisons. The final scorecard compares the selected forecasting procedures; it does not add a new final-period ablation that isolates habits.

The training-mean and seven-day-mean baselines have MAE **1.9734** and **1.8356**. Persistence is therefore still a substantially stronger benchmark than an overall mean. Final errors are higher than the earlier validation errors for all the main models. The later period contains more high ratings, and conditions can change; the scores alone do not isolate the cause. I will report the final values rather than replace them with the more favorable validation numbers or tune again on 2026.
""",

    "final_error_patterns": r"""
The time plots show the same limitation seen in validation: ridge and the shallow forest follow broad changes but smooth the extremes. Reindexing onto calendar days preserves the two missing-target gaps, rather than drawing invented observations through them.

For the **23 zero-rating days**, mean overprediction is **2.684 points for ridge** and **2.801 for the forest**. At observed level 8, the **15 days** have mean biases **−2.296** and **−2.488**. At level 9, the **7 days** have biases **−3.599** and **−3.453**. There are only **two level-10 days**, so those most extreme averages are particularly uncertain.

These are retrospective groups defined by the observed outcome. They show that lower overall MAE does not mean accurate prediction of bad-day severity. They do not by themselves tell me whether an advance warning was issued: that depends on a forecast cutoff fixed before final scoring. The next analysis checks that separate question.
""",

    "final_warnings": r"""
There are **72 high-anxiety target days out of 282 (25.5%)**. The frozen ridge cutoff is about **4.411**, the forest cutoff **4.413**, and persistence uses **6**. These thresholds came from validation F1, not the final curves.

Ridge warns on **91 days**: **51 true positives, 40 false positives and 21 missed high days**. Its precision is **56.0%** and recall **70.8%**. Persistence warns on **72 days**, with **46 true positives, 26 false positives and 26 misses**, giving **63.9% precision and recall**. The forest gives **49 true positives, 38 false positives and 23 misses**, with **56.3% precision and 68.1% recall**. Ridge catches five more high days than persistence but produces fourteen more false alarms. Persistence has the higher F1 here (**0.6389**, versus ridge's **0.6258** and the forest's **0.6164**), so no single warning rule wins every criterion.

For ridge, the complete count table is:

| Ridge decision | Observed rating ≥6 | Observed rating <6 |
|---|---:|---:|
| Warning | 51 true positives | 40 false positives |
| No warning | 21 false negatives | 170 true negatives |

Thus precision is $51/(51+40)$, recall is $51/(51+21)$, and F1 is $102/(102+40+21)\approx0.6258$. The 170 true negatives let me calculate a different quantity: among the 210 days below the high-anxiety threshold, the false-alarm rate is $40/(40+170)\approx19.0\%$. True negatives do not enter F1. Seeing the counts makes the tradeoff more concrete than a single decimal score.

There are **15 qualifying observed episodes**. Persistence covers **15/15**, ridge **14/15**, and the forest **13/15**. Persistence's perfect episode coverage is easy to overinterpret: for a run of at least two high days, copying today's rating can warn about its second day after the first high day has already happened. Coverage is not the same as anticipating onset.

These runs contain **60 of the 72 high days**. The other 12 high dates do not belong to a qualifying two-day observed run and still count in day-level recall. Episode recall gives each of the fifteen runs one vote, rather than giving its longer runs more weight. That explains why the day-level and episode rankings need not match.

The stricter observed-run onset counts are **1/15 for persistence**, **6/15 for ridge**, and **5/15 for the forest**. Even these need care. The run beginning **5 April** follows missing ratings on 3–4 April; the last recorded rating on 2 April was 8. Persistence's forward-fill fallback therefore warns about the resumed observed run, producing its one onset hit. That is not evidence of anticipating a genuinely new episode. Missing data can alter observed episode boundaries, and fifteen runs are a small sample.

These results support a limited conclusion: the fitted models modestly improve average rating forecasts, and the chosen ridge warning catches more high days at a cost in false alarms. They do not provide dependable prediction of every severe day or a validated clinical warning system. I keep the cutoffs and model choices fixed rather than selecting a more favorable point from the final precision–recall curves.
""",

    "uncertainty_results": r"""
**The most stable positive slopes describe recent anxiety.** Current anxiety has a development coefficient of **0.8331**, with conditional credible interval **[0.6335, 1.0327]** and 14-day block interval **[0.6343, 1.0121]**. The three-day mean is **0.4930**, with intervals **[0.2778, 0.7083]** and **[0.2952, 0.6864]**. Both methods retain their positive signs. The seven-day mean's individual intervals cross zero; because the history inputs overlap, that does not show that a week of history is useless.

**Most habit slopes are small or have unresolved signs.** Sleep adequacy is **−0.1085**, exercise **0.0032**, socializing **0.0275**, and sleep shortfall **0.0377**, all per development standard deviation. Both interval types cross zero for each. I cannot resolve their individual conditional signs in this analysis. That is different from proving they have no relationship with anxiety or no value to me.

Sleep quality needs an honest explanation. Its fitted slope is **+0.1700**, with credible interval **[0.0383, 0.3017]** and block interval **[0.0387, 0.2945]**. The positive sign persists under both methods, even though Section 5 showed a negative pooled association. These are different comparisons: here I hold current anxiety, overlapping history summaries, sleep adequacy and other inputs fixed while predicting the next evening. Shared information among predictors, conditioning and the outcome timing can change a coefficient's sign. These calculations do not identify which explanation produced this sign, and they do not show that better sleep causes greater anxiety. The coefficient also does not establish a useful improvement in held-out error; that requires the ablation evidence.

**The bootstrap is not uniformly wider.** For current anxiety its width is about **0.95 times** the posterior width; for sleep quality about **0.97**, and for exercise about **1.09**. The sleep-adequacy missingness flag is more sensitive: its width ratio is about **1.59**, and the block interval crosses zero even though its conditional credible interval does not. A rare observation pattern is a weak basis for a personal conclusion.

Residual correlations at lags **1, 7 and 14 are approximately 0.017, 0.030 and −0.005**. The plotted lags are small, so strong autocorrelation in the raw ratings does not automatically imply equally strong dependence after fitting anxiety-history features. This does not establish independent, Gaussian errors or rule out nonlinear and longer-range dependence. It does mean I should not force the original expectation that all block intervals would be wider into the explanation.

The bootstrap calendar has **1,089 positions and six missing target positions**. Keeping those gaps gives **1,070–1,089 usable sampled pairs** per replicate, with median **1,083**. The median interval widths for **7-, 14- and 28-day blocks are 0.2869, 0.2843 and 0.2952**. Each run has three slopes whose intervals exclude zero, but equal counts alone do not establish identical endpoints or sensitivity for every feature. The sensitivity runs are also smaller and have more Monte Carlo variation. I report the declared 14-day result without claiming an optimal block length or a definitive sign test.

For my original question, the main lesson is that the stable history coefficients and the mixed habit ablations fit together. A stable conditional sleep-quality sign is something to describe carefully, not a reason to reverse the ablation conclusion or recommend changing my sleep.
""",

    "audit_method_results": r"""
Both median-regression feature sets select **$\gamma=0.01$** on the training folds. On the 184 validation dates, history plus calendar has **MAE 1.2928**, compared with **1.2955** for the earlier full ridge reference: a difference of only **0.0027 rating points**, about **0.2% of ridge's validation MAE**. The full median regression is slightly worse at **1.3004**. Adding habits therefore does not improve this median-regression comparison either.

This makes median regression a reasonable candidate for a future comparison, not a clear replacement for ridge. The small gain could change with the period or penalty search; I have not estimated a confidence interval for that paired difference. It also changes the feature set, loss and penalty together when compared with the full ridge reference, so I cannot credit the difference to the loss alone. None of these additional candidates has a new independent final score.

The training-median baseline has validation MAE **1.6141**, better than the training mean's **1.8153**, but worse than persistence's **1.3750**. That is a useful check: a median is a stronger constant benchmark for an MAE comparison, yet it does not remove the need to beat a forecast that uses recent anxiety. Its training-CV and validation errors describe different periods, as with the earlier baselines.
""",

    "median_scratch_results": r"""
The scratch median optimizer passes the small outlier checks at both zero and positive penalty. On the real training inputs, its primal–dual gaps are approximately **$7.66\times10^{-7}$** for history plus calendar and **$4.31\times10^{-7}$** for the full feature set, both below the declared $10^{-6}$ objective tolerance. It reaches that check after **5,925** and **20,025** updates, respectively.

Its validation predictions differ from sklearn by at most about **0.000085** and **0.000794 rating points**. The corresponding scratch MAEs are **1.292769** and **1.300366**, compared with **1.292769** and **1.300354** for the library solver. These small discrepancies are consistent with finite optimization tolerance; they are not extra improvements in forecasting. The training objectives agree within the explicit tolerance, and the dual lower bound checks how close the scratch solution is to an optimum of the same objective.

The library comparison remains visible because it helps test the implementation. Neither solver uses 2026 to learn the median-regression parameters or assess this additional candidate.
""",

}

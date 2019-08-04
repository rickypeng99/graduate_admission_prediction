1. Introduction
===============

The dataset of this project is downloaded from:  
<https://www.kaggle.com/mohansacharya/graduate-admissions>  
which is brought by:  
Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of
Regression Models for Prediction of Graduate Admissions, IEEE
International Conference on Computational Intelligence in Data Science
2019

This dataset includes possibly import parameters that can affect the
admission to graduate schools in the U.S.

The parameters contain the following:

-   `GRE Score`
-   `TOEFL Score`
-   `University Rating` which denotes the rating of the applicant's
    undergraduate university
-   `SOP` which stands for Statement of Purpose
-   `LOR` which stands for letter of recommendation
-   `CGPA` which stands for the culmulative GPA
-   `Research` if the student has done an internship/equivalent research
-   `Chance of Admit` The chance to be admitted

Getting a master degree is difficult and expensive. This project gives
every student a chance to evaluate themselves by predicting their
chances to get admitted, and thus decide if they should consider apply
for a graduate education or not.

Therefore, we aim to build a model that can perform good predictions on
the chances of admission based on different parameters given by the
student.

2. Methods
==========

### 2.1 Loading and checking the data

    admission = read.csv('Admission_Predict_Ver1.1.csv', header = TRUE)
    sum(is.na(admission))

    ## [1] 0

    #checking data types of each column
    sapply(admission, function(x){
      return(typeof(x))
    })

    ##        Serial.No.         GRE.Score       TOEFL.Score University.Rating 
    ##         "integer"         "integer"         "integer"         "integer" 
    ##               SOP               LOR              CGPA          Research 
    ##          "double"          "double"          "double"         "integer" 
    ##   Chance.of.Admit 
    ##          "double"

    #admission$Research = as.factor(admission$Research)
    admission = subset(admission, select = -c(Serial.No.))

There isn't any missing value in the dataset, but the `serial no.` is
useless for modeling.

### 2.2 Collinearity of the data

    library(faraway)
    pairs(admission, col = "dodgerblue")

![](README_files/figure-markdown_strict/unnamed-chunk-2-1.png)

    range(cor(admission)[cor(admission) < 1])

    ## [1] 0.3725256 0.8824126

The predictors averagely have high correlations with each other which
ranges from 0.37 to 0.88. This indicates that we need variable selecting
techniques to reduce the side effects resulted form multicollinearity.

### 2.3 Setting up evaluating methods

    library(lmtest)

    ## Warning: package 'lmtest' was built under R version 3.4.4

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    get_bp_decision = function(model, alpha) {
      decide = unname(bptest(model)$p.value < alpha)
      ifelse(decide, "Reject", "Fail to Reject")
    }

    get_sw_decision = function(model, alpha) {
      decide = unname(shapiro.test(resid(model))$p.value < alpha)
      ifelse(decide, "Reject", "Fail to Reject")
    }

    get_num_params = function(model) {
      length(coef(model))
    }

    get_loocv_rmse = function(model) {
      sqrt(mean((resid(model) / (1 - hatvalues(model))) ^ 2))
    }

    get_adj_r2 = function(model) {
      summary(model)$adj.r.squared
    }

    get_prec_err = function(predicted, actual) {
      mean(abs(actual - predicted) / predicted)
    }

    plot_fitted_resid = function(model, pointcol = "dodgerblue", linecol = "darkorange") {
      plot(fitted(model), resid(model), 
           col = pointcol, pch = 20, cex = 1.5,
           xlab = "Fitted", ylab = "Residuals", main = "Fitted vs residual graph")
      abline(h = 0, col = linecol, lwd = 2)
    }

    plot_qq = function(model, pointcol = "dodgerblue", linecol = "darkorange") {
      qqnorm(resid(model), col = pointcol, pch = 20, cex = 1.5)
      qqline(resid(model), col = linecol, lwd = 2)
    }

### 2.4 Randomly split the dataset into training and testing dataset

    set.seed(1999)
    admission_trn_idx  = sample(nrow(admission), size = trunc(0.50 * nrow(admission)))
    admission_trn_data = admission[admission_trn_idx, ]
    admission_tst_data = admission[-admission_trn_idx, ]

### 2.5 Trying to fit model

We first start with the full additive model:

    model_additive_all = lm(Chance.of.Admit ~ ., data =  admission)

    summary(model_additive_all)

    ## 
    ## Call:
    ## lm(formula = Chance.of.Admit ~ ., data = admission)
    ## 
    ## Residuals:
    ##       Min        1Q    Median        3Q       Max 
    ## -0.266657 -0.023327  0.009191  0.033714  0.156818 
    ## 
    ## Coefficients:
    ##                     Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)       -1.2757251  0.1042962 -12.232  < 2e-16 ***
    ## GRE.Score          0.0018585  0.0005023   3.700 0.000240 ***
    ## TOEFL.Score        0.0027780  0.0008724   3.184 0.001544 ** 
    ## University.Rating  0.0059414  0.0038019   1.563 0.118753    
    ## SOP                0.0015861  0.0045627   0.348 0.728263    
    ## LOR                0.0168587  0.0041379   4.074 5.38e-05 ***
    ## CGPA               0.1183851  0.0097051  12.198  < 2e-16 ***
    ## Research           0.0243075  0.0066057   3.680 0.000259 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.05999 on 492 degrees of freedom
    ## Multiple R-squared:  0.8219, Adjusted R-squared:  0.8194 
    ## F-statistic: 324.4 on 7 and 492 DF,  p-value: < 2.2e-16

    vif(model_additive_all)

    ##         GRE.Score       TOEFL.Score University.Rating               SOP 
    ##          4.464249          3.904213          2.621036          2.835210 
    ##               LOR              CGPA          Research 
    ##          2.033555          4.777992          1.494008

According to the summary of this full additive model, the p\_value
testing whether *β*<sub>*S**O**P*</sub> is zero or not is 0.728263, a
relatively large p\_value. This represents that we would highly possibly
fail to reject the null hypothesis of *β*<sub>*S**O**P*</sub> = 0, which
means SOP might be a meaningless predictor.

So, we decided to firstly remove the predictor "SOP".

    model_add_sop_rem = lm(Chance.of.Admit ~ . - SOP, data =  admission)

#### 2.5.1 AIC / BIC backwards on SOP-removed full additive model

Then we run AIC and BIC on this SOP-removed full additive model.

    model_add_aic = step(model_add_sop_rem, trace = 0)
    model_add_bic = step(model_add_sop_rem, k = log(nrow(admission_trn_data)), trace = 0)
    vif(model_add_aic)

    ##         GRE.Score       TOEFL.Score University.Rating               LOR 
    ##          4.459541          3.867976          2.265898          1.853078 
    ##              CGPA          Research 
    ##          4.619554          1.493400

    vif(model_add_bic)

    ##   GRE.Score TOEFL.Score         LOR        CGPA    Research 
    ##    4.452473    3.799455    1.704623    4.376495    1.486588

#### 2.5.2 AIC / BIC backwards on 2-way interactive model

Another strong candidate for a starting model would be a model using all
possible predictor as well as their 2-way interaction.

    model_int_2 = lm(Chance.of.Admit ~ .^2, data = admission_trn_data)
    vif(model_int_2)

    ##                     GRE.Score                   TOEFL.Score 
    ##                    2548.26844                    6019.55927 
    ##             University.Rating                           SOP 
    ##                    4632.15483                    5020.26079 
    ##                           LOR                          CGPA 
    ##                    3661.67589                    8688.12355 
    ##                      Research         GRE.Score:TOEFL.Score 
    ##                    2452.99450                   22104.65276 
    ##   GRE.Score:University.Rating                 GRE.Score:SOP 
    ##                   11511.22746                   13496.09388 
    ##                 GRE.Score:LOR                GRE.Score:CGPA 
    ##                   11409.29471                   22121.03423 
    ##            GRE.Score:Research TOEFL.Score:University.Rating 
    ##                    5891.73439                    3869.62397 
    ##               TOEFL.Score:SOP               TOEFL.Score:LOR 
    ##                    4271.97371                    3081.62336 
    ##              TOEFL.Score:CGPA          TOEFL.Score:Research 
    ##                   11996.43111                    1784.86492 
    ##         University.Rating:SOP         University.Rating:LOR 
    ##                     182.90275                     173.62664 
    ##        University.Rating:CGPA    University.Rating:Research 
    ##                    4390.22279                      51.86702 
    ##                       SOP:LOR                      SOP:CGPA 
    ##                     201.68924                    4655.45494 
    ##                  SOP:Research                      LOR:CGPA 
    ##                      73.95331                    2267.57063 
    ##                  LOR:Research                 CGPA:Research 
    ##                      48.65011                    1675.40371

We then also run AIC and BIC on this model, as there are a lot variables
with high vif.

    model_int_2_aic = step(model_int_2, trace = 0)
    model_int_2_bic = step(model_int_2, k = log(nrow(admission_trn_data)), trace = 0)

### 2.6 Comparison of the four models (AIC\_ADD, BIC\_ADD, AIC\_INT, BIC\_INT)

    ## additive
    # aic
    model_full_aic = cbind(get_bp_decision(model_add_aic, 0.05), get_sw_decision(model_add_aic, 0.05),
    get_num_params(model_add_aic),
    get_loocv_rmse(model_add_aic),
    get_adj_r2(model_add_aic),
    get_prec_err(predict(model_add_aic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit))
    # bic
    model_full_bic = cbind(get_bp_decision(model_add_bic, 0.05), 
    get_sw_decision(model_add_bic, 0.05), 
    get_num_params(model_add_bic), 
    get_loocv_rmse(model_add_bic), 
    get_adj_r2(model_add_bic), 
    get_prec_err(predict(model_add_bic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit))

    # 2-way interactive
    model_int_aic = cbind(get_bp_decision(model_int_2_aic, 0.05), 
    get_sw_decision(model_int_2_aic, 0.05), 
    get_num_params(model_int_2_aic), 
    get_loocv_rmse(model_int_2_aic), 
    get_adj_r2(model_int_2_aic), 
    get_prec_err(predict(model_int_2_aic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit))

    model_int_bic = cbind(get_bp_decision(model_int_2_bic, 0.05), 
    get_sw_decision(model_int_2_bic, 0.05), 
    get_num_params(model_int_2_bic), 
    get_loocv_rmse(model_int_2_bic), 
    get_adj_r2(model_int_2_bic), 
    get_prec_err(predict(model_int_2_bic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit))

    result = rbind(model_full_aic, model_full_bic, model_int_aic, model_int_bic)
    row.names(result) = c("Additive_aic", "Additice_bic", "Interactice_aic", "Interactive_bic")


    library(knitr)
    library(kableExtra)

    ## Warning: package 'kableExtra' was built under R version 3.4.4

    result %>%
      kable(col.names = c("Bptest decision", "Shapiro decision", "# of parameters", "loocv_rmse", "adjusted r squared", "percentage error error")) %>%
      kable_styling()

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:left;">
Bptest decision
</th>
<th style="text-align:left;">
Shapiro decision
</th>
<th style="text-align:left;">
\# of parameters
</th>
<th style="text-align:left;">
loocv\_rmse
</th>
<th style="text-align:left;">
adjusted r squared
</th>
<th style="text-align:left;">
percentage error error
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Additive\_aic
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
7
</td>
<td style="text-align:left;">
0.0603423316682804
</td>
<td style="text-align:left;">
0.819688923884365
</td>
<td style="text-align:left;">
0.0660836985824088
</td>
</tr>
<tr>
<td style="text-align:left;">
Additice\_bic
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
6
</td>
<td style="text-align:left;">
0.0604304625870431
</td>
<td style="text-align:left;">
0.818844872845129
</td>
<td style="text-align:left;">
0.0669002647886271
</td>
</tr>
<tr>
<td style="text-align:left;">
Interactice\_aic
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
0.0567539938908104
</td>
<td style="text-align:left;">
0.822197175373317
</td>
<td style="text-align:left;">
0.0667456808250702
</td>
</tr>
<tr>
<td style="text-align:left;">
Interactive\_bic
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
6
</td>
<td style="text-align:left;">
0.0568870818740569
</td>
<td style="text-align:left;">
0.818575903908821
</td>
<td style="text-align:left;">
0.0669174707479586
</td>
</tr>
</tbody>
</table>
Comparing the above 4 models, we deicide to pick the 2-way interactive
model that is selected by AIC backwards.

### 2.7 Discovering model violations

Now we need to address with the model violations in this 2-way
interactive model

    plot_fitted_resid(model_int_2_aic)

![](README_files/figure-markdown_strict/unnamed-chunk-12-1.png)

    plot_qq(model_int_2_aic)

![](README_files/figure-markdown_strict/unnamed-chunk-12-2.png)

The normal qq plot and the fitted\_resid plot for the model doesn't seem
too good. In addition, we also reject the sw and bp tests. We hope to
see if the violation of normality and constant variance is resulted from
unusual observations

### 2.8 Outlier diagnosis

#### 2.8.1 Influential points

Influential points are measured by the cook's distance, typically, an
observation is classfied as influential if its cook's distance is larger
than 4/n. We therefore decided to remove the influential observations
from the training dataset and see if the results will be better.

    cook_int_aic = cooks.distance(model_int_2_aic)
    new_int_2_aic = lm(Chance.of.Admit ~ GRE.Score + TOEFL.Score + University.Rating + 
        SOP + LOR + CGPA + Research + GRE.Score:CGPA + University.Rating:SOP, 
        data = admission_trn_data, subset = cook_int_aic < 4 / length(cook_int_aic))

    plot_fitted_resid(new_int_2_aic)

![](README_files/figure-markdown_strict/unnamed-chunk-13-1.png)

    plot_qq(new_int_2_aic)

![](README_files/figure-markdown_strict/unnamed-chunk-13-2.png)

As the previous graphs haven shown, after the removal of influential
points, the normal qq plot and the fitted vs residual graph look much
better. Denoting that the original dataset didn't follow a normal
distribution and a constant variance assumpition. Therefore, we should
not be surprised with the violations.

    new_int_2_aic_row = cbind(get_bp_decision(new_int_2_aic, 0.05), 
    get_sw_decision(new_int_2_aic, 0.05), 
    get_num_params(new_int_2_aic), 
    get_loocv_rmse(new_int_2_aic), 
    get_adj_r2(new_int_2_aic), 
    get_prec_err(predict(new_int_2_aic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit))

    result = rbind(new_int_2_aic_row)
    row.names(result) = c("Without influential")

    library(knitr)
    library(kableExtra)

    result %>%
      kable(col.names = c("Bptest decision", "Shapiro decision", "# of parameters", "loocv_rmse", "adjusted r squared", "percentage error error")) %>%
      kable_styling()

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:left;">
Bptest decision
</th>
<th style="text-align:left;">
Shapiro decision
</th>
<th style="text-align:left;">
\# of parameters
</th>
<th style="text-align:left;">
loocv\_rmse
</th>
<th style="text-align:left;">
adjusted r squared
</th>
<th style="text-align:left;">
percentage error error
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Without influential
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
Fail to Reject
</td>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
0.0458113255715612
</td>
<td style="text-align:left;">
0.87566002668485
</td>
<td style="text-align:left;">
0.0639575535605795
</td>
</tr>
</tbody>
</table>
The testing results from the model trainied by dataset without
influential points also seem to be better than before.

#### 2.8.2 Leverages

Leverages from the orifinal data can also violate the model assumption.

    hat_values_model = hatvalues(model_int_2_aic)[hatvalues(model_int_2_aic) > 2 * mean(hatvalues(model_int_2_aic))]
    length(hat_values_model) / length(hatvalues(model_int_2_aic))

    ## [1] 0.056

Although the amount of leverages aren't significant, we will still try
to remove the leverages from the dataset and see if the model can be
improved.

    new_int_2_aic_no_leverage = lm(Chance.of.Admit ~ GRE.Score + TOEFL.Score + University.Rating + 
        SOP + LOR + CGPA + Research + GRE.Score:CGPA + University.Rating:SOP, 
        data = admission_trn_data, subset = cook_int_aic < 4 / length(cook_int_aic) || hatvalues(model_int_2_aic) < 2 * mean(hatvalues(model_int_2_aic)))

    plot_fitted_resid(new_int_2_aic_no_leverage)

![](README_files/figure-markdown_strict/unnamed-chunk-16-1.png)

    plot_qq(new_int_2_aic_no_leverage)

![](README_files/figure-markdown_strict/unnamed-chunk-16-2.png)

    new_int_2_aic_no_leverage_row = cbind(get_bp_decision(new_int_2_aic_no_leverage, 0.05), 
    get_sw_decision(new_int_2_aic_no_leverage, 0.05), 
    get_num_params(new_int_2_aic_no_leverage), 
    get_loocv_rmse(new_int_2_aic_no_leverage), 
    get_adj_r2(new_int_2_aic_no_leverage), 
    get_prec_err(predict(new_int_2_aic_no_leverage, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit))

    result = rbind(new_int_2_aic_no_leverage_row)
    row.names(result) = c("Without influential & leverages")
    library(knitr)
    library(kableExtra)

    result %>%
      kable(col.names = c("Bptest decision", "Shapiro decision", "# of parameters", "loocv_rmse", "adjusted r squared", "percentage error error")) %>%
      kable_styling()

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:left;">
Bptest decision
</th>
<th style="text-align:left;">
Shapiro decision
</th>
<th style="text-align:left;">
\# of parameters
</th>
<th style="text-align:left;">
loocv\_rmse
</th>
<th style="text-align:left;">
adjusted r squared
</th>
<th style="text-align:left;">
percentage error error
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Without influential & leverages
</td>
<td style="text-align:left;">
Fail to Reject
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
0.0570272375404827
</td>
<td style="text-align:left;">
0.819747526254511
</td>
<td style="text-align:left;">
0.065741117965527
</td>
</tr>
</tbody>
</table>
#### 2.8.3 Comparison between removing influential points & removing both of influential and leverages

    result = rbind(new_int_2_aic_row ,new_int_2_aic_no_leverage_row)
    row.names(result) = c("Without influential", "Without influential & leverages")

    library(knitr)
    library(kableExtra)

    result %>%
      kable(col.names = c("Bptest decision", "Shapiro decision", "# of parameters", "loocv_rmse", "adjusted r squared", "percentage error error")) %>%
      kable_styling()

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:left;">
Bptest decision
</th>
<th style="text-align:left;">
Shapiro decision
</th>
<th style="text-align:left;">
\# of parameters
</th>
<th style="text-align:left;">
loocv\_rmse
</th>
<th style="text-align:left;">
adjusted r squared
</th>
<th style="text-align:left;">
percentage error error
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Without influential
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
Fail to Reject
</td>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
0.0458113255715612
</td>
<td style="text-align:left;">
0.87566002668485
</td>
<td style="text-align:left;">
0.0639575535605795
</td>
</tr>
<tr>
<td style="text-align:left;">
Without influential & leverages
</td>
<td style="text-align:left;">
Fail to Reject
</td>
<td style="text-align:left;">
Reject
</td>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
0.0570272375404827
</td>
<td style="text-align:left;">
0.819747526254511
</td>
<td style="text-align:left;">
0.065741117965527
</td>
</tr>
</tbody>
</table>
Despite the fact that the removed leverages and influtiential points
didn't reject to have a constant variance. The model that is trained by
only removing the influential points win by having much better testing
results and failure to reject the shapiro decision.

3. Result
---------

### 3.1 Decision

Based on the analysis abvove, we decided to choose the model that is
built based on the AIC backwards on full 2-way interactive model, and
the removal of influential data from the training dataset. In detail,
the model will be illustrated as the following:

*C**h**a**n**c**e* = *β*<sub>0</sub> + *β*<sub>*G**R**E*</sub>*x*<sub>1</sub> + *β*<sub>*T**O**E**F**L*</sub>*x*<sub>2</sub> + *β*<sub>Uni\_Rating</sub>*x*<sub>3</sub> + *β*<sub>*S**O**P*</sub>*x*<sub>4</sub> + *β*<sub>*L**O**R*</sub>*x*<sub>5</sub> + *β*<sub>*C**G**P**A*</sub>*x*<sub>6</sub> + *β*<sub>*R**e**s**e**a**r**c**h*</sub>*x*<sub>7</sub> + *β*<sub>GRE.Score:CGPA</sub>*x*<sub>8</sub> + *β*<sub>University.Rating:SOP</sub>*x*<sub>9</sub>

### 3.2 Summary of the "best" model

Below is the result of the best model's predicted value vs the test
data. We can oberserve that the model is surely doing a good job in
fitting the values. Most of the combination of prediction vs testing
lies around the red line of y = x

    plot(predict(new_int_2_aic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit, xlab = "predicted values", ylab = "actual test values", main = "Predicted vs testing data graph")
    abline(0, 1, col = "red")

![](README_files/figure-markdown_strict/unnamed-chunk-18-1.png)

The average percentage error of the "best" model is:

    get_prec_err(predict(new_int_2_aic, newdata = admission_tst_data), admission_tst_data$Chance.of.Admit)

    ## [1] 0.06395755

A 6% percentage error is pretty good.

Below is the summary of the "best" model:

Estimates of parameters from the model:

    summary(new_int_2_aic)$coefficients[,1]

    ##           (Intercept)             GRE.Score           TOEFL.Score 
    ##         -2.5813632363          0.0063032154          0.0024523039 
    ##     University.Rating                   SOP                   LOR 
    ##         -0.0114238779         -0.0015859251          0.0178277731 
    ##                  CGPA              Research        GRE.Score:CGPA 
    ##          0.2901647916          0.0328999525         -0.0005569089 
    ## University.Rating:SOP 
    ##          0.0034242290

R\_squared:

    summary(new_int_2_aic)$r.squared

    ## [1] 0.8804018

Adjusted r\_squared:

    summary(new_int_2_aic)$adj.r.squared

    ## [1] 0.87566

4.Discussion
------------

### 4.1 Context of the model

Accoding to the model, each of the *β* values are listed as the
following:

*β*<sub>0</sub> = `-2.5813632`  
*β*<sub>*G**R**E*</sub> = `0.0063032`  
*β*<sub>*T**O**E**F**L*</sub> = `0.0024523`  
*β*<sub>Uni\_rating</sub> = `-0.0114239`  
*β*<sub>*S**O**P*</sub> = `-0.0015859`  
*β*<sub>*L**O**R*</sub> = `0.0178278`  
*β*<sub>*C**G**P**A*</sub> = `0.2901648`  
*β*<sub>Research</sub> = `0.0329000`  
*β*<sub>GRE.Score:CGPA</sub> = `-0.0005569`  
*β*<sub>University.Rating:SOP</sub> = `0.0034242`

For each of the beta values denoting the amount of average chance of
admission change given a one unit change of a specific predictor.
Notably, as `Research` itself is a binary response, it makes
*β*<sub>*R**e**s**e**a**r**c**h*</sub> only useful in calculation only
when a student has a research.

In our common sense, we would believe that all the parameters should be
positively impacting the chance of admission, as they are good traits of
students that admission office is looking for. By conencting such common
sense to our model, we do notice that the important factors such as
`GRE`,`TOEFL`, `CGPA` be hugely positively affecting the chance of
admission, which is similar to what we would have believed.

However, variables like `uni_rating`, which denotes the level of the
applicant's undergrad university appears to be negative. This is unusual
as we typically believe that better undergrad degree should somewhat aid
an applicant for a better grad degree. The result of this migh give us
several suggestions, such that the American undergrad admission system
possibly lacks the ability of distinguishing students, which makes
stronger students distributed in a wide range of universities. For
instance, despite the fact that the UC Berkeley is considered as a
better public university of Illinois in many ranking methodologies,
there still can be brighter students who goes to U of illinois instead
of UC Berkeley. However, it is also possible that the
`university_rating` doesn't affect too much in the graduate admission,
such that it becomes slightly negative.

### 4.2 Applications

#### 4.2.1 Usefulness

As stated in the result, our model has achieved a overall good fitting.
Therefore, assuming the dataset is real, then we should have a good
estimate of the chances of admission based on the newdata that we input.

For instance, let's say that Ricky, who is from University of Illinois,
has an extremely high GRE TOEFL, and CGPA, he also has several letters
of recommendations, statements of purposes, and he has participated in
research as an undergrad. Ricky should be having a decent chance of
getting into a grad shcool

    predict(new_int_2_aic, newdata = data.frame(GRE.Score = 335, TOEFL.Score = 115,  University.Rating = 5, SOP = 5.0, LOR = 5.0, CGPA = 9.8, Research = 1)) 

    ##         1 
    ## 0.9701073

As a result, Ricky has a 97% chance to get into a grad school. A good
student like Ricky should absolutely achieve a result like this.

#### 4.2.2 Standardized regression coefficients

The standardized regression coefficient denotes the relative influence
of the parameters. In particular, this value represents the expected
number of standard deviation change in the dependent variable from one
standard deviation change in the predictor.

    estimates = summary(new_int_2_aic)$coefficients[,1]
    standard_deviations = sapply(admission_trn_data, function(x){
        return(sd(x))
    })
    poolSD = standard_deviations / sd(admission_trn_data$Chance.of.Admit)
    #gre toefl uni_rat sop lor cgpa research
    standardized_coefficient = c(poolSD[1] * estimates[2],
    poolSD[2] * estimates[3],
    poolSD[3] * abs(estimates[4]),
    poolSD[4] * abs(estimates[5]),
    poolSD[5] * estimates[6],
    poolSD[6] * estimates[7],
    poolSD[7] * estimates[8])

    standardized_coefficient

    ##         GRE.Score       TOEFL.Score University.Rating               SOP 
    ##        0.52329782        0.11072894        0.09513329        0.01135993 
    ##               LOR              CGPA          Research 
    ##        0.12266396        1.29897092        0.12423685

According to the standardardized\_coefficient calculted above, we
immediately know that the the `CGPA` and `GRE` are the most important
parameters, besides from these two significant parameters, the letters
of recommendation and TOEFL score are also important.

5. Appendix
-----------

Summary of the best model:

    summary(new_int_2_aic)

    ## 
    ## Call:
    ## lm(formula = Chance.of.Admit ~ GRE.Score + TOEFL.Score + University.Rating + 
    ##     SOP + LOR + CGPA + Research + GRE.Score:CGPA + University.Rating:SOP, 
    ##     data = admission_trn_data, subset = cook_int_aic < 4/length(cook_int_aic))
    ## 
    ## Residuals:
    ##       Min        1Q    Median        3Q       Max 
    ## -0.134431 -0.027643  0.002316  0.028608  0.103025 
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)           -2.5813632  1.2870863  -2.006   0.0461 *  
    ## GRE.Score              0.0063032  0.0040963   1.539   0.1253    
    ## TOEFL.Score            0.0024523  0.0009497   2.582   0.0104 *  
    ## University.Rating     -0.0114239  0.0110542  -1.033   0.3025    
    ## SOP                   -0.0015859  0.0102060  -0.155   0.8767    
    ## LOR                    0.0178278  0.0043794   4.071 6.47e-05 ***
    ## CGPA                   0.2901648  0.1518377   1.911   0.0573 .  
    ## Research               0.0329000  0.0070936   4.638 5.94e-06 ***
    ## GRE.Score:CGPA        -0.0005569  0.0004772  -1.167   0.2445    
    ## University.Rating:SOP  0.0034242  0.0030751   1.114   0.2667    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.04494 on 227 degrees of freedom
    ## Multiple R-squared:  0.8804, Adjusted R-squared:  0.8757 
    ## F-statistic: 185.7 on 9 and 227 DF,  p-value: < 2.2e-16

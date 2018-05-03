# l1-penalized-ordinal-polytomous-regression-estimators


The function that chooses lambda (the penalization) then performs the regression is the "Main" function of the "choice_of_ lambda_v7" module.
The user only have to provide X (the matrix of explanatory variables) and y (the variable to explain) to use the "Main" function with the default options. For example, "res = Main (X, y)"
The output is composed of the support (i.e. the index of the useful explanatory variables) then beta and finally gamma.
By default, the "Main" function uses the Quantile universal threshold. This can be changed in the optional arguments of "Main".  If the user knows what lambda value he wants to use, he can also directly use the "lonepoly" function in the "ordinal_polychotomous_regression_v7" module to perform the regression.

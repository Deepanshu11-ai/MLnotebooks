# import numpy as np
# import pandas as pd

# # for reproducibility
# np.random.seed(42)

# # number of samples
# n_samples = 200

# # generate feature X
# X = np.random.uniform(0, 10, n_samples)

# # true relationship (unknown to model)
# true_slope = 3.5
# true_intercept = 2.0

# # noise
# noise = np.random.normal(0, 2, n_samples)

# # target variable
# y = true_slope * X + true_intercept + noise

# # create DataFrame
# df = pd.DataFrame({
#     "X": X,
#     "y": y
# })

# # save to csv
# df.to_csv("svm_regression_data.csv", index=False)

# print("Dataset generated and saved as svm_regression_data.csv")
# print(df.head())

import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 200

# Feature
X = np.random.uniform(0, 10, n_samples)

# True underlying relationship
true_slope = 3.5
true_intercept = 2.0

# Noise
noise = np.random.normal(0, 2, n_samples)

# Label (target)
y = true_slope * X + true_intercept + noise

# Labeled dataset
df = pd.DataFrame({
    "Feature_X": X,
    "Target_y": y
})

df.head()
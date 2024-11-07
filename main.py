#%% 0. LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.max_columns', None)

#%% I. DATA PREPROCESSING


###--- Import and filter data ---###
df = pd.read_csv('ml_datasource.csv')

# Examine general infor
df.head()
df.info()

# Examine distribution
df_no_outliers = df[(df['minutes_watched'] < 1000) & 
   (df['courses_started'] < 10) &
   (df['practice_exams_started'] < 10) & 
   (df['minutes_spent_on_exams'] < 40)
   ]

df_no_outliers.info()


###--- Plot analysis ---###
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))

sns.kdeplot(data=df_no_outliers['days_on_platform'], ax=axes[0, 0])

sns.kdeplot(data=df_no_outliers['minutes_watched'], ax=axes[0, 1])

sns.kdeplot(data=df_no_outliers['courses_started'], ax=axes[1, 0])

sns.kdeplot(data=df_no_outliers['practice_exams_started'], ax=axes[1, 1])

sns.kdeplot(data=df_no_outliers['practice_exams_passed'], ax=axes[2, 0])

sns.kdeplot(data=df_no_outliers['minutes_spent_on_exams'], ax=axes[2, 1])

plt.show()


###--- Examine multicollinearity ---###
# Check correlation heatmap
correlation = df_no_outliers.select_dtypes('number').corr()
sns.heatmap(correlation, cmap='coolwarm')
plt.show()

# Check variance inflation factor
variance_inflation_factor(df_no_outliers.iloc[:, 1:], 6)


















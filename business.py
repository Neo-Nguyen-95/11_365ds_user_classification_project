import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor


# %% I. Data Preprocessing

class EDA:
    
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def load_raw_data(self):
        return self.df
        
    def load_processed_data(self):
        # Remove outlier
        result = self.df[(self.df['minutes_watched'] < 1000) & 
           (self.df['courses_started'] < 10) &
           (self.df['practice_exams_started'] < 10) & 
           (self.df['minutes_spent_on_exams'] < 40)
        ]
        
        # Remove multicollinearity column
        # Result: highly correlated columns
        # practice_exams_started, practice_exams_passed, minutes_spent_on_exams
        result = result.drop(
            columns=['practice_exams_started']
            )
        
        # Fill NaN value with 'NAM' country name
        result = result.fillna('NAM')
        
        return result
    
    def plot_cols(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

        sns.kdeplot(data=self.load_processed_data()['days_on_platform'], 
                    ax=axes[0, 0])

        sns.kdeplot(data=self.load_processed_data()['minutes_watched'], 
                    ax=axes[0, 1])

        sns.kdeplot(data=self.load_processed_data()['courses_started'], 
                    ax=axes[1, 0])

        sns.kdeplot(data=self.load_processed_data()['minutes_spent_on_exams'], 
                    ax=axes[1, 1])

        plt.show()
        
    def plot_corr(self):
        correlation = self.load_processed_data().select_dtypes('number').corr()
        sns.heatmap(correlation, cmap='coolwarm')
        plt.show()
        
    def check_vif(self):
        df_number = self.load_processed_data().select_dtypes('number')

        for n in range(len(df_number.columns)):
            vif = variance_inflation_factor(df_number, n)
            print(f'VIF of {df_number.columns[n]} is: {vif:.2f}')
        
        
        
        
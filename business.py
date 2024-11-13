import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder, MinMaxScaler, OneHotEncoder
    )

import numpy as np


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
            columns=['practice_exams_started', 'practice_exams_passed']
            )
        
        # Fill NaN value with 'NAM' country name
        result = result.fillna('NAM')
        
        return result
    
    def plot_cols(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

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
            
    def load_encoded_data(self, method):
        df = self.load_processed_data()
        
        target = ['purchased']
        y = df[target]
        X = df.drop(columns=target)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=365, stratify=y
            )
        
        y_train_array = y_train.to_numpy()
        y_test_array = y_test.to_numpy()
        
        if method == 'ordinal':

            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=175)
            X_train['student_country_enc'] = encoder.fit_transform(
                X_train['student_country']
                .to_numpy()
                .reshape(-1, 1)
                )
            X_train.drop(columns='student_country', inplace=True)
            X_train_array = X_train.to_numpy()
            
    
            X_test['student_country_enc'] = encoder.transform(
                X_test['student_country']
                .to_numpy()
                .reshape(-1, 1)
                )
            X_test.drop(columns='student_country', inplace=True)
            X_test_array = X_test.to_numpy()
            
        # --- NOT DONE YET ---
        elif method == 'one-hot':
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_train_country_array = encoder.fit_transform(
                X_train[['student_country']]
                )

            X_train_nocountry_array = (X_train.drop(columns='student_country')
                                       .to_numpy()
                                       )

            X_train_array = np.concatenate(
                (X_train_country_array, X_train_nocountry_array), 
                axis=1
                )
            
            X_test_country_array = encoder.transform(
                X_test[['student_country']]
                )
            
            X_test_nocountry_array = (X_test.drop(columns='student_country')
                                       .to_numpy()
                                       )
            
            X_test_array = np.concatenate(
                (X_test_country_array, X_test_nocountry_array), 
                axis=1
                )
            
        
        return X_train_array, X_test_array, y_train_array, y_test_array
        
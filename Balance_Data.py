import pandas as pd
import numpy as np
from sklearn.utils import resample

def balance(df):
    df_class_1 = df[df.Activity == 1]
    df_class_2 = df[df.Activity == 2]
    df_class_3 = df[df.Activity == 3]
    df_class_0 = df[df.Activity == 0]

    minority_class = min(len(df_class_1), len(df_class_2), len(df_class_3), len(df_class_0))

    df_class_1_resampled = resample(df_class_1,
                                replace=False,
                                n_samples=minority_class,
                                random_state=42)
    df_class_2_resampled = resample(df_class_2,
                                replace=False,
                                n_samples=minority_class,
                                random_state=42)
    df_class_3_resampled = resample(df_class_3,
                                replace=False,
                                n_samples=minority_class,
                                random_state=42)
    df_class_0_resampled = resample(df_class_0,
                                replace=False,
                                n_samples=minority_class,
                                random_state=42)

    df_balanced = pd.concat([df_class_0_resampled, df_class_1_resampled, df_class_2_resampled, df_class_3_resampled])

    df_balanced = df_balanced.sample(frac=1, random_state=42)
    return(df_balanced)
path = "/home/oc/agentic_ml/dataset/train.csv"

import pandas as pd
from tools.data.leakage import detect_leakage

from tools.data.profile import inspect_dataset, profile_columns, detect_target
from tools.data.quality import check_data_quality
from tools.data.leakage import detect_leakage

df = pd.read_csv(path)

inspection = inspect_dataset(df)
profile = profile_columns(df)
quality = check_data_quality(df, profile=profile)
target_result = detect_target(df, profile=profile, objective="predict customer churn")


print(target_result)
# print(quality)
# print(inspection)
# print(profile)

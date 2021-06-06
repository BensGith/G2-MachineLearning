import pandas as pd
from processing.basic_process import basic_process
df = pd.read_csv('train.csv')
df = basic_process(df)
final_data = []
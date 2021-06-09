from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from grid_search.grid_search import Grid
from processing.basic_process import basic_process
from classifiers.ANN import ann
from processing.OneHotEncoder import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from processing.test_process import predict_test
from processing.test_process import write_csv
import pandas as pd




###### Test ###########
# Process the training set
train_df = pd.read_csv('train.csv')
train_df = basic_process(train_df, train=True)
labels = train_df['label']
train_df.drop(columns=['label'], axis=1, inplace=True)
encoder = OneHotEncoder()
imupter = IterativeImputer()
scaler = MinMaxScaler()
pca = PCA(0.98, svd_solver='full')
stages = [encoder, imupter, scaler, pca]
for stage in stages:
    train_df = stage.fit_transform(train_df)
df = pd.DataFrame(train_df)
grid = Grid(train_df, labels)
print(grid.find_best_params())
# ann.fit(df, labels)
#
# test_df = pd.read_csv('test_without_target.csv')
# predictions = predict_test(df, ann, encoder, imupter, scaler, pca)
# write_csv(predictions)
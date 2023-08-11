# Genetic Algorithm
from GA import GA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_label(x):
    if x> 0.008:
        return 4
    elif x > 0.003 and x<= 0.008:
        return 3
    elif x < -0.008:
        return 0
    elif x>= -0.008 and x < -0.003:
        return 1
    else:
        return 2


data = pd.read_csv('df_508056_fac_shift_1.csv',index_col=[0])
data = data.dropna(subset=data.columns[-1])
data = data.fillna(data.mean())

data['label'] = data['508056.SH_close_chg'].apply(get_label)

train = data[:int(len(data)*0.7)]
test = data[int(len(data)*0.7):]

X_train = train.iloc[:,:-2]
Y_train = train['label']
X_test = test.iloc[:,:-2]
Y_test = test['label']

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Selector = GA(X_train, Y_train, X_test, Y_test, model_type = 1, save_computaion=True)
Selector.Search()
Filter = Selector.bestSolutions
Best_accuracy = Selector.bestAccuracy
X_train_masked = X_train[:, Filter==1]
X_test_masked = X_test[:, Filter==1]

data_masked = data.loc[:, list(Filter == 1) + [False, False]]
feature_selected = list(data_masked.columns)

print("Best Accuracy is {:.4f}".format(Best_accuracy))
pd.DataFrame(feature_selected).to_csv('feature_selected_GA_RF.csv')
#Ev Fiyat Tahmin Modeli

#İş Problemi

#Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
#farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
#gerçekleştirilmek istenmektedir.

#Veri Seti Hikayesi

#Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir
#yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle
#yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları
#boş bırakılmış olup, bu değerleri sizin tahmin etmeniz beklenmektedir.

#Toplam Gözlem 1460

#Sayısal Değişken 38

#Kategorik Değişken 43

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz
import missingno as msno

warnings.simplefilter(action='ignore', category=Warning)

#Görev 1
#Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
train = pd.read_csv("data/house_train.csv")
test = pd.read_csv("data/house_test.csv")

train.head()
train.shape
test.head()
test.shape

df = pd.concat((train, test)).reset_index(drop=True)

def check_df(dataframe, head=5):
    print("############Shape############")
    print(dataframe.shape)
    print("############Types############")
    print(dataframe.dtypes)
    print("############Tail############")
    print(dataframe.tail(head))
    print("############Head############")
    print(dataframe.head(head))
    print("############NA############")
    print(dataframe.isnull().sum())
    print("############Quantiles############")
    print(dataframe.describe([0,0.05, 0.25, 0.50, 0.75, 0.95,0.99,1]).T)

check_df(df)

for i in df.columns:
    print(df[i].value_counts())
    print('***************************************\n')

df.drop('Id', axis = 1, inplace=True)


#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=13, car_th=20):
    cat_cols= [col for col in df.columns if str(df[col].dtypes) in ["category","object", "bool"]]
    num_but_cat=[col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["int", "float"]]
    cat_but_car= [col for col in df.columns if
                   df[col].nunique()>20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols= cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols= [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols= [col for col in num_cols if col not in cat_cols]

    print(f"Observations): {dataframe.shape[0]}")
    print(f"Veriables): {dataframe.shape[1]}")
    print(f"cat_cols): {len(cat_cols)}")
    print(f"num_cols): {len(num_cols)}")
    print(f"cat_but_car): {len(cat_but_car)}")
    print(f"num_but_cat): {len(num_but_cat)}")

    return cat_cols, num_cols,cat_but_car, num_but_cat

cat_cols, num_cols,cat_but_car,num_but_cat= grab_col_names(df)

#Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df.dtypes

#Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def num_summary(dataframe, numerical_col, plot= False):
    quantiles= [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    dataframe[numerical_col].describe().T
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)

for col in num_cols:
    num_summary(df, col, plot=True)

    num_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool": \
            dataframe[col_name] = dataframe[col_name].astype(int)
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("###########################################################")


for col in cat_cols:
    cat_summary(df, col, plot=True)

    cat_summary(df, col, plot=True)


# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_summary_cat(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


target_summary_cat(df, "SalePrice", cat_cols)


# Adım 6: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))



def check_outlier_graph(dataframe, col_name, plot= False):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        print(True)
        if plot:
            sns.boxplot(dataframe[col_name])
            plt.show(block= True)
            print(dataframe[col_name])
            print("###########################################################")
    else:
        print(False)
        print(dataframe[col_name])
        print("###########################################################")

for col in num_cols:
    check_outlier_graph(df, col, plot=True)


#Adım 7: Eksik gözlem var mı inceleyiniz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    msno.bar(df)
    plt.show()
    if na_name:
        return na_columns


missing_values_table(df, True)


f, ax = plt.subplots(figsize=[14, 8])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f")
ax.set_title("Correlation Matrix", fontsize=25)
plt.show()

#Görev 2: Feature Engineering
df.head()

#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
for col in num_cols:
    replace_with_thresholds(df, col)

df.describe()

df.isnull().sum()


df["PoolQC"].fillna("NPQC", inplace=True)
df["MiscFeature"].fillna("NPMF", inplace=True)
df["Alley"].fillna("NPA", inplace=True)
df["Fence"].fillna("NPF", inplace=True)
df["FireplaceQu"].fillna("NPFP", inplace=True)

df=df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" and x.name != "SalePrice" else x, axis=0)
df=df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
df = pd.DataFrame(df, columns=df.columns)
df.head()

#kategorik değişkenlerin kendi kendi içindeki dağılımları
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


    def rare_analyser(dataframe, target, cat_cols):
        for col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


    rare_analyser(df, "SalePrice", cat_cols)


    def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()

        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

        for var in rare_columns:
            tmp = temp_df[var].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

        return temp_df


    # rare oranı 0.01 bu oranın altında kalan sınıfları birleştirir
    new_df = rare_encoder(df, 0.01)

    # Adım 3: Yeni değişkenler oluşturunuz.
    new_df["NEW_1st*GrLiv"] = new_df["1stFlrSF"] * new_df["GrLivArea"]

    new_df["NEW_Garage*GrLiv"] = (new_df["GarageArea"] * new_df["GrLivArea"])

    new_df["TotalQual"] = new_df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                                  "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual",
                                  "GarageCond", "Fence"]].sum(axis=1)  # 42

    # Total Floor
    new_df["NEW_TotalFlrSF"] = new_df["1stFlrSF"] + new_df["2ndFlrSF"]  # 32

    # Total Finished Basement Area
    new_df["NEW_TotalBsmtFin"] = new_df.BsmtFinSF1 + df.BsmtFinSF2  # 56

    # Porch Area
    new_df["NEW_PorchArea"] = new_df.OpenPorchSF + new_df.EnclosedPorch + new_df.ScreenPorch + new_df[
        "3SsnPorch"] + new_df.WoodDeckSF  # 93

    # Total House Area
    new_df["NEW_TotalHouseArea"] = new_df.NEW_TotalFlrSF + new_df.TotalBsmtSF  # 156

    new_df["NEW_TotalSqFeet"] = new_df.GrLivArea + new_df.TotalBsmtSF  # 35

    # Lot Ratio
    new_df["NEW_LotRatio"] = new_df.GrLivArea / new_df.LotArea  # 64

    new_df["NEW_RatioArea"] = new_df.NEW_TotalHouseArea / new_df.LotArea  # 57

    new_df["NEW_GarageLotRatio"] = new_df.GarageArea / new_df.LotArea  # 69

    # MasVnrArea
    new_df["NEW_MasVnrRatio"] = new_df.MasVnrArea / new_df.NEW_TotalHouseArea  # 36

    # Dif Area
    new_df["NEW_DifArea"] = (new_df.LotArea - new_df[
        "1stFlrSF"] - new_df.GarageArea - new_df.NEW_PorchArea - new_df.WoodDeckSF)  # 73

    new_df["NEW_OverallGrade"] = new_df["OverallQual"] * new_df["OverallCond"]  # 61

    new_df["NEW_Restoration"] = new_df.YearRemodAdd - new_df.YearBuilt  # 31

    new_df["NEW_HouseAge"] = new_df.YrSold - new_df.YearBuilt  # 73

    new_df["NEW_RestorationAge"] = new_df.YrSold - new_df.YearRemodAdd  # 40

    new_df["NEW_GarageAge"] = new_df.GarageYrBlt - new_df.YearBuilt  # 17

    new_df["NEW_GarageRestorationAge"] = np.abs(new_df.GarageYrBlt - new_df.YearRemodAdd)  # 30

    new_df["NEW_GarageSold"] = new_df.YrSold - new_df.GarageYrBlt  # 48

#%%
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]
new_df.drop(drop_list, axis = 1, inplace=True)

# Adım 4: Encoding işlemlerini gerçekleştiriniz.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in new_df.columns if new_df[col].dtype not in [int, float]
               and new_df[col].nunique() == 2]

label_cols = [col for col in new_df.columns if new_df[col].dtype not in [int, float]
              and new_df[col].nunique() > 2]

for col in binary_cols:
    label_encoder(new_df, col)

for col in label_cols:
    label_encoder(new_df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in new_df.columns if 10 >= new_df[col].nunique() > 2]

one_hot_encoder(new_df, ohe_cols)

new_df.head()


cat_cols, num_cols,cat_but_car,num_but_cat= grab_col_names(df)

test_dataframe = new_df[new_df['SalePrice'].isnull()]
test_dataframe.head()

train_dataframe = new_df[new_df['SalePrice'].notnull()]
train_dataframe.head()

train_dataframe.isnull().sum().sum()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


y = train_dataframe["SalePrice"]
X = train_dataframe.drop(["SalePrice"], axis=1)

train_dataframe["SalePrice"] = train_dataframe["SalePrice"].astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



#Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400, 500, 1500],
               "colsample_bytree": [0.5, 0.7, 0.9, 0.8, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

lgbm_gs_best.best_params_

cv_results = cross_validate(final_model, X, y, cv=5, scoring="neg_mean_squared_error")
cv_results

cv_results['fit_time'].mean()

cv_results['score_time'].mean()


cv_results['test_score'].mean()

#Adım 4: Değişken önem düzeyini inceleyeniz.
#Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit etmeye uygun halde bir
#dataframe oluşturup sonucunuzu yükleyiniz
#%%
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(final_model, X_train)




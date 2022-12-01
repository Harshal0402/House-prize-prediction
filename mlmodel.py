import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\DELL\Downloads\MCSReal_Estate.csv")
df.head()
def repalce(col):
    if "$" in col:
        return float(col.replace("$"," "))
    else:
        return (float(col.replace("Rs"," ")))/75

df["Price"]=df["Price"].apply(repalce)

def State(add):
    add_list = add.split(",")[-1]
    State = add_list.split()
    return State[-2]

df["State"] = df["Address"].apply(State)

df.drop("Avg Area Comfort",axis=1 ,inplace=True)


from sklearn.impute import SimpleImputer
Si = SimpleImputer(missing_values=np.nan, strategy="mean")
df[["Avg. Area Number of Bedrooms"]] = Si.fit_transform(df[["Avg. Area Number of Bedrooms"]])
df.drop(["Address","ids"],axis=1 ,inplace=True)
df.head

df["Avg. Area House Age"].replace("missing", np.nan, inplace=True)
df["Avg. Area Number of Rooms"].replace("?", np.nan, inplace=True)

df["Avg. Area House Age"] = df["Avg. Area House Age"].astype("float64")
df["Avg. Area Number of Rooms"] = df["Avg. Area Number of Rooms"].astype("float64")

Si = SimpleImputer(missing_values=np.nan, strategy="mean")
df[["Avg. Area House Age","Avg. Area Number of Rooms"]] = Si.fit_transform(df[["Avg. Area House Age","Avg. Area Number of Rooms"]])

from sklearn.preprocessing import OrdinalEncoder

Or =OrdinalEncoder()
col = df.select_dtypes("object").columns
df[col]=Or.fit_transform(df[col])

x = df.drop("Price", axis=1)
y = df.iloc[: , -2]


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.30, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

from sklearn.svm import SVR


def mymodel(model):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    return model

def makeprediction():
    svr = SVR()
    model = mymodel(svr)
    return model

def makeprediction():
    svr = SVR(C = 1000, kernel = 'linear')
    model = mymodel(svr)
    return model



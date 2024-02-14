import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import streamlit as st

st.title(":green[Walmart] Sales Explanation")

col1, col2 = st.columns([0.3, 0.7])

#All functions
def data_prepare(df):
    df["Super_Bowl"] = 0
    df.loc[(df['Date'] == '12-02-2010')|(df['Date'] == '11-02-2011')|(df['Date'] == '10-02-2012'),'Super_Bowl'] = 1

    df["Labor_Day"] = 0
    df.loc[(df['Date'] ==  '10-09-2010')|(df['Date'] == '09-09-2011')|(df['Date'] == '07-09-2012'),'Labor_Day'] = 1

    df["Thanksgiving"] = 0
    df.loc[(df['Date'] == '26-11-2010')|(df['Date'] == '25-11-2011'),'Thanksgiving'] = 1

    df["Christmas"] = 0
    df.loc[(df['Date'] == '31-12-2010')|(df['Date'] == '30-12-2011'),'Christmas'] = 1

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df['week'] =df['Date'].dt.isocalendar().week
    df['month'] =df['Date'].dt.month 
    df['year'] =df['Date'].dt.year
    return df

def drop_cols(df, drop_columns=["Date", "Holiday_Flag", "Unemployment"], target_column="Weekly_Sales"):
    return df.drop(columns=(drop_columns+[target_column]))

def train_lr_pipeline(df, target_column, categorical_cols=None, drop_columns=None, shuffle=False):
    df = df.copy()
    X = drop_cols(df)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=shuffle, random_state=7)
    import numpy as np

    # Preprocess 
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    continuous_transformer = MinMaxScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('cont', continuous_transformer, X.columns.difference(categorical_cols))  # Apply scaler to continuous features
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lr', LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mean_percentage_error = mean_absolute_percentage_error(y_test, y_pred)
    return pipeline, mean_percentage_error, [X_test, y_test]

def get_lr_equation(pipeline, df, target_column="Weekly_Sales", drop_columns=["Date", "Holiday_Flag"]):
    df.drop(columns=drop_columns, inplace=True)
    df.drop(columns=[target_column], inplace=True)
    preprocessor = pipeline.named_steps['preprocessor']
    categorical_cols = preprocessor.transformers_[0][2]
    encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = list(encoded_feature_names)  # Include one-hot encoded categorical variables
    
    continuous_cols = list(pipeline.named_steps['preprocessor'].transformers_[1][2])
    feature_names += continuous_cols  
    
    # feature_names.append('const') 
    
    lr_model = pipeline.named_steps['lr']
    
    coefficients = lr_model.coef_
    
    intercept = lr_model.intercept_
    
    equation_parts = [f"{coef:.2f} * {feat}" for coef, feat in zip(coefficients, feature_names)]
    equation = f"y = {intercept:.2f} + {' + '.join(equation_parts)}"

    effect = pd.DataFrame({"weights" : coefficients, "features" : feature_names})
    effect.loc[len(effect)] = {"weights" : intercept, "features": "intercept"}
    effect['weights'] = effect['weights'].astype(int)
    return equation, effect


#Modeling
@st.cache_data
def read_data():
    df_org = pd.read_csv('Walmart.csv')
    return df_org
df_org = read_data()
print(df_org.shape)
df = data_prepare(df_org.copy())
lr_pipe, error, test_df = train_lr_pipeline(df.copy(), categorical_cols=["Store"], target_column="Weekly_Sales", drop_columns=["Date", "Holiday_Flag"], shuffle=True)
print("Mean Percentage Error:", error)
eqn, weights = get_lr_equation(lr_pipe, df,target_column="Weekly_Sales", drop_columns=["Date", "Holiday_Flag"])


with col1:
    store_num = int(st.text_input(label="Store Number", value=2))
    date = st.text_input("Date", value='26-11-2010')
    temp = int(st.text_input("Temperature", value="10"))
    fuel_price = int(st.text_input("Fuel Price", value="4"))
    CPI = int(st.text_input("CPI", value="180"))

#Getting the test input
test_input = {
    'Store': store_num,
    'Date' : date,
    'Weekly_Sales':None,
    "Holiday_Flag" : 1,
 'Temperature': temp,
 'Fuel_Price': fuel_price,
 'CPI': CPI,
 'Unemployment': 8.9}

test_input_df = data_prepare(pd.DataFrame([test_input]))
X_test = drop_cols(test_input_df)

y_test_pred = lr_pipe.predict(X_test)
print(y_test_pred)
X_test["prediction"] = y_test_pred
X_test["type_"] = "sample"


df = data_prepare(df_org.copy())
mean_input = df[(df.Store == test_input['Store']) & (df.Holiday_Flag == 0)].drop(columns=["Date", "Holiday_Flag"]).median()
y_mean = mean_input['Weekly_Sales']
mean_input.pop('Weekly_Sales')
X_mean = pd.DataFrame(mean_input).transpose() #pd.DataFrame(mean_input).reset_index().rename(columns={"index":"features", 0:'mean_values'})

y_mean_pred = lr_pipe.predict(X_mean)
print(y_mean_pred)
X_mean["prediction"] = y_mean_pred
X_mean["type_"] = "sample_mean"

final1 = pd.concat([X_mean, X_test])

def normalize(X, pred_value, type_, store_number=test_input['Store'], lr_pipe=lr_pipe):
    df = pd.DataFrame(
    data    = lr_pipe["preprocessor"].transform(X).toarray(),
    columns = lr_pipe["preprocessor"].get_feature_names_out()
    ).transpose()
    df.index = df.index.map(lambda x: x.split('__')[1])
    df["Store_number"] = df.index.map(lambda x: x.split('_')[1] if 'Store' in x else 0)
    df = df[df.Store_number.astype(int).isin([test_input['Store'],0])]
    df.drop(columns=["Store_number"], inplace=True)
    df.index = df.index.where(~df.index.str.contains('Store', case=False), 'Store')
    df = df.transpose()
    df["prediction"] = pred_value
    df["type_"] = type_ + "_norm"
    return df

X_test_norm = normalize(X=X_test, type_ ="sample", pred_value = y_test_pred)
X_mean_norm = normalize(X=X_mean, type_ = "sample_mean", pred_value = y_mean_pred)
final = pd.concat([final1, X_test_norm, X_mean_norm])
 
weights["store_number"] = weights.features.apply(lambda x: x.split('_')[1] if 'Store' in x else 0)
weights = weights[weights.store_number.astype(int).isin([test_input['Store'],0])].reset_index(drop=True)
weights.loc[weights.features.str.contains('Store'), "features"] = "Store"
weights.drop(columns=["store_number"], inplace=True)

final2 = pd.concat([weights.set_index("features").transpose().assign(type_ = "weights").reset_index(drop=True), final]).set_index("type_").transpose()
final2["delta"] = final2.sample_norm - final2.sample_mean_norm
final2["effect"] = final2.weights * final2.delta

#Scaling down volumn from Mn to 100k
final2.loc["prediction"] /= 1000
final2["effect"] /= 1000
final2 = final2.sort_values("effect")

#Plotting
plot_df = final2.dropna(subset=["effect"]).loc[final2['effect'] != 0]
plt.figure(figsize=(8, 6))
x_start = final2.loc["prediction"]["sample_mean_norm"]
x_end = final2.loc["prediction"]["sample_norm"]
cum_sum = x_start 

plt.axvline(x=x_start, color=(0, 1-0/8, 0.1+0/8, 1), linestyle='--')
plt.axvline(x=x_end, color=(0, 1-7/8, 0.1+7/8, 1), linestyle='--') 
plt.text(x_start, 1.5, 'Median prediction for \n the given store', verticalalignment='center')
plt.text(x_end, 0.5, 'Prediction for given store \n with input parameters', verticalalignment='center')

for i, (index, row) in enumerate(plot_df.iterrows()):
    x = row['effect']
    cum_sum += x 
    col = (0, 1-i/8, 0.1+i/8, 1)
    plt.arrow(x_start, i, x+0.001, 0, head_width=0.1,width=0.025, head_length=8, ec=col)  # Add arrow
    x_start = cum_sum
    plt.yticks(range(len(plot_df)), plot_df.index)


plt.xlabel('Sales (in $100k)')
plt.ylabel('Feature')
plt.title('Effect Plot for Linear Regression')
plt.grid(True)
fig = plt.gcf()
# plt.show()

with col2:
    st.pyplot(fig)
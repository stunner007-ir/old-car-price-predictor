import pandas as pd
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

car = pd.read_csv("Cleaned Car.csv")

X = car.drop(columns='Price')
y = car['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
var = ohe.categories_
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2_score(y_test, y_pred)


def predict_car_price(Car_Model_Name, Car_Company_Name, Car_Purchase_Year, Nos_of_KM_driven, Fuel_Type):
    a = Car_Model_Name
    b= Car_Company_Name
    c = Car_Purchase_Year
    d = Nos_of_KM_driven
    e = Fuel_Type
    pipe = make_pipeline(column_trans, lr)
    y_pred = pipe.predict(
        pd.DataFrame([[a, b, c, d, e]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

    return y_pred


app = gr.Interface(
    fn=predict_car_price,
    inputs=['text', 'text', 'number', 'number', 'text'], outputs=['number'],
    title="Car Price Predictor",
    description="This Model will predict the price of a old Car.")

app.launch(share=True)

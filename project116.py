import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score 


df = pd.read_csv("project116.csv")

grescore = df["GRE Score"].tolist()

toeflscore = df["TOEFL Score"].tolist()

admission  = df["Chance of admit"].tolist()

colors = []

for i in admission:
    if i == 1:
        colors.append("green")
    else :
        colors.append("red")


fig = go.Figure(data=go.Scatter(
    x=toeflscore,
    y=grescore,
    mode='markers',
    marker=dict(color=colors)
))

fig.show()

x = df[["TOEFL Score","GRE Score"]]

y = df["Chance of admit"]


# splitting the data in 75% and 25% ratio to train and test

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25)

# initializing the sc to give a score to the factors
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)

lr = LogisticRegression(random_state=0)

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

print("Accuracy : " , accuracy_score(y_test,y_pred))

gre = int(input("Enter your GRE Score : "))

toefl = int(input("Enter your TOEFL Score "))

user_test = sc.transform([[gre,toefl]])

user_pred = lr.predict(user_test)

if user_pred[0] == 1:
    print("The student may get admission ")
else:
    print("The student might not get admission ")






from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


path = '/Users/shinhyunwoo/Downloads/project/'
df = pd.read_csv(path + 'medical_noshow.csv')
print('Count of rows', str(df.shape[0]))
print('Count of Columns', str(df.shape[1]))

df.isnull().any().any()

for i in df.columns:
    print(i+":", len(df[i].unique()))


df['PatientId'].astype('int64')
df.set_index('AppointmentID', inplace=True)
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

df['PreviousApp'] = df.sort_values(
    by=['PatientId', 'ScheduledDay']).groupby(['PatientId']).cumcount()
df['PreviousNoShow'] = (df[df['PreviousApp'] > 0].sort_values(['PatientId', 'ScheduledDay']).groupby(
    ['PatientId'])['No-show'].cumsum() / df[df['PreviousApp'] > 0]['PreviousApp'])

df['PreviousNoShow'] = df['PreviousNoShow'].fillna(0)
df['PreviousNoShow']

# Number of Appointments Missed by Patient
df['Num_App_Missed'] = df.groupby(
    'PatientId')['No-show'].apply(lambda x: x.cumsum())
df['Num_App_Missed']

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.strftime('%Y-%m-%d')
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['ScheduledDay']

df['AppointmentDay'] = pd.to_datetime(
    df['AppointmentDay']).dt.strftime('%Y-%m-%d')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['AppointmentDay']

df['Day_diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Day_diff'].unique()

df = df[(df.Age >= 0)]
df.drop(['ScheduledDay'], axis=1, inplace=True)
df.drop(['AppointmentDay'], axis=1, inplace=True)
df.drop('PatientId', axis=1, inplace=True)
df.drop('Neighbourhood', axis=1, inplace=True)

# Convert to Categorical
df['Handcap'] = pd.Categorical(df['Handcap'])
# Convert to Dummy Variables
Handicap = pd.get_dummies(df['Handcap'], prefix='Handicap')
df = pd.concat([df, Handicap], axis=1)
df.drop(['Handcap'], axis=1, inplace=True)

df = df[(df.Age >= 0) & (df.Age <= 100)]
df.info()

X = df.drop(['No-show'], axis=1)
y = df['No-show']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)
clf_report = classification_report(y_test, y_pred_rd_clf)

print(f"Classification Report : \n{clf_report}")


accuracy = cross_val_score(estimator=rd_clf, X=X, y=y, cv=8)
print("avg acc: ", np.mean(accuracy))
print("acg std: ", np.std(accuracy))


accuracy = cross_val_score(estimator=rd_clf, X=X, y=y, cv=8)
print("avg acc: ", np.mean(accuracy))
print("acg std: ", np.std(accuracy))

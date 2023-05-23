#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[22]:


# 1. 데이터
path = '../_data/'
datasets = pd.read_csv(path + 'medical_noshow.csv')
x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]

x = x.drop(['PatientId', 'AppointmentID'], axis=1)

x = x.fillna(np.NaN)    

# 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index)    # 오브젝트 컬럼 리스트 추출
print(ob_col)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)
# no = 0 , yes = 1

x = x.fillna(np.NaN)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 8
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# scler 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[23]:


param = {
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'colsample_bylevel': [0.5, 0.7, 1.0],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.5, 0.7, 1.0],
    'random_strength': [0.1, 0.5, 1, 5],
    'bagging_temperature': [0.1, 0.5, 1, 5],
    'border_count': [5, 10, 20, 50]
    }


# In[24]:


# 2. 모델
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
lgbm = CatBoostClassifier()
model = RandomizedSearchCV(lgbm, param, cv = kfold, verbose = 1, refit = True, n_jobs = -1, n_iter=2048, random_state=42)


# In[25]:


import time
start_time = time.time()
# 3. 훈련
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('최적의 파라미터 : ', model.best_params_)
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')


# In[ ]:


# Fitting 8 folds for each of 2048 candidates, totalling 16384 fits
# 0:	learn: 0.6705645	total: 113ms	remaining: 56.5s
# 1:	learn: 0.6496777	total: 149ms	remaining: 37.2s
# 2:	learn: 0.6325623	total: 174ms	remaining: 28.8s
# 3:	learn: 0.6154500	total: 237ms	remaining: 29.4s
# 4:	learn: 0.6010493	total: 283ms	remaining: 28.1s
# 5:	learn: 0.5899976	total: 295ms	remaining: 24.2s
# 6:	learn: 0.5779462	total: 341ms	remaining: 24s
# 7:	learn: 0.5664170	total: 387ms	remaining: 23.8s
# 8:	learn: 0.5583903	total: 425ms	remaining: 23.2s
# 9:	learn: 0.5489458	total: 463ms	remaining: 22.7s
# 10:	learn: 0.5416701	total: 487ms	remaining: 21.7s
# 11:	learn: 0.5355419	total: 511ms	remaining: 20.8s
# 12:	learn: 0.5293598	total: 551ms	remaining: 20.6s
# 13:	learn: 0.5233071	total: 618ms	remaining: 21.5s
# 14:	learn: 0.5184129	total: 649ms	remaining: 21s
# 15:	learn: 0.5143679	total: 674ms	remaining: 20.4s
# 16:	learn: 0.5101971	total: 711ms	remaining: 20.2s
# 17:	learn: 0.5069594	total: 748ms	remaining: 20s
# 18:	learn: 0.5039461	total: 786ms	remaining: 19.9s
# 19:	learn: 0.5013502	total: 846ms	remaining: 20.3s
# 20:	learn: 0.4981109	total: 902ms	remaining: 20.6s
# 21:	learn: 0.4950387	total: 972ms	remaining: 21.1s
# 22:	learn: 0.4934231	total: 1s	remaining: 20.7s
# 23:	learn: 0.4909426	total: 1.06s	remaining: 21s
# 24:	learn: 0.4896139	total: 1.09s	remaining: 20.8s
# 25:	learn: 0.4873707	total: 1.15s	remaining: 20.9s
# 26:	learn: 0.4857583	total: 1.21s	remaining: 21.2s
# 27:	learn: 0.4847539	total: 1.23s	remaining: 20.7s
# 28:	learn: 0.4836008	total: 1.25s	remaining: 20.4s
# 29:	learn: 0.4822775	total: 1.32s	remaining: 20.6s
# 30:	learn: 0.4805596	total: 1.36s	remaining: 20.6s
# 31:	learn: 0.4794334	total: 1.41s	remaining: 20.6s
# 32:	learn: 0.4786133	total: 1.46s	remaining: 20.6s
# 33:	learn: 0.4773654	total: 1.58s	remaining: 21.7s
# 34:	learn: 0.4768246	total: 1.64s	remaining: 21.8s
# 35:	learn: 0.4762262	total: 1.77s	remaining: 22.8s
# 36:	learn: 0.4752004	total: 1.87s	remaining: 23.5s
# 37:	learn: 0.4741019	total: 2.09s	remaining: 25.4s
# 38:	learn: 0.4733597	total: 2.27s	remaining: 26.9s
# 39:	learn: 0.4725246	total: 2.33s	remaining: 26.8s
# 40:	learn: 0.4722076	total: 2.4s	remaining: 26.8s
# 41:	learn: 0.4715922	total: 2.47s	remaining: 27s
# 42:	learn: 0.4712058	total: 2.54s	remaining: 27.1s
# 43:	learn: 0.4708349	total: 2.62s	remaining: 27.1s
# 44:	learn: 0.4705046	total: 2.7s	remaining: 27.3s
# 45:	learn: 0.4702677	total: 2.76s	remaining: 27.2s
# 46:	learn: 0.4701141	total: 2.82s	remaining: 27.2s
# 47:	learn: 0.4694129	total: 2.86s	remaining: 26.9s
# 48:	learn: 0.4692466	total: 2.88s	remaining: 26.5s
# 49:	learn: 0.4688357	total: 2.92s	remaining: 26.3s
# 50:	learn: 0.4685217	total: 2.96s	remaining: 26s
# 51:	learn: 0.4681270	total: 3s	remaining: 25.8s
# 52:	learn: 0.4676336	total: 3.04s	remaining: 25.7s
# 53:	learn: 0.4674844	total: 3.08s	remaining: 25.4s
# 54:	learn: 0.4671877	total: 3.1s	remaining: 25.1s
# 55:	learn: 0.4667934	total: 3.14s	remaining: 24.9s
# 56:	learn: 0.4663524	total: 3.19s	remaining: 24.8s
# 57:	learn: 0.4659184	total: 3.23s	remaining: 24.6s
# 58:	learn: 0.4654867	total: 3.27s	remaining: 24.4s
# 59:	learn: 0.4651864	total: 3.31s	remaining: 24.3s
# 60:	learn: 0.4649931	total: 3.35s	remaining: 24.1s
# 61:	learn: 0.4646397	total: 3.39s	remaining: 23.9s
# 62:	learn: 0.4644121	total: 3.43s	remaining: 23.8s
# 63:	learn: 0.4640113	total: 3.47s	remaining: 23.6s
# 64:	learn: 0.4636779	total: 3.51s	remaining: 23.5s
# 65:	learn: 0.4633881	total: 3.56s	remaining: 23.4s
# 66:	learn: 0.4633018	total: 3.59s	remaining: 23.2s
# 67:	learn: 0.4632571	total: 3.61s	remaining: 22.9s
# 68:	learn: 0.4632135	total: 3.63s	remaining: 22.7s
# 69:	learn: 0.4631264	total: 3.66s	remaining: 22.5s
# 70:	learn: 0.4626900	total: 3.7s	remaining: 22.4s
# 71:	learn: 0.4626696	total: 3.73s	remaining: 22.1s
# 72:	learn: 0.4626572	total: 3.74s	remaining: 21.9s
# 73:	learn: 0.4626020	total: 3.77s	remaining: 21.7s
# 74:	learn: 0.4623977	total: 3.81s	remaining: 21.6s
# 75:	learn: 0.4623525	total: 3.84s	remaining: 21.4s
# 76:	learn: 0.4623359	total: 3.85s	remaining: 21.2s
# 77:	learn: 0.4621017	total: 3.89s	remaining: 21.1s
# 78:	learn: 0.4619184	total: 3.93s	remaining: 21s
# 79:	learn: 0.4617266	total: 3.98s	remaining: 20.9s
# 80:	learn: 0.4616419	total: 4.02s	remaining: 20.8s
# 81:	learn: 0.4613560	total: 4.06s	remaining: 20.7s
# 82:	learn: 0.4611782	total: 4.11s	remaining: 20.7s
# 83:	learn: 0.4609228	total: 4.16s	remaining: 20.6s
# 84:	learn: 0.4608114	total: 4.2s	remaining: 20.5s
# 85:	learn: 0.4605580	total: 4.25s	remaining: 20.5s
# 86:	learn: 0.4603219	total: 4.3s	remaining: 20.4s
# 87:	learn: 0.4602029	total: 4.32s	remaining: 20.2s
# 88:	learn: 0.4601661	total: 4.36s	remaining: 20.1s
# 89:	learn: 0.4600934	total: 4.39s	remaining: 20s
# 90:	learn: 0.4600220	total: 4.42s	remaining: 19.9s
# 91:	learn: 0.4599025	total: 4.47s	remaining: 19.8s
# 92:	learn: 0.4598734	total: 4.52s	remaining: 19.8s
# 93:	learn: 0.4598091	total: 4.57s	remaining: 19.7s
# 94:	learn: 0.4595262	total: 4.61s	remaining: 19.6s
# 95:	learn: 0.4593539	total: 4.65s	remaining: 19.6s
# 96:	learn: 0.4593535	total: 4.66s	remaining: 19.4s
# 97:	learn: 0.4591755	total: 4.69s	remaining: 19.2s
# 98:	learn: 0.4591671	total: 4.7s	remaining: 19s
# 99:	learn: 0.4590002	total: 4.74s	remaining: 19s
# 100:	learn: 0.4588948	total: 4.79s	remaining: 18.9s
# 101:	learn: 0.4587911	total: 4.83s	remaining: 18.8s
# 102:	learn: 0.4586275	total: 4.87s	remaining: 18.8s
# 103:	learn: 0.4585817	total: 4.91s	remaining: 18.7s
# 104:	learn: 0.4583289	total: 4.95s	remaining: 18.6s
# 105:	learn: 0.4581018	total: 4.99s	remaining: 18.5s
# 106:	learn: 0.4580885	total: 5.02s	remaining: 18.4s
# 107:	learn: 0.4580738	total: 5.04s	remaining: 18.3s
# 108:	learn: 0.4578914	total: 5.07s	remaining: 18.2s
# 109:	learn: 0.4576987	total: 5.11s	remaining: 18.1s
# 110:	learn: 0.4576469	total: 5.14s	remaining: 18s
# 111:	learn: 0.4574856	total: 5.18s	remaining: 17.9s
# 112:	learn: 0.4573420	total: 5.23s	remaining: 17.9s
# 113:	learn: 0.4571544	total: 5.27s	remaining: 17.9s
# 114:	learn: 0.4570345	total: 5.32s	remaining: 17.8s
# 115:	learn: 0.4570046	total: 5.35s	remaining: 17.7s
# 116:	learn: 0.4569378	total: 5.38s	remaining: 17.6s
# 117:	learn: 0.4568319	total: 5.41s	remaining: 17.5s
# 118:	learn: 0.4566570	total: 5.46s	remaining: 17.5s
# 119:	learn: 0.4565418	total: 5.5s	remaining: 17.4s
# 120:	learn: 0.4564788	total: 5.55s	remaining: 17.4s
# 121:	learn: 0.4563853	total: 5.59s	remaining: 17.3s
# 122:	learn: 0.4562119	total: 5.63s	remaining: 17.3s
# 123:	learn: 0.4561510	total: 5.64s	remaining: 17.1s
# 124:	learn: 0.4560437	total: 5.68s	remaining: 17.1s
# 125:	learn: 0.4560144	total: 5.72s	remaining: 17s
# 126:	learn: 0.4558330	total: 5.77s	remaining: 16.9s
# 127:	learn: 0.4558178	total: 5.79s	remaining: 16.8s
# 128:	learn: 0.4556886	total: 5.84s	remaining: 16.8s
# 129:	learn: 0.4556529	total: 5.88s	remaining: 16.7s
# 130:	learn: 0.4555702	total: 5.92s	remaining: 16.7s
# 131:	learn: 0.4554673	total: 5.97s	remaining: 16.7s
# 132:	learn: 0.4553341	total: 6.01s	remaining: 16.6s
# 133:	learn: 0.4552907	total: 6.06s	remaining: 16.6s
# 134:	learn: 0.4551203	total: 6.11s	remaining: 16.5s
# 135:	learn: 0.4550309	total: 6.16s	remaining: 16.5s
# 136:	learn: 0.4549446	total: 6.19s	remaining: 16.4s
# 137:	learn: 0.4548461	total: 6.24s	remaining: 16.4s
# 138:	learn: 0.4548232	total: 6.26s	remaining: 16.3s
# 139:	learn: 0.4547802	total: 6.31s	remaining: 16.2s
# 140:	learn: 0.4547212	total: 6.35s	remaining: 16.2s
# 141:	learn: 0.4545622	total: 6.4s	remaining: 16.1s
# 142:	learn: 0.4544181	total: 6.45s	remaining: 16.1s
# 143:	learn: 0.4543510	total: 6.49s	remaining: 16.1s
# 144:	learn: 0.4542557	total: 6.54s	remaining: 16s
# 145:	learn: 0.4540932	total: 6.59s	remaining: 16s
# 146:	learn: 0.4540559	total: 6.67s	remaining: 16s
# 147:	learn: 0.4539434	total: 6.77s	remaining: 16.1s
# 148:	learn: 0.4539279	total: 6.95s	remaining: 16.4s
# 149:	learn: 0.4537699	total: 7.2s	remaining: 16.8s
# 150:	learn: 0.4537695	total: 7.33s	remaining: 16.9s
# 151:	learn: 0.4535974	total: 7.67s	remaining: 17.6s
# 152:	learn: 0.4534760	total: 7.72s	remaining: 17.5s
# 153:	learn: 0.4533554	total: 7.78s	remaining: 17.5s
# 154:	learn: 0.4533033	total: 7.84s	remaining: 17.5s
# 155:	learn: 0.4532501	total: 7.91s	remaining: 17.4s
# 156:	learn: 0.4531308	total: 7.98s	remaining: 17.4s
# 157:	learn: 0.4530956	total: 8.04s	remaining: 17.4s
# 158:	learn: 0.4529975	total: 8.12s	remaining: 17.4s
# 159:	learn: 0.4528770	total: 8.21s	remaining: 17.4s
# 160:	learn: 0.4528193	total: 8.28s	remaining: 17.4s
# 161:	learn: 0.4527608	total: 8.33s	remaining: 17.4s
# 162:	learn: 0.4527113	total: 8.36s	remaining: 17.3s
# 163:	learn: 0.4525894	total: 8.4s	remaining: 17.2s
# 164:	learn: 0.4524744	total: 8.44s	remaining: 17.1s
# 165:	learn: 0.4524604	total: 8.48s	remaining: 17.1s
# 166:	learn: 0.4523750	total: 8.52s	remaining: 17s
# 167:	learn: 0.4522891	total: 8.56s	remaining: 16.9s
# 168:	learn: 0.4521813	total: 8.59s	remaining: 16.8s
# 169:	learn: 0.4521215	total: 8.62s	remaining: 16.7s
# 170:	learn: 0.4520622	total: 8.66s	remaining: 16.7s
# 171:	learn: 0.4520010	total: 8.7s	remaining: 16.6s
# 172:	learn: 0.4518938	total: 8.74s	remaining: 16.5s
# 173:	learn: 0.4516417	total: 8.78s	remaining: 16.5s
# 174:	learn: 0.4515223	total: 8.82s	remaining: 16.4s
# 175:	learn: 0.4514889	total: 8.86s	remaining: 16.3s
# 176:	learn: 0.4513671	total: 8.89s	remaining: 16.2s
# 177:	learn: 0.4512792	total: 8.94s	remaining: 16.2s
# 178:	learn: 0.4511610	total: 8.97s	remaining: 16.1s
# 179:	learn: 0.4510929	total: 9.02s	remaining: 16s
# 180:	learn: 0.4510411	total: 9.06s	remaining: 16s
# 181:	learn: 0.4509564	total: 9.13s	remaining: 15.9s
# 182:	learn: 0.4509481	total: 9.15s	remaining: 15.9s
# 183:	learn: 0.4508640	total: 9.19s	remaining: 15.8s
# 184:	learn: 0.4507851	total: 9.24s	remaining: 15.7s
# 185:	learn: 0.4507850	total: 9.25s	remaining: 15.6s
# 186:	learn: 0.4507333	total: 9.29s	remaining: 15.5s
# 187:	learn: 0.4506671	total: 9.32s	remaining: 15.5s
# 188:	learn: 0.4506018	total: 9.38s	remaining: 15.4s
# 189:	learn: 0.4506003	total: 9.39s	remaining: 15.3s
# 190:	learn: 0.4504999	total: 9.43s	remaining: 15.3s
# 191:	learn: 0.4504520	total: 9.47s	remaining: 15.2s
# 192:	learn: 0.4503832	total: 9.52s	remaining: 15.1s
# 193:	learn: 0.4503112	total: 9.55s	remaining: 15.1s
# 194:	learn: 0.4501976	total: 9.59s	remaining: 15s
# 195:	learn: 0.4501957	total: 9.61s	remaining: 14.9s
# 196:	learn: 0.4501335	total: 9.67s	remaining: 14.9s
# 197:	learn: 0.4500659	total: 9.71s	remaining: 14.8s
# 198:	learn: 0.4499189	total: 9.76s	remaining: 14.8s
# 199:	learn: 0.4498645	total: 9.8s	remaining: 14.7s
# 200:	learn: 0.4497763	total: 9.84s	remaining: 14.6s
# 201:	learn: 0.4497173	total: 9.87s	remaining: 14.6s
# 202:	learn: 0.4495596	total: 9.91s	remaining: 14.5s
# 203:	learn: 0.4494641	total: 9.96s	remaining: 14.4s
# 204:	learn: 0.4494128	total: 10s	remaining: 14.4s
# 205:	learn: 0.4493054	total: 10s	remaining: 14.3s
# 206:	learn: 0.4493051	total: 10.1s	remaining: 14.2s
# 207:	learn: 0.4492511	total: 10.1s	remaining: 14.2s
# 208:	learn: 0.4492457	total: 10.2s	remaining: 14.2s
# 209:	learn: 0.4491780	total: 10.3s	remaining: 14.2s
# 210:	learn: 0.4491717	total: 10.3s	remaining: 14.1s
# 211:	learn: 0.4490825	total: 10.4s	remaining: 14.1s
# 212:	learn: 0.4489928	total: 10.4s	remaining: 14.1s
# 213:	learn: 0.4489483	total: 10.5s	remaining: 14s
# 214:	learn: 0.4489135	total: 10.6s	remaining: 14s
# 215:	learn: 0.4489134	total: 10.6s	remaining: 13.9s
# 216:	learn: 0.4488751	total: 10.7s	remaining: 13.9s
# 217:	learn: 0.4488453	total: 10.7s	remaining: 13.8s
# 218:	learn: 0.4487055	total: 10.7s	remaining: 13.8s
# 219:	learn: 0.4486824	total: 10.8s	remaining: 13.7s
# 220:	learn: 0.4486777	total: 10.8s	remaining: 13.6s
# 221:	learn: 0.4486144	total: 10.8s	remaining: 13.6s
# 222:	learn: 0.4485552	total: 10.9s	remaining: 13.5s
# 223:	learn: 0.4484614	total: 11s	remaining: 13.5s
# 224:	learn: 0.4484572	total: 11s	remaining: 13.4s
# 225:	learn: 0.4484566	total: 11s	remaining: 13.3s
# 226:	learn: 0.4483583	total: 11.1s	remaining: 13.3s
# 227:	learn: 0.4483465	total: 11.1s	remaining: 13.2s
# 228:	learn: 0.4482923	total: 11.2s	remaining: 13.2s
# 229:	learn: 0.4481393	total: 11.2s	remaining: 13.2s
# 230:	learn: 0.4479748	total: 11.3s	remaining: 13.2s
# 231:	learn: 0.4478714	total: 11.4s	remaining: 13.1s
# 232:	learn: 0.4477320	total: 11.4s	remaining: 13.1s
# 233:	learn: 0.4476722	total: 11.5s	remaining: 13.1s
# 234:	learn: 0.4475860	total: 11.6s	remaining: 13s
# 235:	learn: 0.4474393	total: 11.6s	remaining: 13s
# 236:	learn: 0.4473061	total: 11.7s	remaining: 12.9s
# 237:	learn: 0.4471318	total: 11.7s	remaining: 12.9s
# 238:	learn: 0.4471283	total: 11.8s	remaining: 12.8s
# 239:	learn: 0.4469806	total: 11.8s	remaining: 12.8s
# 240:	learn: 0.4468673	total: 11.8s	remaining: 12.7s
# 241:	learn: 0.4467051	total: 11.9s	remaining: 12.7s
# 242:	learn: 0.4466591	total: 11.9s	remaining: 12.6s
# 243:	learn: 0.4465929	total: 11.9s	remaining: 12.5s
# 244:	learn: 0.4465255	total: 12s	remaining: 12.5s
# 245:	learn: 0.4464184	total: 12s	remaining: 12.4s
# 246:	learn: 0.4462569	total: 12.1s	remaining: 12.4s
# 247:	learn: 0.4461973	total: 12.1s	remaining: 12.3s
# 248:	learn: 0.4461154	total: 12.2s	remaining: 12.3s
# 249:	learn: 0.4460549	total: 12.2s	remaining: 12.2s
# 250:	learn: 0.4460057	total: 12.2s	remaining: 12.1s
# 251:	learn: 0.4459109	total: 12.3s	remaining: 12.1s
# 252:	learn: 0.4458514	total: 12.3s	remaining: 12s
# 253:	learn: 0.4457133	total: 12.4s	remaining: 12s
# 254:	learn: 0.4455516	total: 12.4s	remaining: 11.9s
# 255:	learn: 0.4454133	total: 12.5s	remaining: 11.9s
# 256:	learn: 0.4452701	total: 12.5s	remaining: 11.8s
# 257:	learn: 0.4451807	total: 12.5s	remaining: 11.8s
# 258:	learn: 0.4450619	total: 12.6s	remaining: 11.7s
# 259:	learn: 0.4449095	total: 12.6s	remaining: 11.7s
# 260:	learn: 0.4447730	total: 12.7s	remaining: 11.6s
# 261:	learn: 0.4447053	total: 12.7s	remaining: 11.5s
# 262:	learn: 0.4446096	total: 12.7s	remaining: 11.5s
# 263:	learn: 0.4444861	total: 12.8s	remaining: 11.4s
# 264:	learn: 0.4444082	total: 12.8s	remaining: 11.4s
# 265:	learn: 0.4442807	total: 12.9s	remaining: 11.3s
# 266:	learn: 0.4441797	total: 12.9s	remaining: 11.3s
# 267:	learn: 0.4440242	total: 12.9s	remaining: 11.2s
# 268:	learn: 0.4439228	total: 13s	remaining: 11.2s
# 269:	learn: 0.4438435	total: 13s	remaining: 11.1s
# 270:	learn: 0.4436816	total: 13.1s	remaining: 11.1s
# 271:	learn: 0.4435753	total: 13.1s	remaining: 11s
# 272:	learn: 0.4434141	total: 13.2s	remaining: 11s
# 273:	learn: 0.4432676	total: 13.2s	remaining: 10.9s
# 274:	learn: 0.4430637	total: 13.3s	remaining: 10.9s
# 275:	learn: 0.4429866	total: 13.3s	remaining: 10.8s
# 276:	learn: 0.4428566	total: 13.3s	remaining: 10.7s
# 277:	learn: 0.4426769	total: 13.4s	remaining: 10.7s
# 278:	learn: 0.4424986	total: 13.4s	remaining: 10.6s
# 279:	learn: 0.4423959	total: 13.5s	remaining: 10.6s
# 280:	learn: 0.4422610	total: 13.5s	remaining: 10.5s
# 281:	learn: 0.4421740	total: 13.6s	remaining: 10.5s
# 282:	learn: 0.4421138	total: 13.6s	remaining: 10.4s
# 283:	learn: 0.4419796	total: 13.6s	remaining: 10.4s
# 284:	learn: 0.4418603	total: 13.7s	remaining: 10.3s
# 285:	learn: 0.4418526	total: 13.7s	remaining: 10.3s
# 286:	learn: 0.4417347	total: 13.8s	remaining: 10.2s
# 287:	learn: 0.4416447	total: 13.8s	remaining: 10.2s
# 288:	learn: 0.4415172	total: 13.9s	remaining: 10.1s
# 289:	learn: 0.4413881	total: 13.9s	remaining: 10.1s
# 290:	learn: 0.4412363	total: 14s	remaining: 10s
# 291:	learn: 0.4410566	total: 14s	remaining: 10s
# 292:	learn: 0.4409454	total: 14.1s	remaining: 9.95s
# 293:	learn: 0.4408142	total: 14.1s	remaining: 9.9s
# 294:	learn: 0.4407199	total: 14.2s	remaining: 9.85s
# 295:	learn: 0.4405787	total: 14.2s	remaining: 9.8s
# 296:	learn: 0.4403974	total: 14.3s	remaining: 9.75s
# 297:	learn: 0.4402489	total: 14.3s	remaining: 9.71s
# 298:	learn: 0.4400822	total: 14.4s	remaining: 9.65s
# 299:	learn: 0.4399032	total: 14.4s	remaining: 9.6s
# 300:	learn: 0.4398093	total: 14.4s	remaining: 9.55s
# 301:	learn: 0.4397387	total: 14.5s	remaining: 9.5s
# 302:	learn: 0.4396228	total: 14.5s	remaining: 9.45s
# 303:	learn: 0.4394897	total: 14.6s	remaining: 9.4s
# 304:	learn: 0.4393461	total: 14.6s	remaining: 9.35s
# 305:	learn: 0.4392473	total: 14.7s	remaining: 9.3s
# 306:	learn: 0.4390706	total: 14.7s	remaining: 9.25s
# 307:	learn: 0.4389647	total: 14.8s	remaining: 9.2s
# 308:	learn: 0.4388692	total: 14.8s	remaining: 9.15s
# 309:	learn: 0.4387397	total: 14.8s	remaining: 9.1s
# 310:	learn: 0.4386300	total: 14.9s	remaining: 9.04s
# 311:	learn: 0.4385396	total: 14.9s	remaining: 8.99s
# 312:	learn: 0.4384426	total: 15s	remaining: 8.94s
# 313:	learn: 0.4383420	total: 15s	remaining: 8.89s
# 314:	learn: 0.4382257	total: 15.1s	remaining: 8.84s
# 315:	learn: 0.4380439	total: 15.1s	remaining: 8.79s
# 316:	learn: 0.4379160	total: 15.1s	remaining: 8.74s
# 317:	learn: 0.4377948	total: 15.2s	remaining: 8.69s
# 318:	learn: 0.4376581	total: 15.2s	remaining: 8.64s
# 319:	learn: 0.4375342	total: 15.3s	remaining: 8.6s
# 320:	learn: 0.4374608	total: 15.4s	remaining: 8.57s
# 321:	learn: 0.4373410	total: 15.4s	remaining: 8.53s
# 322:	learn: 0.4372036	total: 15.5s	remaining: 8.49s
# 323:	learn: 0.4370840	total: 15.6s	remaining: 8.46s
# 324:	learn: 0.4369919	total: 15.7s	remaining: 8.46s
# 325:	learn: 0.4368809	total: 15.8s	remaining: 8.43s
# 326:	learn: 0.4367985	total: 15.9s	remaining: 8.39s
# 327:	learn: 0.4366778	total: 15.9s	remaining: 8.34s
# 328:	learn: 0.4365917	total: 15.9s	remaining: 8.29s
# 329:	learn: 0.4364884	total: 16s	remaining: 8.24s
# 330:	learn: 0.4364068	total: 16s	remaining: 8.19s
# 331:	learn: 0.4363364	total: 16.1s	remaining: 8.14s
# 332:	learn: 0.4362292	total: 16.1s	remaining: 8.09s
# 333:	learn: 0.4360818	total: 16.2s	remaining: 8.04s
# 334:	learn: 0.4359270	total: 16.2s	remaining: 7.99s
# 335:	learn: 0.4357858	total: 16.3s	remaining: 7.94s
# 336:	learn: 0.4356990	total: 16.3s	remaining: 7.89s
# 337:	learn: 0.4356189	total: 16.4s	remaining: 7.84s
# 338:	learn: 0.4355089	total: 16.4s	remaining: 7.79s
# 339:	learn: 0.4354102	total: 16.5s	remaining: 7.74s
# 340:	learn: 0.4352673	total: 16.5s	remaining: 7.69s
# 341:	learn: 0.4351296	total: 16.5s	remaining: 7.64s
# 342:	learn: 0.4350445	total: 16.6s	remaining: 7.6s
# 343:	learn: 0.4349318	total: 16.7s	remaining: 7.55s
# 344:	learn: 0.4348412	total: 16.7s	remaining: 7.5s
# 345:	learn: 0.4347261	total: 16.8s	remaining: 7.47s
# 346:	learn: 0.4345923	total: 16.8s	remaining: 7.43s
# 347:	learn: 0.4344493	total: 16.9s	remaining: 7.39s
# 348:	learn: 0.4343655	total: 17s	remaining: 7.35s
# 349:	learn: 0.4342953	total: 17s	remaining: 7.31s
# 350:	learn: 0.4342001	total: 17.1s	remaining: 7.27s
# 351:	learn: 0.4340711	total: 17.2s	remaining: 7.23s
# 352:	learn: 0.4339750	total: 17.3s	remaining: 7.19s
# 353:	learn: 0.4338415	total: 17.4s	remaining: 7.16s
# 354:	learn: 0.4336921	total: 17.4s	remaining: 7.13s
# 355:	learn: 0.4335699	total: 17.5s	remaining: 7.08s
# 356:	learn: 0.4334443	total: 17.6s	remaining: 7.04s
# 357:	learn: 0.4333089	total: 17.6s	remaining: 7s
# 358:	learn: 0.4332558	total: 17.7s	remaining: 6.96s
# 359:	learn: 0.4331816	total: 17.8s	remaining: 6.92s
# 360:	learn: 0.4331037	total: 17.8s	remaining: 6.87s
# 361:	learn: 0.4330055	total: 17.9s	remaining: 6.83s
# 362:	learn: 0.4329108	total: 18s	remaining: 6.79s
# 363:	learn: 0.4327857	total: 18.1s	remaining: 6.75s
# 364:	learn: 0.4326531	total: 18.1s	remaining: 6.71s
# 365:	learn: 0.4325299	total: 18.2s	remaining: 6.66s
# 366:	learn: 0.4324145	total: 18.3s	remaining: 6.62s
# 367:	learn: 0.4323106	total: 18.3s	remaining: 6.58s
# 368:	learn: 0.4322259	total: 18.4s	remaining: 6.53s
# 369:	learn: 0.4321243	total: 18.5s	remaining: 6.49s
# 370:	learn: 0.4320202	total: 18.5s	remaining: 6.44s
# 371:	learn: 0.4319506	total: 18.6s	remaining: 6.41s
# 372:	learn: 0.4318489	total: 18.7s	remaining: 6.38s
# 373:	learn: 0.4317446	total: 18.8s	remaining: 6.33s
# 374:	learn: 0.4316526	total: 18.9s	remaining: 6.3s
# 375:	learn: 0.4315353	total: 18.9s	remaining: 6.25s
# 376:	learn: 0.4314525	total: 19s	remaining: 6.2s
# 377:	learn: 0.4313488	total: 19.1s	remaining: 6.16s
# 378:	learn: 0.4312263	total: 19.2s	remaining: 6.11s
# 379:	learn: 0.4311609	total: 19.2s	remaining: 6.07s
# 380:	learn: 0.4310816	total: 19.3s	remaining: 6.02s
# 381:	learn: 0.4310200	total: 19.3s	remaining: 5.97s
# 382:	learn: 0.4309292	total: 19.4s	remaining: 5.92s
# 383:	learn: 0.4308218	total: 19.4s	remaining: 5.87s
# 384:	learn: 0.4307257	total: 19.5s	remaining: 5.83s
# 385:	learn: 0.4306080	total: 19.6s	remaining: 5.78s
# 386:	learn: 0.4305258	total: 19.6s	remaining: 5.72s
# 387:	learn: 0.4304369	total: 19.6s	remaining: 5.67s
# 388:	learn: 0.4303318	total: 19.7s	remaining: 5.62s
# 389:	learn: 0.4302400	total: 19.7s	remaining: 5.57s
# 390:	learn: 0.4301734	total: 19.8s	remaining: 5.51s
# 391:	learn: 0.4301056	total: 19.8s	remaining: 5.46s
# 392:	learn: 0.4299746	total: 19.9s	remaining: 5.41s
# 393:	learn: 0.4298587	total: 19.9s	remaining: 5.35s
# 394:	learn: 0.4297717	total: 19.9s	remaining: 5.3s
# 395:	learn: 0.4296614	total: 20s	remaining: 5.25s
# 396:	learn: 0.4295841	total: 20s	remaining: 5.2s
# 397:	learn: 0.4294629	total: 20.1s	remaining: 5.15s
# 398:	learn: 0.4293469	total: 20.1s	remaining: 5.09s
# 399:	learn: 0.4292734	total: 20.2s	remaining: 5.04s
# 400:	learn: 0.4291882	total: 20.2s	remaining: 4.99s
# 401:	learn: 0.4291113	total: 20.3s	remaining: 4.94s
# 402:	learn: 0.4290401	total: 20.3s	remaining: 4.89s
# 403:	learn: 0.4289438	total: 20.4s	remaining: 4.84s
# 404:	learn: 0.4288585	total: 20.4s	remaining: 4.79s
# 405:	learn: 0.4287873	total: 20.5s	remaining: 4.74s
# 406:	learn: 0.4286559	total: 20.5s	remaining: 4.68s
# 407:	learn: 0.4285897	total: 20.5s	remaining: 4.63s
# 408:	learn: 0.4284992	total: 20.6s	remaining: 4.58s
# 409:	learn: 0.4284232	total: 20.6s	remaining: 4.53s
# 410:	learn: 0.4283314	total: 20.7s	remaining: 4.48s
# 411:	learn: 0.4282507	total: 20.7s	remaining: 4.42s
# 412:	learn: 0.4281772	total: 20.8s	remaining: 4.37s
# 413:	learn: 0.4280777	total: 20.8s	remaining: 4.32s
# 414:	learn: 0.4279995	total: 20.8s	remaining: 4.27s
# 415:	learn: 0.4279143	total: 20.9s	remaining: 4.22s
# 416:	learn: 0.4278361	total: 20.9s	remaining: 4.17s
# 417:	learn: 0.4277587	total: 21s	remaining: 4.12s
# 418:	learn: 0.4276647	total: 21s	remaining: 4.06s
# 419:	learn: 0.4275923	total: 21.1s	remaining: 4.01s
# 420:	learn: 0.4275211	total: 21.1s	remaining: 3.96s
# 421:	learn: 0.4274554	total: 21.2s	remaining: 3.91s
# 422:	learn: 0.4273791	total: 21.2s	remaining: 3.86s
# 423:	learn: 0.4272922	total: 21.2s	remaining: 3.81s
# 424:	learn: 0.4272136	total: 21.3s	remaining: 3.76s
# 425:	learn: 0.4271369	total: 21.3s	remaining: 3.71s
# 426:	learn: 0.4270563	total: 21.4s	remaining: 3.65s
# 427:	learn: 0.4269787	total: 21.4s	remaining: 3.61s
# 428:	learn: 0.4269122	total: 21.5s	remaining: 3.56s
# 429:	learn: 0.4268271	total: 21.6s	remaining: 3.51s
# 430:	learn: 0.4267591	total: 21.6s	remaining: 3.46s
# 431:	learn: 0.4266752	total: 21.7s	remaining: 3.41s
# 432:	learn: 0.4266134	total: 21.7s	remaining: 3.36s
# 433:	learn: 0.4265345	total: 21.8s	remaining: 3.31s
# 434:	learn: 0.4264707	total: 21.8s	remaining: 3.26s
# 435:	learn: 0.4263841	total: 21.8s	remaining: 3.21s
# 436:	learn: 0.4262949	total: 21.9s	remaining: 3.15s
# 437:	learn: 0.4261842	total: 21.9s	remaining: 3.1s
# 438:	learn: 0.4260848	total: 22s	remaining: 3.05s
# 439:	learn: 0.4260095	total: 22s	remaining: 3s
# 440:	learn: 0.4259318	total: 22.1s	remaining: 2.95s
# 441:	learn: 0.4258129	total: 22.1s	remaining: 2.9s
# 442:	learn: 0.4257084	total: 22.1s	remaining: 2.85s
# 443:	learn: 0.4256176	total: 22.2s	remaining: 2.8s
# 444:	learn: 0.4255426	total: 22.2s	remaining: 2.75s
# 445:	learn: 0.4254731	total: 22.3s	remaining: 2.69s
# 446:	learn: 0.4254059	total: 22.3s	remaining: 2.64s
# 447:	learn: 0.4253186	total: 22.3s	remaining: 2.59s
# 448:	learn: 0.4252567	total: 22.4s	remaining: 2.54s
# 449:	learn: 0.4251848	total: 22.4s	remaining: 2.49s
# 450:	learn: 0.4251424	total: 22.5s	remaining: 2.44s
# 451:	learn: 0.4250870	total: 22.6s	remaining: 2.39s
# 452:	learn: 0.4250103	total: 22.6s	remaining: 2.34s
# 453:	learn: 0.4249243	total: 22.6s	remaining: 2.29s
# 454:	learn: 0.4248773	total: 22.7s	remaining: 2.24s
# 455:	learn: 0.4248077	total: 22.8s	remaining: 2.19s
# 456:	learn: 0.4247507	total: 22.8s	remaining: 2.15s
# 457:	learn: 0.4246352	total: 22.9s	remaining: 2.1s
# 458:	learn: 0.4245504	total: 23s	remaining: 2.05s
# 459:	learn: 0.4244563	total: 23s	remaining: 2s
# 460:	learn: 0.4243404	total: 23s	remaining: 1.95s
# 461:	learn: 0.4242378	total: 23.1s	remaining: 1.9s
# 462:	learn: 0.4241131	total: 23.2s	remaining: 1.85s
# 463:	learn: 0.4240266	total: 23.2s	remaining: 1.8s
# 464:	learn: 0.4239396	total: 23.3s	remaining: 1.75s
# 465:	learn: 0.4238604	total: 23.4s	remaining: 1.71s
# 466:	learn: 0.4237549	total: 23.5s	remaining: 1.66s
# 467:	learn: 0.4237053	total: 23.7s	remaining: 1.62s
# 468:	learn: 0.4236214	total: 23.9s	remaining: 1.58s
# 469:	learn: 0.4235502	total: 24s	remaining: 1.53s
# 470:	learn: 0.4234639	total: 24s	remaining: 1.48s
# 471:	learn: 0.4233811	total: 24.1s	remaining: 1.43s
# 472:	learn: 0.4233066	total: 24.1s	remaining: 1.38s
# 473:	learn: 0.4232404	total: 24.2s	remaining: 1.32s
# 474:	learn: 0.4231811	total: 24.2s	remaining: 1.27s
# 475:	learn: 0.4231216	total: 24.2s	remaining: 1.22s
# 476:	learn: 0.4230505	total: 24.3s	remaining: 1.17s
# 477:	learn: 0.4229820	total: 24.3s	remaining: 1.12s
# 478:	learn: 0.4229109	total: 24.3s	remaining: 1.07s
# 479:	learn: 0.4228214	total: 24.4s	remaining: 1.02s
# 480:	learn: 0.4227527	total: 24.4s	remaining: 965ms
# 481:	learn: 0.4226748	total: 24.5s	remaining: 914ms
# 482:	learn: 0.4226219	total: 24.5s	remaining: 862ms
# 483:	learn: 0.4225255	total: 24.5s	remaining: 811ms
# 484:	learn: 0.4223916	total: 24.6s	remaining: 760ms
# 485:	learn: 0.4223108	total: 24.6s	remaining: 709ms
# 486:	learn: 0.4222276	total: 24.6s	remaining: 658ms
# 487:	learn: 0.4221676	total: 24.7s	remaining: 607ms
# 488:	learn: 0.4221053	total: 24.7s	remaining: 556ms
# 489:	learn: 0.4220542	total: 24.8s	remaining: 505ms
# 490:	learn: 0.4219638	total: 24.8s	remaining: 455ms
# 491:	learn: 0.4219278	total: 24.9s	remaining: 404ms
# 492:	learn: 0.4218548	total: 24.9s	remaining: 353ms
# 493:	learn: 0.4217748	total: 24.9s	remaining: 303ms
# 494:	learn: 0.4217124	total: 25s	remaining: 252ms
# 495:	learn: 0.4216507	total: 25s	remaining: 202ms
# 496:	learn: 0.4215442	total: 25.1s	remaining: 152ms
# 497:	learn: 0.4214415	total: 25.2s	remaining: 101ms
# 498:	learn: 0.4213747	total: 25.2s	remaining: 50.5ms
# 499:	learn: 0.4212987	total: 25.3s	remaining: 0us
# 최적의 파라미터 :  {'subsample': 1.0, 'random_strength': 5, 'n_estimators': 500, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'depth': 9, 'colsample_bylevel': 0.5, 'border_count': 50, 'bagging_temperature': 0.1}
# 최적의 매개변수 :  <catboost.core.CatBoostClassifier object at 0x7f6ec393d040>
# best_score :  0.7997081914716901
# model_score :  0.8002352302542296
# 걸린 시간 :  47229.55378460884 초


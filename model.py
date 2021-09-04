# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# matplotlib 의 기본 scheme 말고 
# seaborn scheme 을 세팅하여, 일일이 graph의 font size 지정할 필요 없이 
# seaborn 의 font_scale 을 사용하는 것을 추천드립니다.
plt.style.use('seaborn')
sns.set(font_scale=2.5)

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# null dataset 확인
msno.matrix(df=df_train)

# 직업유형, 휴대폰 소지여부 삭제
df_train.drop('FLAG_MOBIL', axis=1, inplace=True)
df_train.drop('occyp_type', axis=1, inplace=True)

df_test.drop('FLAG_MOBIL', axis=1, inplace=True)
df_test.drop('occyp_type', axis=1, inplace=True)

# 범주형 데이터 정규화
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
categorical_column_train = ['gender', 'car', 'reality', 'income_type', 'edu_type', 'family_type', 'house_type', 'credit']
for i in categorical_column_train:
    df_train[i] = encoder.fit_transform(df_train[i])

categorical_column_test = ['gender', 'car', 'reality', 'income_type', 'edu_type', 'family_type', 'house_type']
for i in categorical_column_test:
    df_test[i] = encoder.fit_transform(df_test[i])




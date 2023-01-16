# Importing warnings filter libraries
import warnings
warnings.filterwarnings('ignore')

#Importing data manipulation and analysis lybraries
import numpy as np
import pandas as pd

#Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE,SelectFromModel

from sklearn.cluster import KMeans


from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve

df = pd.read_excel('Rocket Loans.xlsx').drop('Loan_ID',axis=1)
print(df.head().to_string())
print( )
print(df.info())
print( )
print(df.describe().to_string())
print( )

b_plot = df.boxplot(vert = False)
plt.show()

corr = df.corr()
f, ax = plt.subplots(figsize = (10,10))
c_plot = sns.heatmap(corr,annot=True)
plt.show()

def grab_col_names(dataframe, car_th=10, cat_th=20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > cat_th and dataframe[col].dtype == 'O']

    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')

    return cat_col, num_col, cat_but_car
cat_col, num_col, cat_but_car = grab_col_names(df)

for i in num_col:
    df[i].fillna(df[i].mean(),inplace = True)

for i in cat_col:
    df[i].fillna(df[i].mode()[0],inplace = True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Married'] = le.fit_transform(df['Married'])
df['Qualification'] = le.fit_transform(df['Qualification'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Location_type'] = le.fit_transform(df['Location_type'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])
df['No. of People in the Family'].replace('3+',3,inplace=True)

print( )
print(df.info())
print( )

def replace_outlier (mydf, col, method = 'Quartile',strategy = 'median' ):
    if method == 'Quartile':
        Q1 = mydf[col].quantile (0.25)
        Q2 =  mydf[col].quantile (0.50)
        Q3 = mydf[col].quantile (0.75)
        IQR =Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR
    elif method == 'standard deviation':
        mean = mydf[col].mean()
        std = mydf[col].std()
        LW = mean - (2*std)
        UW = mean + (2*std)
    else:
        print('Pass a correct method')
#Printing all the outliers
    outliers = mydf.loc[(mydf[col]<LW) | (mydf[col]>UW),col]
    outliers_density = round(len(outliers)/ len(mydf),2)
    if len(outliers)==0:
        print(f'feature {col} does not have any outliers')
    else:
        print(f'feature {col} has outliers')
        print(f'Total number of outliers in this {col} is:', (len(outliers)))
        print(f'Outliers percentage in {col} is {outliers_density*100}%')
    if strategy=='median':
    #mydf.loc[ (mydf[col] < LW), col] = Q2 # used for first method
    #mydf.loc[ (mydf[col] > UW), col] = Q2 # used for first method
        mydf.loc[(mydf [col] < LW), col] = Q1 # second method.. the data may get currupted. so we are res
        mydf.loc[(mydf [col] > UW), col] = Q3 #second method.. as the outliers are more and not treated
    elif strategy == 'mean':
        mydf.loc[(mydf [col] < LW), col] = mean
        mydf.loc[(mydf [col] > UW), col] = mean
    else:
        print('Pass the correct strategy')
    return mydf

def odt_plots (mydf, col):
    f, (ax1, ax2) = plt.subplots (1,2,figsize=(25, 8))
    # descriptive statistic boxplot
    sns.boxplot (mydf[col], ax = ax1)
    ax1.set_title (col + ' boxplot')
    ax1.set_xlabel('values')
    ax1.set_ylabel('boxplot')
    #replacing the outliers
    mydf_out = replace_outlier (mydf, col)
    #plotting boxplot without outliers
    sns.boxplot (mydf_out[col], ax = ax2)
    ax2.set_title (col + 'boxplot')
    ax2.set_xlabel('values')
    ax2.set_ylabel('boxplot')
    plt.show()

for col in df.drop(cat_col,axis = 1).columns:
    odt_plots(df,col)

def VIF(independent_variables):
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor (independent_variables.values,i) for i in range (independent_variables.shape[1])]
    vif['independent_variables']= independent_variables.columns
    vif = vif.sort_values(by=['vif'],ascending=False)      #to sort the values in descending order
    return vif

print(VIF(df.drop('Loan_Status',axis=1)))

def CWT (data, tcol):
    independent_variables = data.drop(tcol, axis=1).columns
    corr_result = []
    for col in independent_variables :
        corr_result.append(data[tcol].corr(data[col]))
    result = pd.DataFrame([independent_variables, corr_result], index=['independent variables', 'correlation']).T    #T is for transpose
    return result.sort_values(by = 'correlation',ascending = False)

print(CWT(df,'Loan_Status'))

def PCA_1(x):
    n_comp = len(x.columns)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Applying PCA
    for i in range(1, n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(x)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if evr[i - 1] > 0.9:
            n_components = i
            break
    print('Ecxplained varience ratio after pca is: ', evr)
    # creating a pcs dataframe
    col = []
    for j in range(1, n_components + 1):
        col.append('PC_' + str(j))
    pca_df = pd.DataFrame(p_comp, columns=col)
    return pca_df
transformed_df = PCA_1(df.drop('Loan_Status',axis = 1))
transformed_df = transformed_df.join(df['Loan_Status'],how = 'left')
print(transformed_df.head().to_string())

def train_and_test_split(data,t_col, testsize=0.3):
    x = data.drop(t_col, axis=1)
    y = data[t_col]
    return train_test_split(x,y,test_size=testsize, random_state=1)

def model_builder(model_name, estimator, data, t_col):
    x_train,x_test,y_train,y_test = train_and_test_split(data, t_col)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    return [model_name, accuracy]

def multiple_models(data,t_col):
    col_names = ['model_name', 'accuracy_score' ]
    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = model_builder('LogisticRegression',LogisticRegression(),data,t_col)
    result.loc[len(result)] = model_builder('DecisionTreeClassifier',DecisionTreeClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('KneighborClassifier',KNeighborsClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('RandomForestClassifier',RandomForestClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('SVC',SVC(),data,t_col)
    result.loc[len(result)] = model_builder('AdaBoostClassifier',AdaBoostClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('GradientBoostingClassifier',GradientBoostingClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('XGBClassifier',XGBClassifier(),data,t_col)
    return result.sort_values(by='accuracy_score',ascending=False)


def kfoldCV(x, y, fold=10):
    score_lr = cross_val_score(LogisticRegression(), x, y, cv=fold)
    score_dt = cross_val_score(DecisionTreeClassifier(), x, y, cv=fold)
    score_kn = cross_val_score(KNeighborsClassifier(), x, y, cv=fold)
    score_rf = cross_val_score(RandomForestClassifier(), x, y, cv=fold)
    score_svc = cross_val_score(SVC(), x, y, cv=fold)
    score_ab = cross_val_score(AdaBoostClassifier(), x, y, cv=fold)
    score_gb = cross_val_score(GradientBoostingClassifier(), x, y, cv=fold)
    score_xb = cross_val_score(XGBClassifier(), x, y, cv=fold)

    model_names = ['Logisticregression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'RandomForestClassifier',
                   'SVC', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']
    scores = [score_lr, score_dt, score_kn, score_rf, score_svc, score_ab, score_gb, score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names, score_mean, score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result, columns=['model_names', 'cv_score', 'cv_std'])
    return kfold_df.sort_values(by='cv_score', ascending=False)

def tuning(x,y,fold = 10):
   #parameters grids for different models
    param_dtc = {'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt']}
    param_knn = {'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}
    param_svc = {'gamma':['scale','auto'],'C': [0.1,1,1.5,2]}
    param_rf = {'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt'],'n_estimators':[50,100,150,200]}
    param_ab = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_gb = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_xb = {'eta':[0.1,0.5,10.7,1,5,10,20],'max_depth':[3,5,7,9,10],'gamma':[0,10,20,50],'reg_lambda':[0,1,3,5,7,10],'alpha':[0,1,3,5,7,10]}
    #Creating Model object
    tune_dtc = GridSearchCV(DecisionTreeClassifier(),param_dtc,cv=fold)
    tune_knn = GridSearchCV(KNeighborsClassifier(),param_knn,cv=fold)
    tune_svc = GridSearchCV(SVC(),param_svc,cv=fold)
    tune_rf = GridSearchCV(RandomForestClassifier(),param_rf,cv=fold)
    tune_ab = GridSearchCV(AdaBoostClassifier(),param_ab,cv=fold)
    tune_gb = GridSearchCV(GradientBoostingClassifier(),param_gb,cv=fold)
    tune_xb = GridSearchCV(XGBClassifier(),param_xb,cv=fold)
    #Model fitting
    tune_dtc.fit(x,y)
    tune_knn.fit(x,y)
    tune_svc.fit(x,y)
    tune_rf.fit(x,y)
    tune_ab.fit(x,y)
    tune_gb.fit(x,y)
    tune_xb.fit(x,y)

    tune = [tune_rf,tune_xb,tune_gb,tune_knn,tune_svc,tune_dtc,tune_ab]
    models = ['RF','XB','GB','KNN','SVC','DTR','AB']
    for i in range(len(tune)):
        print('model:',models[i])
        print('Best_params:',tune[i].best_params_)
tuning(transformed_df.drop('Loan_Status',axis=1),transformed_df['Loan_Status'])

def cv_post_hpt(x,y,fold = 10):
    score_lr = cross_val_score(LogisticRegression(),x,y,cv= fold)
    score_dt = cross_val_score(DecisionTreeClassifier(criterion= 'gini' ,max_depth= 3 ,max_features= 6),x,y,cv= fold)
    score_kn = cross_val_score(KNeighborsClassifier(weights ='uniform' ,algorithm ='auto' ),x,y,cv=fold)
    score_rf = cross_val_score(RandomForestClassifier(max_depth= 3,max_features=2 ,n_estimators= 150),x,y,cv=fold)
    score_svc = cross_val_score(SVC(gamma='scale' ,C= 0.1 ),x,y,cv=fold)
    score_ab = cross_val_score(AdaBoostClassifier(n_estimators= 150 ,learning_rate=0.1 ),x,y,cv=fold)
    score_gb = cross_val_score(GradientBoostingClassifier(n_estimators=50 ,learning_rate= 0.1),x,y,cv=fold)
    score_xb = cross_val_score(XGBClassifier(eta=3 ,max_depth= 3,gamma=0 ,reg_lambda = 10,alpha=10 ),x,y,cv=fold)

    model_names = ['LogisticRegression','RandomForestClassifier','DecisionTreeClassifier','KNeighborsClassifier','SVC','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier']
    scores = [score_lr,score_rf, score_dt,score_kn,score_svc,score_ab,score_gb,score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names,score_mean,score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result,columns=['model_names','cv_score','cv_std'])
    return kfold_df.sort_values(by='cv_score',ascending=False)
print('Set 1')
print( )
print(multiple_models(transformed_df,'Loan_Status'))
print( )
print(kfoldCV(transformed_df.drop('Loan_Status',axis = 1),transformed_df['Loan_Status']))
print( )
print(cv_post_hpt(transformed_df.drop('Loan_Status',axis=1),transformed_df['Loan_Status']))
print( )

labels = KMeans(n_clusters=2, random_state=2)
clusters = labels.fit_predict(df.drop('Loan_Status', axis=1))
sc_plot = sns.scatterplot(x=df['Loan_Bearer_Income'], y=df['Loan_Status'], hue=clusters)
plt.show()

def clustering(x,tcol,clusters):
    column = list(set(list(x.columns)) - set(list('Loan_Status')))
    #column = list(x.column)
    r = int(len(column)/2)
    if len(column)%2 == 0:
        r=r
    else:
        r += 1      #same as r+1
    f,ax = plt.subplots(r,2,figsize = (15,15))
    a = 0
    for row in range(r):
        for col in range(2):
            if a!= len(column):
                ax[row][col].scatter(x[tcol] , x[column[a]], c = clusters)
                ax[row][col].set_xlabel(tcol)
                ax[row][col].set_ylabel(column[a])
                a += 1
x = df.drop('Loan_Status',axis = 1)
for col in x.columns:
    clustering(x , col , clusters)
plt.show()

new_df = df.join(pd.DataFrame(clusters,columns=['cluster']),how = 'left')
new_f = new_df.groupby('cluster')['Loan_Bearer_Income'].agg(['mean','median'])

cluster_df = new_df.merge(new_f, on = 'cluster',how= 'left')
print('Set 2')
print( )
print(multiple_models(cluster_df,'Loan_Status'))
print( )
print(kfoldCV(cluster_df.drop('Loan_Status',axis = 1),cluster_df['Loan_Status']))
print( )
print(cv_post_hpt(cluster_df.drop('Loan_Status',axis=1),cluster_df['Loan_Status']))
print( )

x = cluster_df.drop('Loan_Status',axis=1)
y = cluster_df['Loan_Status']


model = RandomForestClassifier()
model.fit(x, y)

selector = SelectFromModel(model, threshold='median')

selector.fit(x, y)

selected_indices = selector.get_support()

selected_features = x.iloc[:, selected_indices]

f_df = selected_features
print(f_df.head().to_string())
print(cluster_df.columns)
to_drop = ['Sex', 'Qualification','Self_Employed','cluster', 'mean', 'median']

final_df = cluster_df.drop(to_drop,axis=1)
print(final_df.head().to_string())
print('Set 3')
print( )
print(multiple_models(final_df,'Loan_Status'))
print( )
print(kfoldCV(final_df.drop('Loan_Status',axis = 1),final_df['Loan_Status']))
print( )
print(cv_post_hpt(final_df.drop('Loan_Status',axis=1),final_df['Loan_Status']))
print( )

new__df = cluster_df

rfe = RFE(estimator = LogisticRegression())

rfe.fit(new__df.drop('Loan_Status',axis=1),new__df['Loan_Status'])

print(rfe.support_)

print(new__df.columns)

c_to_drop = ['Sex','Married', 'No. of People in the Family','Loan_Bearer_Income','Amount Disbursed','cluster', 'mean', 'median']

fin_df = cluster_df.drop(c_to_drop,axis=1)
print(fin_df.head().to_string())
print('Set 4')
print( )
print(multiple_models(fin_df,'Loan_Status'))
print( )
print(kfoldCV(fin_df.drop('Loan_Status',axis = 1),fin_df['Loan_Status']))
print( )
print(cv_post_hpt(fin_df.drop('Loan_Status',axis=1),fin_df['Loan_Status']))
print( )

def generate_learning_curve(model_name,estimater,x,y):
    train_size,train_score,test_score = learning_curve(estimater,x,y,cv= 10)
#     print('train_size',train_size)
#     print('train_score',train_score)
#     print('test_score',test_score)
    train_score_mean = np.mean(train_score,axis=1)
    test_score_mean = np.mean(test_score,axis=1)
    plt.plot(train_size,train_score_mean, c = 'blue')
    plt.plot(train_size,test_score_mean, c = 'red')
    plt.xlabel('Samples')
    plt.ylabel('Scores')
    plt.title('Learning curve for '+model_name)
    plt.legend(('Training accuray','Testing accuracy'))
model_names = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), SVC(),AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for i, model in enumerate(model_names):
#     print(i)
#     print(model_names[i])
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(5,2,i+1)
    generate_learning_curve(type(model).__name__,model,fin_df.drop('Loan_Status',axis=1),fin_df['Loan_Status'])
plt.show(())



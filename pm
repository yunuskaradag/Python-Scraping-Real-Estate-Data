
import pandas as pd
import catboost as cat
from sklearn.preprocessing import LabelEncoder
import csv
import os
import pickle
from sklearn.metrics import roc_auc_score
import pyodbc


####READ AND PICKLE DATASETS####

def sqlreader(query,datasetname,sqldriver,server,path,toPickle=True):
    print( "\nReading data from database...")
    cnxn = pyodbc.connect('DRIVER={'+sqldriver+'};SERVER='+server+';Trusted_Connection=yes')
    print ("Executing query: ", query)
    queryresult = pd.read_sql(query, cnxn)
    print ("Query succesful. Dataframe created.")
    if toPickle == True:
        print ("Pickling dataset")
        queryresult.to_pickle(os.path.join(path,datasetname))
    return queryresult

# =============================================================================
# 
# 
# print ("Reading dataset for 2018-01-31")
# 
# Df_0131 = sqlreader(query = "select * from DM_DS.model.HV_ProtocolPredictionModel_RealEstate_Trainset_20180131 where LiabilityUnresolvedLoanF = 1"
#                     ,datasetname = "Df_0131"
#                     ,sqldriver = "ODBC Driver 13 for SQL Server" 
#                     ,server = "development-01"
#                     ,path = "D:\\Machine_Learning_Models\\Real_Estate_Protocol_Prediction_Model\\Datasets"
#                     ,toPickle=True)
# 
# print ("Reading dataset for 2018-04-30")
# 
# 
# Df_0430 = sqlreader(query = "select * from DM_DS.model.HV_ProtocolPredictionModel_RealEstate_Trainset_20180430 where LiabilityUnresolvedLoanF = 1"
#                     ,datasetname = "Df_0430"
#                     ,sqldriver = "ODBC Driver 13 for SQL Server" 
#                     ,server = "development-01"
#                     ,path = "D:\\Machine_Learning_Models\\Real_Estate_Protocol_Prediction_Model\\Datasets"
#                     ,toPickle=True)
# 
# print ("Reading dataset for 2018-07-31")
# 
# 
# Df_0731 = sqlreader(query = "select * from DM_DS.model.HV_ProtocolPredictionModel_RealEstate_Trainset_20180731 where LiabilityUnresolvedLoanF = 1"
#                     ,datasetname = "Df_0731"
#                     ,sqldriver = "ODBC Driver 13 for SQL Server" 
#                     ,server = "development-01"
#                     ,path = "D:\\Machine_Learning_Models\\Real_Estate_Protocol_Prediction_Model\\Datasets"
#                     ,toPickle=True)
# 
# 
# print ("Reading dataset for 2018-10-31")
# 
# 
# Df_1031 = sqlreader(query = "select * from DM_DS.model.HV_ProtocolPredictionModel_RealEstate_Trainset_20181031 where LiabilityUnresolvedLoanF = 1"
#                     ,datasetname = "Df_1031"
#                     ,sqldriver = "ODBC Driver 13 for SQL Server" 
#                     ,server = "development-01"
#                     ,path = "D:\\Machine_Learning_Models\\Real_Estate_Protocol_Prediction_Model\\Datasets"
#                     ,toPickle=True)
# 
# 
# print ("Reading dataset for 2018-12-31")
# 
# 
# Df_1231 = sqlreader(query = "select * from DM_DS.model.HV_ProtocolPredictionModel_RealEstate_Trainset_20181231 where LiabilityUnresolvedLoanF = 1"
#                     ,datasetname = "Df_1231"
#                     ,sqldriver = "ODBC Driver 13 for SQL Server" 
#                     ,server = "development-01"
#                     ,path = "D:\\Machine_Learning_Models\\Real_Estate_Protocol_Prediction_Model\\Datasets"
#                     ,toPickle=True)
# =============================================================================


Df_0131 =pd.read_pickle('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/datasets/Df_0131')
Df_0430 =pd.read_pickle('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/datasets/Df_0430')
Df_0731 =pd.read_pickle('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/datasets/Df_0731')
Df_1031 =pd.read_pickle('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/datasets/Df_1031')
Df_1231 =pd.read_pickle('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/datasets/Df_1231')

Df_0131.AsOfDate.value_counts()
Df_0430.AsOfDate.value_counts()
Df_0731.AsOfDate.value_counts()
Df_1031.AsOfDate.value_counts()
Df_1231.AsOfDate.value_counts()

####PREPROCESS DATASETS####

#For removing nonunique columns
def removenonuniquecol(dataset):
    dropcols = [col for col in dataset.columns if dataset[col].nunique(dropna=True)==1]
    print ('Removing columns: ',dropcols)
    dataset.drop(dropcols,axis= 1,inplace= True,errors= 'ignore')
    return dropcols

#For converting csv file to dictionary
def csvloader(path):
    csv_file_object = csv.reader(open(path, 'r' )) #r is for read b is for binary
    listname = [] 
    for row in csv_file_object:
        row = row[0]
        listname.append(row) 
    return listname

#For label encoding
def labelencoder(dataset,featurelist):
    objectlist = dataset.select_dtypes(include=['object']).copy()
    cat_col = [col for col in dataset.columns if col in objectlist and col in featurelist]
    for col in cat_col:
        print("Encoding ",col)
        lbl = LabelEncoder()
        dataset[col].fillna(-999)
        lbl.fit(list(dataset[col].values.astype('str')))
        dataset[col] = lbl.transform(list(dataset[col].values.astype('str')))
    return cat_col

def TrainPrep(DataPath,PredictorPath,Datasetname,Labelname,Id):
    picklePath = os.path.join(DataPath,Datasetname)
    Df = pd.read_pickle(picklePath)
    removedCols = removenonuniquecol(Df)    
    Predictors = csvloader(PredictorPath)
    Predictors = [col for col in Predictors if col not in removedCols]
    Predictors = [col for col in Predictors if col not in ["ContactCode",'TrainsetContactF']]
    DfLabel = Df[Labelname]
    encodedList = labelencoder(Df, Predictors)
    return Predictors,Df, DfLabel, removedCols, encodedList

def TestPrep(DataPath,PredictorPath,Datasetname, dropcols, encodedList,Labelname):
    picklePath = os.path.join(DataPath,Datasetname)
    Df = pd.read_pickle(picklePath)
    Df.drop(dropcols,axis= 1,inplace= True,errors= 'ignore')
    Predictors = csvloader(PredictorPath)
    Predictors = [col for col in Predictors if col not in dropcols]
    Predictors = [col for col in Predictors if col not in ['ContactCode','TrainsetContactF']]
    DfLabel = Df[Labelname]
    for col in encodedList:
        print("Encoding ",col)
        lbl = LabelEncoder()
        Df[col].fillna(-999)
        lbl.fit(list(Df[col].values.astype('str')))
        Df[col] = lbl.transform(list(Df[col].values.astype('str')))
    return Df, DfLabel

print ("Loading and preparing datasets...")

FeaturePath = '//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/Predictors/Predictors.csv'
DataPath = '//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/RE_Pro_ml/datasets'

Predictors,Df_0131, Label_0131, removedCols, encodedList = TrainPrep(DataPath,FeaturePath,'Df_0131',"LiveProtocolF_3Month","ContactCode")

Df_0430, Label_0430 = TestPrep(DataPath,FeaturePath,"Df_0430",removedCols, encodedList,"LiveProtocolF_3Month")

Df_0731, Label_0731 = TestPrep(DataPath,FeaturePath,"Df_0731",removedCols, encodedList,"LiveProtocolF_3Month")

Df_1031, Label_1031 = TestPrep(DataPath,FeaturePath,"Df_1031",removedCols, encodedList,"LiveProtocolF_3Month")

Df_1231, Label_1231 = TestPrep(DataPath,FeaturePath,"Df_1231",removedCols, encodedList,"LiveProtocolF_3Month")


####TRAINING CLASSIFIERS####


def catboosttrainer(X,y,features,initparam,modelname,modelpath,docpath,cvfold = 5):
    print ("searching for optimal iteration count...")
    trainpool = cat.Pool(X[features],y)
    cvresult = cat.cv(params= initparam, fold_count=cvfold, pool=trainpool,stratified = True)
    initparam['iterations'] = (len(cvresult)) - (initparam['od_wait']+1)   
    del initparam['od_wait'] 
    del initparam['od_type']
    #Update sonrası kontrol et
    print ("optimal iteration count is ", initparam['iterations'])
    print ("fitting model...")
    clf = cat.CatBoostClassifier(** initparam)
    clf.fit(trainpool)
    imp = clf.get_feature_importance(trainpool,fstr_type='FeatureImportance')
    dfimp = pd.DataFrame(imp,columns = ['CatBoostImportance'])
    dfimp.insert(0,column='Feature', value=features) 
    dfimp = dfimp.sort_values(['CatBoostImportance','Feature'], ascending= False)
    xlsxpath = os.path.join(docpath,modelname+".xlsx")
    dfimp.to_excel(xlsxpath)
    print ("pickling model...")
    picklepath = os.path.join(modelpath,modelname)
    with open(picklepath,'wb') as fout:
        pickle.dump(clf, fout)
    return cvresult,clf,initparam,dfimp



print ("Setting paths...")

modelpath = '//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/models'
docpath = '//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/features'


print ("Training with 0131 dataset...")

CatBoostParam = { 'iterations': 10000, 'od_type': 'Iter', 'od_wait': 100,'loss_function': 'Logloss','eval_metric': 'AUC' }

cvresult_0131,clf_0131,initparam,dfimp_0131 = catboosttrainer(Df_0131,Label_0131,Predictors,CatBoostParam,'CB_Model_0131',modelpath,docpath,cvfold = 5)


print ("Training with 0430 dataset...")

CatBoostParam = { 'iterations': 10000, 'od_type': 'Iter', 'od_wait': 100,'loss_function': 'Logloss','eval_metric': 'AUC' }

cvresult_0430,clf_0430,initparam,dfimp_0430 = catboosttrainer(Df_0430,Label_0430,Predictors,CatBoostParam,'CB_Model_0430',modelpath,docpath,cvfold = 5)


print ("Training with 0731 dataset...")

CatBoostParam = { 'iterations': 10000, 'od_type': 'Iter', 'od_wait': 100,'loss_function': 'Logloss','eval_metric': 'AUC' }

cvresult_0731,clf_0731,initparam,dfimp_0731 = catboosttrainer(Df_0731,Label_0731,Predictors,CatBoostParam,'CB_Model_0731',modelpath,docpath,cvfold = 5)

print ("Predicting 1031 dataset")

proba_0131 = clf_0131.predict_proba(Df_1031[Predictors])[:,1]
proba_0430 = clf_0430.predict_proba(Df_1031[Predictors])[:,1]
proba_0731 = clf_0731.predict_proba(Df_1031[Predictors])[:,1]

Df_1031.insert(len(Df_1031.columns),"Proba_0131",proba_0131)
Df_1031.insert(len(Df_1031.columns),"Proba_0430",proba_0430)
Df_1031.insert(len(Df_1031.columns),"Proba_0731",proba_0731)

for proba in ["Proba_0131","Proba_0430","Proba_0731"]:
    print ("auc for ",proba," is: ",roc_auc_score(Label_1031,Df_1031[proba]))


print ("Searching optimal weights...")

proba_0131_v2 = clf_0131.predict_proba(Df_1231[Predictors])[:,1]
proba_0430_v2 = clf_0430.predict_proba(Df_1231[Predictors])[:,1]
proba_0731_v2 = clf_0731.predict_proba(Df_1231[Predictors])[:,1]

Df_1231.insert(len(Df_1231.columns),"Proba_0131",proba_0131_v2)
Df_1231.insert(len(Df_1231.columns),"Proba_0430",proba_0430_v2)
Df_1231.insert(len(Df_1231.columns),"Proba_0731",proba_0731_v2)



space = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


optdict = dict()
iteration = 0
for w1 in space:
    iteration +=1
    print (iteration)
    for w2 in space:
        w3 = 1-w1-w2
        if w3 < 0:
            pass
        else:
           weightdict = dict() 
           auc = 0 
           Df_1031["WeightedProba"] = w1*Df_1031["Proba_0131"]+w2*Df_1031["Proba_0430"]+w3*Df_1031["Proba_0731"] 
           weightdict["w1"] = w1
           weightdict["w2"] = w2
           weightdict["w3"] = w3
           auc += roc_auc_score(Label_1031, Df_1031["WeightedProba"])
           print (auc)
           key = str(auc)
           optdict[key] = weightdict

maxlist = list()
for key in optdict.keys():
    maxlist.append(key)
maxauc = max(maxlist)
print (maxauc)
print (optdict[maxauc])    

optdict[maxauc]["w1"]

Df_1231["WeightedProba"] = optdict[maxauc]["w1"]*Df_1231["Proba_0131"]+optdict[maxauc]["w2"]*Df_1231["Proba_0430"]+optdict[maxauc]["w3"]*Df_1231["Proba_0731"]

for proba in ["Proba_0131","Proba_0430","Proba_0731","WeightedProba"]:
    print ("auc for ",proba," is: ",roc_auc_score(Label_1231,Df_1231[proba]))

print ("Write results to excel...")

perffeatures = csvloader('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/features/PerfFeatures.csv')

DfTest = Df_1231[perffeatures]

DfTest.to_excel('//file-04/FOLDER-REDIRECTION-MALI-ISLER/yunus.karadag/Desktop/result/Df_Test_v1.xlsx')
 

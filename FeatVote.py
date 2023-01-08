
###################
# User Preferences and input files 
# Usge: python3 train_file_name test_file_name 
###################

from sklearn.svm import SVC
model = SVC(max_iter = 10000)


parents = [
['A+C', 'A+G', 'A+K', 'A+N', 'A+S', 'A+Y', 'A-H', 'A-I', 'A-M', 'A-W', 'A-Y', 'C+E', 'C+I', 'C+P', 'C+R', 'C+S', 'C+V', 'C-E', 'C-H', 'C-K', 'C-L', 'C-M', 'C-P', 'C-W', 'D+K', 'D+N', 'D+Q', 'D-H', 'D-I', 'D-W', 'E+H', 'E+K', 'E+M', 'E-F', 'E-H', 'E-L', 'E-R', 'E-V', 'F+Y', 'F-G', 'F-R', 'F-W', 'F-Y', 'G', 'G-K', 'G-R', 'G-S', 'G-V', 'G-W', 'H', 'H+I', 'H+K', 'H+N', 'H-I', 'H-P', 'H-R', 'H-T', 'I+K', 'I+L', 'I+N', 'I+P', 'I+Q', 'I+S', 'I+T', 'I+W', 'I-L', 'I-P', 'I-V', 'I-Y', 'K+L', 'K+M', 'K+P', 'K+Q', 'K+S', 'K+W', 'K-P', 'K-S', 'L+T', 'L+Y', 'L-R', 'L-V', 'L-W', 'L-Y', 'M+N', 'M+P', 'M+S', 'M-R', 'M-S', 'M-T', 'M-W', 'N', 'N+P', 'N+R', 'N+W', 'N-T', 'N-W', 'P', 'P+R', 'P+T', 'P-R', 'Q+R', 'Q-R', 'Q-Y', 'R-T', 'R-V', 'R-Y', 'S+T', 'S+Y', 'T+Y', 'T-Y', 'V+Y', 'aro_non', 'neg_non', 'neg_pos', 'non_non', 'non_pos', 'pol', 'pol_pol', 'pos_pos']
]


import sys 

  
if len(sys.argv) > 1 : 
    train_file = sys.argv[1]
    if len(sys.argv) > 2 : 
        test_file = sys.argv[2]
    else: 
        test_file = 0
else: 
    csv_files = input('Enter the name of your train (required) and test (optional) csv files:  ')
    csv_files = csv_files.split()
    if len(csv_files) > 0  : 
        train_file= csv_files[0]
        test_file = 0 
        if len(csv_files) == 2 : 
            test_file = csv_files[1]
    else: 
        print('\n\nNo input CSV files have been specified /n/n')
        sys.exit(1)








#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
import pandas as pd
import numpy as np 
from sklearn.model_selection import cross_val_score , ShuffleSplit, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from random import shuffle, random
from datetime import datetime


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

###                                                  functions 

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

def loadcsv(datafile) : 
    data = pd.read_csv(datafile) 
    if 'Unnamed: 0' in data.columns : 
        data = data.drop('Unnamed: 0' , axis = 1)
    y = data['cpp']
    X = data.drop('cpp' , axis = 1)
    return X , y 

# correlation 
def correlation (data , cutoff , target) : 
    # finding the correlated features (R >= cutoff) to the target 
  correlated = abs(data.corr()[target].drop([target]))
  correlated = pd.DataFrame(correlated[abs(correlated) >= cutoff]).sort_values('cpp', ascending = False)
  return correlated


# K-Fold Cross Validation
def crossVal(model, X, y) :  
    # calculates and returns the mean score of accuracy and the standard deviation 
    scores = cross_val_score(model, X , y, cv = cv_split, scoring='accuracy')
    mean = round(scores.mean()*100,2)
    std = round(scores.std()*100,2)
    return mean , std 


# Predictions
def predict(model , Xtrain , ytrain , Xtest , ytest) :     
    model.fit(Xtrain , ytrain)
    ptest = model.predict(Xtest)
    test_acc = round(100*accuracy_score(ytest , ptest),2)
    return test_acc , ptest

def scores(model , R , ytrain , S , ytest, constr): 
    cvscore , std = crossVal(model, R, ytrain)
    ptrain = cross_val_predict(model, R, ytrain, cv = 10)
    tn, fp, fn, tp = confusion_matrix(ytrain, ptrain).ravel()
    specificity = round(tn / (tn + fp)*100,2)
    recall = round(recall_score(ytrain, ptrain)*100,2)
    train_acc = round(100*(tp+tn)/(tp+tn+fp+fn),2)
    MCC = round(matthews_corrcoef(ytrain, ptrain)*100,2)
    if X : 
        test_acc , ptest = predict(model , R , ytrain , S , ytest)  
        tn, fp, fn, tp = confusion_matrix(ytest, ptest).ravel()
        test_sn = round(recall_score(ytest, ptest)*100,2)
        test_sp = round(tn / (tn + fp)*100,2)  
    else: 
        test_acc = cvscore + 0 
        test_sn = recall + 0
        test_sp = specificity + 0

            
    #scores = [cvscore,specificity, recall, test_sn, test_sp]
    #scores = [cvscore,test_acc]
    #scores = [specificity, recall, test_sn, test_sp]
    scores_pot = [cvscore]


    if constr == 0 :
        score = min(scores_pot)
    elif constr == 1:  
        diff = np.std(scores_pot)
        score = round(np.mean(scores_pot)-diff,2)
    elif constr == 2 :  
        diff = max(scores_pot)-min(scores_pot)
        score = round(min(scores_pot)-diff,2)
    if X : 
        return [score, cvscore, test_acc, -std, -len(R.columns), recall, specificity, test_sn, test_sp]+ [r for r in R.columns]
    else: 
        return [score, cvscore, cvscore, -std, -len(R.columns), recall, specificity, recall, specificity]+ [r for r in R.columns]


###########################################################################################################
# general variables 

seed = 43
cv_split = ShuffleSplit(n_splits = 10, test_size = .25, train_size = .75, random_state = seed )


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

###                                 feature selection algorithms

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

################################### 
# scanner
def scan(l=1 , n0=1 , parents=0 ,  constr=0 , ga='a' , itrs=1, features=0, pool=[]): 
    if not features : 
      allcols = list(Xtrain.columns)
    else: 
      allcols = features + []

    shuffle(allcols)
    ncol = len(allcols)
    children = [] 

    if not parents : 
        children = [[0, 0, 0, 0, 0, 0,0, -100, -ncol] + sorted(allcols)]  
   
    if parents : 
        children = parents+[]
        nn = len(children[0][9:])
        if n0 > ncol/4: 
            raise Exception('n0 should not exceed {} because the number of features = {}'.format(int(nn/5) , nn))

    for i in range(len(children)): 
        feats = children[i][9:]
        if X != 0 : 
            children[i] = scores(model , Xtrain[feats] , ytrain , Xtest[feats] , ytest, constr)
        else: 
            children[i] = scores(model , Xtrain[feats] , ytrain , 0 , 0, constr)

    print(children[0])


    pool = [] 
    parents = []
    n = n0+1
    void = 1  
    for itr in range(1,itrs+1): 
        top = children[0]        
        if n == 1 :
            children = [children[0]] + [c for c in children[1:] if not c in parents]
            if parents and parents[0] == children[0] : 
                parents = children[1:]
            else: 
                parents = children+[]
        else : 
            parents = children + []
            if n > 1 :
                n = n-1 
        stop_itr = 1
        for par in parents :
            if not limits_reached(par,constr) :
                stop_itr = 0
                ins = par[9:]
                outs = [c for c in allcols if not c in ins]
                shuffle(ins)                     
                shuffle(outs)
                combins = [ins[n*i:n*i+n] for i in range(int(len(ins)/n)+1)]
                combouts = [outs[n*i:n*i+n] for i in range(int(len(outs)/n)+1)]

                if 'd' in ga: 
                    for combi in combins : 
                        newchild = [c for c in ins if not c in combi]
                        if take_votes(combi,newchild) <= 0 : 
                            children, flag, sto = register_child(newchild,children,constr)
                            if flag :
                                update_votes(combi,newchild,-flag)
                                if sto : 
                                    break 


                if 'a' in ga : 
                    for combo in combouts :    
                        co = not_correlated(combo, ins)   
                        if take_votes(co,ins) >= 0 :               
                            newchild = ins + [c for c in co if not c in ins] 
                            children, flag, sto = register_child(newchild,children,constr)
                            if flag :
                                update_votes(co, newchild,flag)
                                if sto: 
                                    break

                if 'r' in ga :                         
                    for combi in combins[:int(len(combins)/5)] : 
                        child = [c for c in ins if not c in combi]
                        for combo in combouts[:int(len(combouts)/5)] : 
                            co = not_correlated(combo, child)   
                            if take_votes(co,child)-take_votes(combi,newchild)  >= 0 :               
                                newchild = child + co 
                                accepted = take_votes(co,newchild)-take_votes(combi,newchild) 
                                if accepted >= 0 : 
                                    children, flag, sto = register_child(newchild,children,constr, 'r')
                                    if flag : 
                                        update_votes(combi,newchild,-flag)
                                        update_votes(co,newchild,flag)
                                        if sto: 
                                            break
        if stop_itr : 
            print('iterations stopped at ', itr)
            break 
        else : 
            void = 0
    return children, void 


def limits_reached(newchild,constr):
    sto = max(newchild[:9])-min(newchild[:9])
    #if (constr in [1,2] and sto < tol_min) or (constr == 0 and sto > tol_max) : 
    if (constr in [1,2] and sto < tol_min) : 
        return 

def register_child(newchild,children,constr) : 
    newchild = sorted(newchild)      
    if X :                             
        newchild = scores(model , Xtrain[newchild] , ytrain , Xtest[newchild] , ytest, constr) 
    else: 
        newchild = scores(model , Xtrain[newchild] , ytrain , 0 , 0, constr)
    flag = 0
    if newchild > children[0] : 
        print(timer(),newchild)
    if not newchild in children : 
        for i in range(len(children)) : 
            child = children[i]            
            if newchild[:9] > child[:9] : 
                com = [f for f in newchild if f in child] 
                flag = 1
                if len(com) / min(len(child),len(newchild)) >= 0.7 : 
                    children[i] = newchild
                else : 
                    children += [newchild]
                break
            elif newchild[:9] < child[:9]:
                flag = -1
        if len(children) < l: 
            children += [newchild]    
    sto = limits_reached(newchild,constr)
    return sorted(children)[::-1][:l], flag, sto

def not_correlated(combo, ins) : 
    ex = [] 
    for c in combo :
        for i in ins : 
            if corrs[c][i] > 0.7 : 
                ex += [c] 
                break
    return [c for c in combo if not c in ex] 

def take_votes(c,curr): 
    accepted = 0 
    for i in curr[9:] : 
        if not i in c: 
            for j in c :  
                accepted += votes[i][j]
    return accepted

def update_votes(c,curr, flag) : 
    for i in curr[9:] : 
        if not i in c : 
            for j in c : 
                votes[i][j] += flag

def timer(): 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    #print("Current Time =", current_time)
    end_time = datetime.strptime(current_time, "%H:%M:%S")
    delta = end_time - start_time
    hours = str(int(delta.total_seconds()/3600))
    if len(hours) == 1 : 
        hours = '0'+hours
    rem = int(delta.total_seconds()%3600)
    minutes = str(int(rem/60))
    if len(minutes) == 1 : 
        minutes = '0'+minutes
    secs = str(int(rem%60))
    if len(secs) == 1 : 
        secs = '0'+secs
    return hours+':'+minutes+':'+secs
##############################################################################################################################


train=pd.read_csv(train_file)
print('train positive ' ,  train[train['cpp']==1].shape)
print('train negative ' ,  train[train['cpp']==0].shape)
if test_file: 
    test=pd.read_csv(test_file)
    print('test positive ' ,  test[test['cpp']==1].shape)
    print('test negative ' ,  test[test['cpp']==0].shape)

Xtrain , ytrain = loadcsv(train_file)   
Xtrain = Xtrain.drop('peps',axis=1)
if test_file: 
    Xtest  , ytest  = loadcsv(test_file)  
    Xtest = Xtest.drop('peps',axis=1)
    X = 1
else: 
    X = 0 

votes = {c:{f:0 for f in Xtrain.columns} for c in Xtrain.columns}
corrs = Xtrain.corr()



# start time and end time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
start_time = datetime.strptime(current_time, "%H:%M:%S")

tol_min, tol_max = 0.05, 5
feats = 0
stop = 100
l = 5
n0 = 10
timer()
print('  score    cv  test_acc   std   length Rsen   Rspec   Ssen    Sspec')
if parents: 
    parents = [[0,0,0,0,0,0,0,0,0]+parents[0]]
    voids = []
else:
    parents, void = scan(l=20, n0=n0, parents=0, constr=0, ga='da', itrs=10, features=feats)

for i in range(1,stop) :  
    constrs = [0,1,2]
    for constr in constrs:   
        votes = {c:{f:0 for f in Xtrain.columns} for c in Xtrain.columns}
        gas = ['da']
        void_limit = 7*len(gas)+len(constrs)
        for ga in gas: 
            if not n0 :
                n0 = int(len(parents[0][9:])/10)
            print('\n',i, '-->' , stop , 'constr=', constr, 'n0=', n0 , 'ga=', ga,  '__________________________________________________________')           
            parents, void = scan(l=l, n0=n0, parents=parents, constr=constr, ga=ga, itrs=n0+3, features=feats)
            voids += [void]     
    if sum(voids[-void_limit:]) > void_limit : 
        print('No better results expected after ', void_limit , 'void outcomes')
        break 


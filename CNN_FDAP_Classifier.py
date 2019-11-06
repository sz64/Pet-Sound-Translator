#https://github.com/HareeshBahuleyan/music-genre-classification/blob/master/5_model_building.ipynb

import pandas as pd
import numpy as np
import itertools
import os
import pickle
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp
from datashape.coretypes import int32

from sklearn.model_selection import cross_val_score

target_names = ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

label_dict = {'Angry':0,
              'Defence':1,
              'Fighting':2,
              'Happy':3,
              'HuntingMind':4,
              'Mating':5,
              'MotherCall':6,
              'Paining':7,
              'Resting':8,
              'Warning':9,
             }



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.figure(figsize=(7, 7), dpi=100)
    #plt.figure(figsize=(8,8))
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=11)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of Cat Sounds', rotation=270, labelpad=30, fontsize=10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
   
    plt.tight_layout()
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    plt.savefig('pred_probas_CNN/'+ str(title)+'.png')
    #plt.show()
    

def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels

def display_results(y_test, pred_probs, name_str, cm = True):
    pred = np.argmax(pred_probs, axis=-1)
    one_hot_true = one_hot_encoder(y_test, len(pred), len(label_dict))
    #print(name_str + 'test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_test, pred_probs)* 100))
    #print(name_str + 'Test Set F-score =  {0:.2f}'.format(f1_score(y_test, pred_probs, average='macro')))
    
    print(name_str +' Test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_test, pred)* 100))
    print(name_str +' Test Set F-score =  {0:.2f}'.format(f1_score(y_test, pred, average='macro')))
    # print(name_str + ' ROC AUC = {0:.3f}'.format(roc_auc_score(y_true=one_hot_true, y_score=pred_probs, average='macro')))
    # if cm:
    #     plot_confusion_matrix(confusion_matrix(y_test, pred), classes=target_names, title= name_str +' Confusion matrix')
    #     plot_confusion_matrix(confusion_matrix(y_test, pred),normalize=True, classes=target_names, title='Normalized ' + name_str +' Confusion matrix')


def support_vector(x_train,y_train, x_test):
    #Parameters for C-SVM
    param = [
        {
            "kernel": ["linear"],
            "C": [0.1, 2.0, 8.0, 32.0]
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 2.0, 8.0, 32.0],
            "gamma": [0.5 ** i for i in [3, 5, 7, 9, 11, 13]] + ['auto']
        }
    ]
    #print('C-Support Vector Classifier starting ...')   
    svm_c = SVC(probability=True)   # request probability estimation
    
    scores_SVM =cross_val_score(svm_c, x_train,y_train,cv=10) #Cross Validation
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores_SVM.mean(), scores_SVM.std() * 2)) # Accuracy: 0.98 (+/- 0.03)
    
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    SVM_clf = model_selection.GridSearchCV(svm_c, param, cv=10, n_jobs=4, verbose=3)
    
    SVM_clf.fit(x_train, y_train)
    pred_probs = SVM_clf.predict_proba(x_test)   # Predict
    display_results(y_test, pred_probs, "C-Support Vector") # Results
    # Save
    # with open('pred_probas_CNN/Support_Vector_classifier.pkl', 'wb') as f:
    #     pickle.dump(pred_probs, f)
    
    #If you want to display the training accuracy
    pred_probs_train = SVM_clf.predict_proba(x_train)
    pred_train = np.argmax(pred_probs_train, axis=-1)
    print('Training Accuracy =  {0:.2f}'.format(accuracy_score(y_train, pred_train)* 100))
    print()

def simple_classifer(x_train,y_train, x_test):
    #Different Classifiers 
    print('Random Forest Classifier starting ...')
    RF_clf = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
    
    scores_RF =cross_val_score(RF_clf, x_train,y_train,cv=10) #Cross Validation
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores_RF.mean(), scores_RF.std() * 2)) # Accuracy: 0.98 (+/- 0.03)
    
    RF_clf.fit(x_train, y_train)
    pred_probs = RF_clf.predict_proba(x_test)   # Predict
    display_results(y_test, pred_probs, "Random Forest") # Results
    # Save
    # with open('pred_probas_CNN/Random_Forest_classifier.pkl', 'wb') as f:
    #     pickle.dump(pred_probs, f)
    
    #If you want to display the training accuracy
    # pred_probs_train = RF_clf.predict_proba(x_train)
    # pred_train = np.argmax(pred_probs_train, axis=-1)
    # print('Training Accuracy =  {0:.2f}'.format(accuracy_score(y_train, pred_train)* 100))
    # print()
    
    print('K-Nearest Neighbours Classifier starting ...')
    KNN_clf = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
    
    scores_KNN =cross_val_score(KNN_clf, x_train,y_train,cv=10) #Cross Validation
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores_KNN.mean(), scores_KNN.std() * 2)) # Accuracy: 0.98 (+/- 0.03)
    
    KNN_clf.fit(x_train, y_train)
    pred_probs = KNN_clf.predict_proba(x_test)   # Predict
    display_results(y_test, pred_probs, "K-Nearest Neighbours") # Results
    # Save
    # with open('pred_probas_CNN/K-Nearest_classifier.pkl', 'wb') as f:
    #     pickle.dump(pred_probs, f)
    
    
    print('Extra Trees Classifier starting ...')
    ET_clf = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
                                  max_features=50, max_depth=40, min_samples_leaf=4)
    
    scores_ET =cross_val_score(ET_clf, x_train,y_train,cv=10) #Cross Validation
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores_ET.mean(), scores_ET.std() * 2)) # Accuracy: 0.98 (+/- 0.03)
    
    ET_clf.fit(x_train, y_train)
    pred_probs = ET_clf.predict_proba(x_test)   # Predict
    display_results(y_test, pred_probs,"Extra Trees") # Results
    # Save
    # with open('pred_probas_CNN/Extra_Trees_classifier.pkl', 'wb') as f:
    #     pickle.dump(pred_probs, f)
    
    print('Linear Discriminant Analysis Classifier starting ...')
    LD_clf = LinearDiscriminantAnalysis()
    
    scores_LD =cross_val_score(LD_clf, x_train,y_train,cv=10) #Cross Validation
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores_LD.mean(), scores_LD.std() * 2)) # Accuracy: 0.98 (+/- 0.03)
    
    LD_clf.fit(x_train, y_train)
    pred_probs = LD_clf.predict_proba(x_test)   # Predict
    display_results(y_test, pred_probs, "Linear Discriminant Analysis") # Results
    # Save
    # with open('pred_probas_CNN/Linear_Discriminant_classifier.pkl', 'wb') as f:
    #     pickle.dump(pred_probs, f)


if __name__ == "__main__":    
    FOLDER_CSV = '.' #Dir of .csv file   
    csv_filename = 'CatSound_Dataset.csv'     #The orginal data level 
    
    FOLDER_FEATS = './CNN_Features' #The extracted features path from CNN Network
    feature_name1 = 'Layer_1_features.npy'  #(2962, 96, 313, 32)
    feature_name2 = 'Layer_2_features.npy'  #(2962, 48, 78, 32)
    feature_name3 = 'Layer_3_features.npy'  #(2962, 16, 19, 32)
    feature_name4 = 'Layer_4_features.npy'  #(2962, 8, 3, 32)
    feature_name5 = 'Layer_5_features.npy'  #(2962, 4, 0, 32)
    
        
    L1_FDAP = np.load(os.path.join(FOLDER_FEATS, feature_name1))
    print("Shape of layer-1", L1_FDAP.shape) #(11843, 32)

    L1_G1 = np.mean(L1_FDAP[:, :25, :, :].astype(int), axis=(1, 2))
    L1_G2 = np.mean(L1_FDAP[:, 23:49, :, :].astype(int), axis=(1, 2))
    L1_G3 = np.mean(L1_FDAP[:, 47:73, :, :].astype(int), axis=(1, 2))
    L1_G4 = np.mean(L1_FDAP[:, 71:, :, :].astype(int), axis=(1, 2))
    print("Feature X1_G1 Shape", L1_G1.shape)
    print("Feature X1_G2 Shape", L1_G2.shape)
    print("Feature X1_G3 Shape", L1_G3.shape)
    print("Feature X1_G4 Shape", L1_G4.shape)
  
    L2_FDAP = np.load(os.path.join(FOLDER_FEATS, feature_name2))
    print("Shape of layer-2", L2_FDAP.shape) #(11843, 32)
 
    L2_G1 = np.mean(L2_FDAP[:, :13, :, :].astype(int), axis=(1, 2))
    L2_G2 = np.mean(L2_FDAP[:, 11:25, :, :].astype(int), axis=(1, 2))
    L2_G3 = np.mean(L2_FDAP[:, 23:37, :, :].astype(int), axis=(1, 2))
    L2_G4 = np.mean(L2_FDAP[:, 35:, :, :].astype(int), axis=(1, 2))
    print("Feature X1_G1 Shape", L2_G1.shape)
    print("Feature X1_G2 Shape", L2_G2.shape)
    print("Feature X1_G3 Shape", L2_G3.shape)
    print("Feature X1_G4 Shape", L2_G4.shape)
   
    
    X1 = np.load(os.path.join(FOLDER_FEATS, feature_name3))
    print("Shape of layer-3", X1.shape)
    
    X1_G1 = np.mean(X1[:, :7, :, :].astype(int), axis=(1, 2))
    X1_G2 = np.mean(X1[:, 6:13, :, :].astype(int), axis=(1, 2))
    X1_G3 = np.mean(X1[:,12:19, :, :].astype(int), axis=(1, 2))
    X1_G4 = np.mean(X1[:, 18:, :, :].astype(int), axis=(1, 2))
    print("Feature X1_G1 Shape", X1_G1.shape)
    print("Feature X1_G2 Shape", X1_G2.shape)
    print("Feature X1_G3 Shape", X1_G3.shape)
    print("Feature X1_G4 Shape", X1_G4.shape)

    X2 = np.load(os.path.join(FOLDER_FEATS, feature_name4))
    print("Shape of layer-4", X2.shape)
    
    X2_G1 = np.mean(X2[:, :4 :, :].astype(int), axis=(1, 2))
    X2_G2 = np.mean(X2[:, 3:7, :, :].astype(int), axis=(1, 2))
    X2_G3 = np.mean(X2[:, 6:10, :, :].astype(int), axis=(1, 2))
    X2_G4 = np.mean(X2[:, 9:, :, :].astype(int), axis=(1, 2))
    print("Feature X2_G1 Shape", X2_G1.shape)
    print("Feature X2_G2 Shape", X2_G2.shape)
    print("Feature X2_G3 Shape", X2_G3.shape)
    print("Feature X2_G4 Shape", X2_G4.shape)
    
    X = np.concatenate((L1_G1,L1_G2,L1_G3, L1_G4,L2_G1,L2_G2,
                        L2_G3, L2_G4, X1_G1, X1_G2,X1_G3, X1_G4,
                        X2_G1, X2_G2, X2_G3, X2_G4), axis=1)
    
    y = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))['label']
    
    print("Total Feature Shape", X.shape)
    print("Label Shape", y.shape)
    
    labels = np.unique(y)
        
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.1, random_state=42)
    
    print('Simple Classifiers starting ...')
    simple_classifer(x_train,y_train, x_test)
    
    print('C-Support Vector Classifier starting ...')
    support_vector(x_train,y_train, x_test)
    
    # print('All classifier result save in disk ')
    # plot_ROC()
    # print('Ploted ROC for multi-class')
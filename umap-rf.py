'''
UMAP and RF---Predicting
'''
from __future__ import print_function
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from astropy.io import fits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns
import os.path
from collections import OrderedDict
from scipy.stats import gaussian_kde
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from imblearn.over_sampling import SMOTE
from astropy.table import Table
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def spectra_int(file_list):
    flux = []
    for file_name in file_list:
        hdulist = fits.open(file_name)
        hdudata = hdulist[0].data
        wl = hdudata[2]
        Flux = hdudata[0]
        # Mask
        m = (wl >= 6530.) & (wl <= 6600.)
        Flux_int = Flux[m]
        flux.append(Flux_int)
    return flux

# Sample for training no-emission
pattern_noemiss =  "For-training-noemission/*.fits"
file_list_noemis = glob.glob(pattern_noemiss)    

flux_noemiss = spectra_int(file_list_noemis)

shape_noemiss = (len(file_list_noemis), len(flux_noemiss[0]))
flux_noemiss_mat = np.array(flux_noemiss).reshape(shape_noemiss)

# Crete data frame
df_noemiss = pd.DataFrame(flux_noemiss_mat)
df_noemiss["Label"] = 0

print("Number objects for training NO emission:", df_noemiss.shape)

# Sample for training with emission
pattern_emis =  "Spectra-Skoda/For-training/*.fits"
file_list_emis = glob.glob(pattern_emis)    

flux_emis = spectra_int(file_list_emis)

shape_emis = (len(file_list_emis), len(flux_emis[0]))
flux_emis_mat = np.array(flux_emis).reshape(shape_emis)

# Crete data frame
df_emis = pd.DataFrame(flux_emis_mat)
df_emis["Label"] = 1

print("Number objects for training with emission:", df_emis.shape)
# Concanate
df_final_train = pd.concat([df_noemiss, df_emis])
y = df_final_train["Label"]
y = y.values

df_final_train_value = df_final_train.drop(['Label'], axis=1)
df_final_train_value = df_final_train_value.values

########################################################################################################################
#Standarized the data ##################################################################################################
########################################################################################################################
sc = StandardScaler()

X_scale = sc.fit_transform(df_final_train_value)

########################################################################################################################
#Aplyinh UMAP to dimesionality reduce ##################################################################################
########################################################################################################################
reducer = umap.UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                    n_components=3,
                    random_state=42)
X_trans = reducer.fit_transform(X_scale)

#Balancing the data will to better classification models. We will try balancing our data using SMOTE.
sm = SMOTE(random_state = 33) #33
X_trans_new, y_new = sm.fit_resample(X_trans, y)

print("Sample after pass for SMOTE:", X_trans_new.shape)

#######################################################################################################################
# Apply the random Forest #############################################################################################
#######################################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Create a Gaussian Classifier
clf_test= RandomForestClassifier(n_estimators=100)
print("Gaussian Classifier:", clf_test)

# Training
X_train, X_test, y_train, y_test = train_test_split(X_trans_new, y_new, test_size=0.3) # 70% training and 30% test
#Train the model using the training sets y_pred=clf.predict(X_test)
clf_test.fit(X_train, y_train)

y_pred=clf_test.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

##########################################################################################################################
####Predicting   #########################################################################################################
##########################################################################################################################

# data for clasifying
pattern_new =  "13J14/*.fits"
file_list_new = glob.glob(pattern_new)    

flux_new = []
inffits = []
for name_fit in file_list_new:
    hdulist = fits.open(name_fit)
    inffits.append(name_fit.split("/")[-1].split(".fit")[0])
    #c = SkyCoord(ra=float(hdulist[0].header["RA"])*u.degree, dec=float(hdulist[0].header["DEC"])*u.degree) 
    #inffits.append('LAMOST{0}{1}'.format(c.ra.to_string(u.hour, sep='', precision=2, pad=True), c.dec.to_string(sep='', precision=1, alwayssign=True, pad=True)))
    inffits.append(hdulist[0].header["DESIG"])
    inffits.append(float(hdulist[0].header["RA"]))
    inffits.append(float(hdulist[0].header["DEC"]))
    hdudata = hdulist[0].data
    wl = hdudata[2]
    Flux = hdudata[0]
    # Mask
    m = (wl >= 6530.) & (wl <= 6600.)
    Flux_int = Flux[m]
    flux_new.append(Flux_int)
      
#flux_new = spectra_int(file_list_new)

shape_new = (len(file_list_new), len(flux_new[0]))
flux_new_mat = np.array(flux_new).reshape(shape_new)

# Crete data frame
df_new = pd.DataFrame(flux_new_mat)

print("Array with new data:", df_new.shape)

df_new_value = df_new.values

df_new_scal = sc.transform(df_new_value)
df_new_scal_tran = reducer.transform(df_new_scal)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_trans_new, y_new)

y_pred_new = clf.predict(df_new_scal_tran)
y_prob = clf.predict_proba(df_new_scal_tran)

##################################################
#creating table file with the results  ###########
##################################################
# Util information from the files

shape_new = (len(file_list_new), 4)
XX_fits = np.array(inffits).reshape(shape_new)
print("Data shape for the final table:", XX_fits.shape)

# Tables with all information 
tab = Table(XX_fits, names=('Namefile', 'ID', 'RA', 'DEC'), meta={'name': 'first table'}, dtype=('S', 'S', 'f8', 'f8'))
  
#Add new colums
#df_new = pd.DataFrame(df_new)
tab['Labels'] = y_pred_new
tab['Prob(No_emissi)'] = y_prob[:,0]
tab['Prob(Emissi)'] = y_prob[:,1]


# Save tables resulting
# m1 = df_ha["Labels"] == 0
m1 = tab["Labels"] == 1
# XX_pn = df_ha[m1]
tab1 = tab[m1]
print("Number of objects with emission:", len(tab1))

tab1.write("emission-objects-13J14.ecsv", format="ascii.ecsv")

# asciifile = "Class-halpha-DR3_noFlag_3ferr_merge-3version.csv"
# asciifile1 = "Class-halpha-DR3_noFlag_3ferr_merge-3version.ecsv" 
# df_ha.to_csv(asciifile)
# Table.from_pandas(df_ha).write(asciifile1, format="ascii.ecsv")

# asciifile2 = "PN-DR3_noFlag_3ferr_merge-3version.csv"
# asciifile3 = "PN-DR3_noFlag_3ferr_merge-3version.ecsv" 
# XX_pn.to_csv(asciifile2)
# Table.from_pandas(XX_pn).write(asciifile3, format="ascii.ecsv")

fig, ax = plt.subplots(figsize=(15, 15)), plt.axes(projection='3d')
ax.scatter3D(X_trans_new[:, 0], X_trans_new[:, 1], X_trans_new[:, 2],
             c=y_new, s=80,
            cmap=plt.cm.get_cmap('Accent', 10))
ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
ax.set_zlabel('component 3')
#ax.set(xlim=[-1.0, 1.0], ylim=[4.0, 5.0], zlim=[-5.0, 6.0])
plt.savefig("result-umap-teste-3d.pdf")


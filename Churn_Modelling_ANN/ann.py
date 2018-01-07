import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Amaç elimdeki dataya göre müşteri bankadan ayrılacak mı ayrılmayacak mı?
# False: Ayrılmaz, True: Ayrılır.

# Part 1 - PreProcessing

dataset = pd.read_csv("Churn_Modelling.csv")
# Okunan veri girdi(X) ve çıktı(Y) olarak ayrıştırılır.
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values    
            
# Encoding kategorik veri. Geography ve Gender kolonlarının sayısal veriye dönüştürülmesi
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Fransa:0, İspanya:2 , Almanya:1
# Yaşanılan ülkenin sonuçlarda baskın gelmemesi için bu işlem yapılır
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

# Cinsiyet özelliğindeki kategorik veri encod edilir. Erkek:1, Kadın:0  gibi
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])  

# Geography kolonundaki veriler 0,1,2 şeklinde numaralandırıldı. Eğer daha fazla ülke olsaydı bu sayı daha da artacaktı.
# 100 adet ülke olabilirdi. 0-100'e kadar numaralandırılmış ülkeler ağın eğitilmesi için verildiğinde
# Numarası 100 olan ülke, 1 olan ülkeye baskın gelebilir. Bundan kurtulmak için OneHotEncoder kullanılır.
# Sonuç olarak Almanya artık 1 ile temsil edilmez. Geography kolonu yerine 3 adet kolon eklenir.
# Almanya : 0 1 0, Fransa: 1 0 0, İspanya: 0 0 1 şeklinde temsil edilir. Yani 3 kolondan her biri
# o ülkenin bulunma durumunu temsil eder.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()   
X = X[:,1:]

# Verinin %80'i train, %20'si test verisi olacak şekilde ayrılır. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
         
# Z-Score normalizasyon işlemi yapılır.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PART 2 - CREATE ANN (http://keras.io - Farklı parametreleri kullanmak isteyenler için)
# Yapay sinir ağını oluşturmak için sıralı bir model kullanacağım belirtilir.
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_model():
    model = Sequential()
    model.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
    model.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform'))
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


# Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
cv_classifier = KerasClassifier(build_fn=build_model,batch_size = 25,nb_epoch = 1000)
accuracies = cross_val_score(estimator = cv_classifier, X = X_train, y = y_train, cv = 10)

accuracySum = 0
for accuracy in accuracies:
    accuracySum += accuracy
 
print(accuracySum / accuracies.size)

# -- Cross Val. Son -- #

classifier = build_model()
# Oluşturulan model train verileri ile eğitilir. Yapay Sinir Ağı eğitilmeye başlar.
# nb_epoch: İterasyon sayısı
classifier.fit(X_train,y_train,epochs = 100)
# Train verileri ile model eğitildikten sonra test verileri ile doğruluk oranlarına bakılır.
loss,accuracy = classifier.evaluate(X_test,y_test)

# Part 3 - Tahmin Etme
# Rastgele bir veri seçilerek tahmin işlemi gerçekleştirilebilir.
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix en fazla 2 sınıfın olduğu veriler için kullanılır.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Elle girilen veri için verilecek çıktı test edilir.
# France,Credit_Score,Gender:Male,Age,Tenure,Balance,Number of Product,Has Credit Card, Is Active Member, Estimated Salary
myTest_data = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
myTest_data = sc.transform(myTest_data)

my_predict = classifier.predict(myTest_data)
my_predict = (my_predict > 0.5)

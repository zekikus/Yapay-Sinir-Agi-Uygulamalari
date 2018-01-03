import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import keras
from keras.utils import np_utils

# Veri dosya üzerinden okunur.
data = pd.read_csv('spambase.data')

# PART 1 - DATAPREPROCESSING

# Okunan veri girdi ve çıktı olarak ayrıştırılır.
input_datas = np.array(data.iloc[:,:57])
output_datas = np.array(data.iloc[:,57])

# Verinin %80'i train, %20'si test verisi olacak şekilde ayrılır. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_datas, output_datas, test_size = 0.2, random_state = 0)

# Z-Score normalizasyon işlemi yapılır.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# PART 2 - CREATE ANN (http://keras.io - Farklı parametreleri kullanmak isteyenler için)
# Yapay sinir ağını oluşturmak için sıralı bir model kullanacağım belirtilir.
# Sıralı Model: Input Layer - Hidden Layer - Output Layer gibi..
# Input Layer ve İlk Hidden Layer katmanı eklenir.
# Units: O katman için kullanılacak olan düğüm sayısı
# Activation: Değerlerin belirli bir aralığa getirilmesi için kullanılacak fonksiyon
# Kernel_Initiliazer: Başlangıçta ağırlıkların hangi metod ile belirleneceği
# Input_shape veya Input_dim: Input layerdaki girdi sayısı. Verinizde her bir satırdaki özellik sayısı 

def build_model():
    model = Sequential()
    model.add(Dense(units = 128, activation = 'relu', kernel_initializer = 'glorot_uniform', input_dim = 57))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 128, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Cross Validation İsteğe Bağlı
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
cv_classifier = KerasClassifier(build_fn=build_model,batch_size = 25,nb_epoch = 1000)
accuracies = cross_val_score(estimator = cv_classifier, X = X_train, y = y_train, cv = 10)

accuracySum = 0
for accuracy in accuracies:
    accuracySum += accuracy
 
print(accuracySum / accuracies.size)

# -- Cross Val. Son -- #

# Oluşturulan model eğitilir. Zorunlu
classifier = build_model()
history_callback = classifier.fit(X_train, y_train, epochs = 100)
acc_history = history_callback.history["acc"]

# Her bir adımdaki "accuracy" değerlerini grafiğe dökmek için (Opsiyonel - Sonuca bir etkisi yok!)
acc_history.sort()
counter = 0
file = open("acc_history_128.csv","a")
result = "Iteration,Fitness\n"
for val in acc_history:
    result = result + str(counter) + "," + str(val) + "\n"
    counter = counter + 1
file.write(result)
file.close()
#### Görselleştirme Son ###########

# Opsiyonel. Matrix üzerinden doğru ve yanlış tahmin edilen değerler gösterilir.
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Train edildikten sonra test datasının doğruluk ve kayıp değerleri hesaplanır.
loss, accuracy = classifier.evaluate(X_test, y_test)
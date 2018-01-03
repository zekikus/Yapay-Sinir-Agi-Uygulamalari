import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import keras
from keras.utils import np_utils


# PART 1 - DATAPREPROCESSING

data = pd.read_csv('cancer.data')
# Bazen kullanılan verilerde kayıp yada eksik değerler olabilir.
# İlk adımda eksik olan değerler -99999 değeri ile değiştirildi. (Opsiyonel)
data.replace('?', -99999, inplace='true')

# Imputer sınıfı ile eksik olan veriler. O veriye karşılık gelen kolonun(özelliğin)
# ortalaması, standart sapması vb. yöntemlerle doldurulur.
imp = Imputer(missing_values=-99999, strategy="mean", axis=0)
data = pd.DataFrame(imp.fit_transform(data))
data = data.drop(0,1)

# Okunan veri girdi ve çıktı olarak ayrıştırılır.
output_data = np.array(data.iloc[:,9])
input_data = np.array(data.iloc[:,:9])

# Çıktı verisi kategorik veri olduğundan bu veriyi numerik veriye çevirmek gerekir.
output_data = np_utils.to_categorical(output_data) 

# Verinin %80'i train, %20'si test verisi olacak şekilde ayrılır. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size = 0.2, random_state = 0)

# Z-Score normalizasyon işlemi yapılır.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PART 2 - CREATE ANN (http://keras.io - Farklı parametreleri kullanmak isteyenler için)
# Yapay sinir ağını oluşturmak için sıralı bir model kullanacağım belirtilir.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_model():
    # Sıralı Model: Input Layer - Hidden Layer - Output Layer gibi..
    model = Sequential()
    # Input Layer ve İlk Hidden Layer katmanı eklenir.
    # Units: O katman için kullanılacak olan düğüm sayısı
    # Activation: Değerlerin belirli bir aralığa getirilmesi için kullanılacak fonksiyon
    # Kernel_Initiliazer: Başlangıçta ağırlıkların hangi metod ile belirleneceği
    # Input_shape veya Input_dim: Input layerdaki girdi sayısı. Verinizde her bir satırdaki özellik sayısı 
    model.add(Dense(units = 20, activation = 'relu', kernel_initializer = 'glorot_uniform', input_dim = 9))
    
    # Overfitting olayını engellemek için Dropout kullanılır. Her adımda belirtilen değer kadar düğümün
    # değeri unutulur. Örneğin: 100 düğüm varsa rate = 0.1 için her adımda 10 düğüm unutulur.
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 20, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 5, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
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
classifier.fit(X_train, y_train, epochs = 100)
# Train verileri ile model eğitildikten sonra test verileri ile doğruluk oranlarına bakılır.
loss,accuracy = classifier.evaluate(X_train, y_train)

# Rastgele bir veri seçilerek tahmin işlemi gerçekleştirilebilir.
tahmin = sc.transform(np.array([8,8,7,4,10,10,7,8,7])).reshape(1,9)
predict_class = classifier.predict_classes(tahmin)[0]

if predict_class == 2: 
    print('Tahmin: Verilen Örnek Kanserli') 
else:
    print('Tahmin: Verilen Örnek Kansersiz')
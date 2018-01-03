import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils

# Veri dosya üzerinden okunur.
data = pd.read_csv('iris.data')

# PART 1 - DATAPREPROCESSING

# Okunan veri girdi(X) ve çıktı(Y) olarak ayrıştırılır.
X = data.iloc[:,:4].values
Y = data.iloc[:,4].values  
            
# Okunan verinin çıktı kısmı birden fazla sınıf içerdiğinden kategorik olarak ayrıştırılır.
# Toplamda 3 sınıf var: Iris-setosa,Iris-versicolor,Iris-virginica
# Bu sınıflar kategorik veri olarak adlandırıldığından programın işleyeceği sayısal verilere dönüştürülür
# Örnek Iris Setosa:      1 0 0
#       Iris Versicolor:  0 1 0
#       Iris virginica:   0 0 1
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y) 
Y = np_utils.to_categorical(Y)    

# Verinin %80'i train, %20'si test verisi olacak şekilde ayrılır. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

# Z-Score normalizasyon işlemi yapılır.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)          

import keras
from keras.models import Sequential
from keras.layers import Dense

# PART 2 - CREATE ANN (http://keras.io - Farklı parametreleri kullanmak isteyenler için)
# Yapay sinir ağını oluşturmak için sıralı bir model kullanacağım belirtilir.
def build_model():
    # Sıralı Model: Input Layer - Hidden Layer - Output Layer gibi..
    model = Sequential()
    # Input Layer ve İlk Hidden Layer katmanı eklenir.
    # Units: O katman için kullanılacak olan düğüm sayısı
    # Activation: Değerlerin belirli bir aralığa getirilmesi için kullanılacak fonksiyon
    # Kernel_Initiliazer: Başlangıçta ağırlıkların hangi metod ile belirleneceği
    # Input_shape veya Input_dim: Input layerdaki girdi sayısı. Verinizde her bir satırdaki özellik sayısı 
    model.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'uniform', input_shape = (4,)))
    # İkinci Hidden Layer katmanı eklenir. (İsteğe bağlı)
    model.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'uniform'))
    # Output Layer katmanı eklenir.
    model.add(Dense(units = 3, activation = 'sigmoid', kernel_initializer = 'uniform'))
    # Loss = Beklenen değer ile hesaplanan değer arasındaki farkı hesaplayarak ne kadar kayıp olduğunu hesaplar
    # Optimizer = Hatanın optimize edilmesini sağlar. Aslında yapay sinir ağının öğrenme işlemini gerçekleştirir.
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

classifier = build_model()
# Oluşturulan model train verileri ile eğitilir. Yapay Sinir Ağı eğitilmeye başlar.
# nb_epoch: İterasyon sayısı
history_callback = classifier.fit(X_train,y_train,nb_epoch=100)

#(Opsiyonel) Eğer modelin eğitilmesi sırasında elde edilen "accuracy" ve "loss" değerlerine -
# erişilmek istenirse kullanılır. Yakınsama grafiğini çizdirmek için kullanılabilir.
acc_history = history_callback.history["acc"]


# Train verileri ile model eğitildikten sonra test dataları ile doğruluk oranlarına bakılır.
loss,accuracy = classifier.evaluate(X_test,y_test)

# Rastgele bir veri seçilerek çıktı önizlenir.
tahmin = sc.transform(np.array([5.0,2.0,3.5,1.0])).reshape(1,4)
predict = classifier.predict(tahmin)
predict_class = classifier.predict_classes(tahmin)[0]
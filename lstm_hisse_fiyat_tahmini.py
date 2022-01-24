import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas_datareader.data as web
from dateutil.relativedelta import relativedelta
from keras.callbacks import TensorBoard


#Verileri bir numpy arrayine X ve Y halinde ekliyoruz.
def veri_kaydirmaca(hisse_verisi, gun_sayisi): 
    data_X, data_Y = [], []
    for i in range(len(hisse_verisi)-gun_sayisi-1):
        a = hisse_verisi[i:(i+gun_sayisi), 0]
        data_X.append(a)
        data_Y.append(hisse_verisi[i + gun_sayisi, 0])
    return np.array(data_X), np.array(data_Y)


#hisse verisini csv dosyasından alıyoruz
def veri_al():
    df = pd.read_csv("ISMEN.IS_2017-2022.csv", index_col=0) #0. indexteki columndan baslayarak almaya devam ediyor
    df.drop(['Volume'], 1, inplace=True)    #pandas ile eklenen csv dosyasının içerisindeki Volume columunu kaldırıyoruz, inplace ederek ise, kopyasını geri döndörmek yerine none geri döndürerek siliyor
    df.drop(['Adj Close'], 1, inplace=True)        
    return df


#Parametreler
verbose=2 #train olurken gösterilecek olan bilginin düzeyini belirliyoruz. Eğerki 0 verirsek, ekranda birşey göstermeden train eder, 1 yaparsak sadece progressbar görünür ve ilerlemeyi o şekilde takip ederiz. Êğerki 2 yi seçersek, bu sefer daha detaylı bir bilgi sunar
gun_sayisi=1
train = True #modeli train etmek istiyorsak true olarak tutabiliriz, ancak test edeceksek false'a çevirmemiz gerekiyor
#DOWNLOAD DATA
hisse_verisi = veri_al()

hisse_verisi_np = np.arange(1, len(hisse_verisi) + 1, 1)

#plot üzerinde kapanış değerlerini göstermek için kullanıyoruz.
def plot_stock(df):
    print(df.tail())
    plt.figure(figsize=(16, 7))
    plt.plot(df['Close'], color='red', label='Close')
    plt.legend(loc='best')
    plt.show()
    
plot_stock(hisse_verisi)

kapanis_degeri = hisse_verisi[['Close']] #Kapanış değerini buraya yazıyoruz

#NORMALIZE
kapanis_degeri = np.reshape(kapanis_degeri.values, (len(kapanis_degeri),1))  #arrayi baska bir forma sokuyor, kapanis_degeri.values column, (len(kapanis_degeri),1) bu da row olacak sekilde dusunebiliriz
scaler = MinMaxScaler(feature_range=(0, 1)) #veriyi 0 ve 1 arasina aliyor
kapanis_degeri = scaler.fit_transform(kapanis_degeri) #datayi 0,1 arasinda olacak sekilde fitleyip, bir degere aktariyor

#TRAIN - TEST SPLIT
train_kapanis = int(len(kapanis_degeri) * 0.75)
test_kapanis = len(kapanis_degeri) - train_kapanis
train_kapanis, test_kapanis = kapanis_degeri[0:train_kapanis,:], kapanis_degeri[train_kapanis:len(kapanis_degeri),:]

trainX, trainY = veri_kaydirmaca(train_kapanis, 1)
testX, testY = veri_kaydirmaca(test_kapanis, 1)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#STM Model
checkpointer = ModelCheckpoint(filepath="hisseKahin.h5", #Callback to save the Keras model or model weights at some frequency.
                               monitor="val_loss", #valudation loss gormek istedigimizi belirtiyoruz
                               verbose=verbose, 
                               save_best_only=True, #sadece en iyi degeri kaydetmesi icin seceriz 
                               mode='auto') #mode: one of {'auto', 'min', 'max'}. 


#eğerki patience kadar epoch yapmasına rağmen hala herhangi bri artış gerçekleşmiyorsa, ilerleme durar.
eStop = EarlyStopping(monitor='val_loss', #Stop training when a monitored metric has stopped improving.
                      patience=50,
                      verbose=verbose)

model = Sequential() #dogrusa bir katman olan sequencial() layerını kullanıyoruz

model.add(LSTM(128, input_shape=(1, gun_sayisi), return_sequences=True)) #128 cikacak lstm bir sequence tahmin etmeye calisiyor, bir cumle verdik cumle tahmin edecek, train listesinde 100 cumle var, her bir cumlede 15-20 cumle var, lstm e cumleyi vermemiz gerekiyor, cumlenin kelime sayisi timestamp oluyor, verdigimiz cumle sayisi batch size oluyor, bu da cumle sayisi oluyor, bu da 
model.add(Dropout(0.2))#sistemin overfit olmasini engellemek icin verinini bir kismini atiyoruz

model.add(LSTM(128, input_shape=(1, gun_sayisi), return_sequences=False)) #
model.add(Dropout(0.2))

model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

model.compile(loss='mse',optimizer='adam') #Model çalışırken mean square error değerine göre compile olacak ve aynı zamanda, train olurken optimizasyon için adam algoritmasını kullanacak
model.summary() # modelimizin nasıl birşeye benzediğnii basitçe gösteren bir tablo gösteriyoruz.




#Train'e başla
if(train):
    train_sonuc = model.fit(trainX, trainY,  #model.fit diyerek modelimizi train ediyoruz ve checkpoınter içerisinde ismini verdiğimiz dosya adı ile kaydediyoruz.
                            epochs=75, 
                            batch_size=8,
                            validation_data=(testX, testY),
                            verbose=verbose, callbacks=[eStop,checkpointer]) #callback bizim her bir epochta kontrol edilecek fonksiyonlarımızdır.

model.load_weights('hisseKahin.h5') #chekcpoint ile kaydettiğimiz dosyamızı yükleyerek test edebiliriz. eğerki train etmeden başka bir dosyayı test edeeceksek buraya dosya pathini verebiliriz.
#Train'e başla
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])



#Training set ve Test seti sonuçları
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

#Training set ve Test seti sonuçları
Train_MAPE = math.sqrt(mean_absolute_percentage_error(trainY[0], trainPredict[:,0]))*100
print('Train MAPE: %.2f' % (Train_MAPE))

Test_MAPE = math.sqrt(mean_absolute_percentage_error(testY[0], testPredict[:,0]))*100
print('Test MAPE: %.2f' % (Test_MAPE))


 
#MAPE ile gösterilen errorleri burada istersek grafik halinde inceleyebiliriz.
plt.figure(figsize=(16, 7))
plt.plot(train_sonuc.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()
#Training set ve Test seti sonuçları

#Tahmin ve gerçek sonuçlar
trainPredictPlot = np.empty_like(kapanis_degeri)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[gun_sayisi:len(trainPredict)+gun_sayisi, :] = trainPredict
testPredictPlot = np.empty_like(kapanis_degeri)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(gun_sayisi*2)+1:len(kapanis_degeri)-1, :] = testPredict
kapanis_degeri = scaler.inverse_transform(kapanis_degeri)


plt.figure(figsize=(16, 7))
plt.plot(kapanis_degeri, 'g', label = 'gerçek')
plt.plot(trainPredictPlot, 'r', label = 'train')
plt.plot(testPredictPlot, 'b', label = 'test')
plt.legend(loc = 'upper right')
plt.xlabel('Günlük veri')
plt.ylabel('Fiyat')
plt.show()
#Tahmin ve gerçek sonuçlar

#Yarınki fiyat
son_gun_fiyat = kapanis_degeri[-1]
son_gun_fiyat_scaled = scaler.fit_transform(np.reshape(son_gun_fiyat, (1,1)))
ertesi_gun_fiyat = model.predict(np.reshape(son_gun_fiyat_scaled, (1,1,1)))
son_gun_fiyat = son_gun_fiyat.item()
ertesi_gun_fiyat = scaler.inverse_transform(ertesi_gun_fiyat).item()

if ertesi_gun_fiyat > son_gun_fiyat: print("Al")
if ertesi_gun_fiyat < son_gun_fiyat: print("Sat")
if ertesi_gun_fiyat == son_gun_fiyat: print("Tut")

print(ertesi_gun_fiyat)
print(son_gun_fiyat)

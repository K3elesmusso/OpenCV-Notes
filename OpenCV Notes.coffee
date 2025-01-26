

{
  "nbformat": 4, -> Bu, Jupyter Notebook dosyasının formatının 4. sürüm olduğunu belirtir.
  "nbformat_minor": 0, -> Bu, dosya formatının 4. sürümüne ait alt sürümün 0 olduğunu ifade eder.
  "metadata": { -> Bu bölüm, dosyayla ilgili genel bilgileri ve ayarları tutar.
    "kernelspec": { -> Notebook'ta hangi kernelin (programlama dili ve çalıştırma ortamı) kullanıldığını belirtir.


"cells": [ -> Jupyter Notebook hücrelerini tutan bir liste.
    {
      "cell_type": "code", -> Bu hücrenin bir "kod hücresi" olduğunu belirtir.
      "execution_count": null, -> Hücrenin henüz çalıştırılmadığını gösterir (çalıştırılınca sıralı bir sayı olur).
      "source": [ -> Hücredeki kodları tutan liste.
        "# Installing `caer` and `canaro` since they don't come pre-installed\n", -> `caer` ve `canaro` kütüphanelerini yüklemek için açıklama satırı.
        "!pip install --upgrade caer canaro" -> Komut satırında `pip` ile bu kütüphaneleri yükler ve günceller.
      ],
      "outputs": [], -> Hücrenin çıktısı, şu an boş (çalıştırılmadığı için).
      "metadata": { -> Hücreye ait ek bilgiler (meta veriler).
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5", -> Hücrenin benzersiz bir kimlik numarası.
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19", -> Hücreye özel başka bir kimlik numarası.
        "trusted": true, -> Bu hücre, güvenilir bir hücre olarak işaretlenmiş.
        "id": "MQoxjirY4S5o" -> Bu hücrenin benzersiz bir kimlik numarası daha.
      }
    },
    {
      "cell_type": "code", -> Yeni bir kod hücresi.
      "metadata": { -> Meta veriler (hücreyle ilgili ek bilgiler).
        "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a", -> Hücrenin benzersiz kimliği.
        "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", -> Hücreye özel başka bir kimlik numarası.
        "trusted": true, -> Bu hücre güvenilir olarak işaretlenmiş.
        "id": "_yldS7bG4S58" -> Hücre için başka bir benzersiz kimlik numarası.
      },
      "execution_count": null, -> Henüz çalıştırılmamış hücre.
      "source": [ -> Hücrenin kod içeriği.
        "import os\n", -> Python'da dosya ve dizin işlemleri için `os` modülünü içe aktarıyor.
        "import caer\n", -> Görüntü işleme kütüphanesi olan `caer` modülünü içe aktarıyor.
        "import canaro\n", -> Derin öğrenme modelleri için kullanılan `canaro` modülünü içe aktarıyor.
        "import numpy as np\n", -> Matematiksel işlemler için kullanılan `numpy` kütüphanesini `np` olarak içe aktarıyor.
        "import cv2 as cv\n", -> OpenCV kütüphanesini görüntü işleme için `cv` adıyla içe aktarıyor.
        "import gc\n", -> Python'da bellek yönetimi için kullanılan `gc` modülünü içe aktarıyor.
        "import sklearn.model_selection as skm \n", -> Scikit-learn kütüphanesinden `model_selection` modülünü `skm` adıyla içe aktarıyor.
        "#pylint:disable=no-member (Removes linting problems with cv)" -> Pylint'teki OpenCV ile ilgili uyarıları kapatır.
      ]
    },
    {
      "cell_type": "code", -> Yeni bir kod hücresi.
      "metadata": {
        "trusted": true, -> Güvenilir bir hücre.
        "id": "TuC64gTq4S6K" -> Hücreye ait benzersiz bir kimlik numarası.
      },
      "execution_count": null, -> Çalıştırılmamış hücre.
      "source": [
        "IMG_SIZE = (80,80)\n", -> Görsellerin yeniden boyutlandırılacağı boyut (80x80 piksel).
        "channels = 1\n", -> Görsellerin tek kanallı olacağını belirtir (siyah-beyaz).
        "char_path = r'../input/the-simpsons-characters-dataset/simpsons_dataset'" -> Görsellerin bulunduğu dizin.
      ]
    },
    {
      "cell_type": "code", -> Yeni bir kod hücresi.
      "metadata": {
        "trusted": true, -> Güvenilir bir hücre.
        "id": "RD5OHUE84S6U" -> Hücreye ait benzersiz bir kimlik numarası.
      },
      "execution_count": null, -> Çalıştırılmamış hücre.
      "source": [
        "# Creating a character dictionary, sorting it in descending order\n", -> Karakterlerin sayılacağı bir sözlük oluşturma ve sıralama açıklaması.
        "char_dict = {}\n", -> Boş bir sözlük tanımlanıyor.


{
  "cell_type": "code", -> Bu bir kod hücresidir.
  "metadata": { -> Hücreye ait meta veriler (ek bilgiler).
    "trusted": true, -> Güvenilir olarak işaretlenmiş.
    "id": "RD5OHUE84S6U" -> Hücreye atanmış benzersiz kimlik numarası.
  },
  "execution_count": null, -> Hücre henüz çalıştırılmamış.
  "source": [
    "char_dict = caer.sort_dict(char_dict, descending=True)\n", -> `caer.sort_dict` ile sözlük sıralanıyor (azalan sırada).
    "char_dict" -> Sıralanmış karakter sözlüğünü ekrana yazdırır.
  ],
  "outputs": [] -> Çıktılar boş, çünkü henüz çalıştırılmamış.
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true, -> Güvenilir olarak işaretlenmiş.
    "id": "OQ09DqmI4S6g" -> Hücreye atanmış benzersiz kimlik numarası.
  },
  "execution_count": null, -> Hücre çalıştırılmamış.
  "source": [
    "# Getting the first 10 categories with the most number of images\n", -> Görsellerin en fazla bulunduğu ilk 10 kategoriyi almayı amaçlayan açıklama.
    "characters = []\n", -> Boş bir karakter listesi oluşturuluyor.
    "        break\n", -> Döngüyü durdurmak için `break` komutu.
    "characters" -> Karakter listesini ekrana yazdırır.
  ],
  "outputs": [] -> Henüz bir çıktı yok.
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "X4UnTWk74S6q"
  },
  "execution_count": null,
  "source": [
    "# Create the training data\n", -> Eğitim verisini oluşturmak için açıklama.
    "train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)" -> `caer.preprocess_from_dir` ile belirtilen klasörden eğitim verisi hazırlanıyor.
  ],
  "outputs": []
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "ZaSuzC2J4S6z"
  },
  "execution_count": null,
  "source": [
    "# Number of training samples\n", -> Eğitim veri setindeki örneklerin sayısını kontrol etmek için açıklama.
    "len(train)" -> Eğitim veri setindeki toplam örnek sayısını döndürür.
  ],
  "outputs": []
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "hSw-V2H24S7A"
  },
  "execution_count": null,
  "source": [
    "# Visualizing the data (OpenCV doesn't display well in Jupyter notebooks)\n", -> Veriyi görselleştirmek için açıklama.
    "import matplotlib.pyplot as plt\n", -> Görselleştirme için Matplotlib'i içe aktarır.
    "plt.figure(figsize=(30,30))\n", -> Görüntü penceresinin boyutunu 30x30 olarak ayarlar.
    "plt.imshow(train[0][0], cmap='gray')\n", -> İlk eğitim görüntüsünü siyah-beyaz (grayscale) olarak görüntüler.
    "plt.show()" -> Görüntüyü ekranda gösterir.
  ],
  "outputs": []
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "arO-90034S7J"
  },
  "execution_count": null,
  "source": [
    "# Separating the array and corresponding labels\n", -> Görsellerin ve etiketlerinin ayrılmasıyla ilgili açıklama.
    "featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)" -> Eğitim setinden özellik (görseller) ve etiketler ayrıştırılıyor.
  ],
  "outputs": []
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "Sl8VnLCY4S7O"
  },
  "execution_count": null,
  "source": [
    "from tensorflow.keras.utils import to_categorical\n", -> TensorFlow'dan `to_categorical` fonksiyonu içe aktarılıyor (etiketleri kategorik hale getirmek için).
  ]
}



{
  "cell_type": "code", -> Bu bir kod hücresidir.
  "metadata": {
    "trusted": true, -> Hücre güvenilir olarak işaretlenmiş.
    "id": "Sl8VnLCY4S7O"
  },
  "execution_count": null, -> Henüz çalıştırılmamış.
  "source": [
    "# Converting numerical labels to binary class vectors\n", -> Sayısal etiketlerin, ikili sınıf vektörlerine dönüştürülmesini açıklar.
    "labels = to_categorical(labels, len(characters))" -> `to_categorical` fonksiyonuyla etiketler bir sıcaklık (one-hot encoding) formatına dönüştürülür.
  ]
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "pzXXrqbt4S7S"
  },
  "execution_count": null,
  "source": [
    "x_train, x_val, y_train, y_val = caer.train_test_split(featureSet, labels, val_ratio=.2)" -> Veri seti eğitim ve doğrulama olarak ikiye ayrılır (`val_ratio` %20 doğrulama için kullanılır).
  ],
  "outputs": []
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "emsrpYWZ4S7W"
  },
  "execution_count": null,
  "source": [
    "del train\n", -> Bellek yönetimi için `train` değişkeni silinir.
    "del featureSet\n", -> Bellek yönetimi için `featureSet` değişkeni silinir.
    "del labels \n", -> Bellek yönetimi için `labels` değişkeni silinir.
    "gc.collect()" -> Python'un çöp toplayıcı mekanizması çalıştırılır ve gereksiz veriler bellekten temizlenir.
  ]
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "NkS1ceD94S7a"
  },
  "execution_count": null,
  "source": [
    "# Useful variables when training\n", -> Eğitim sırasında kullanılacak önemli değişkenler için açıklama.
    "BATCH_SIZE = 32\n", -> Her iterasyonda işlenecek veri sayısını tanımlar.
    "EPOCHS = 10" -> Eğitim sürecindeki toplam epoch (döngü) sayısını belirler.
  ]
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "_atEyygG4S7g"
  },
  "execution_count": null,
  "source": [
    "# Image data generator (introduces randomness in network ==> better accuracy)\n", -> Görüntü veri jeneratörüyle ağda rastgelelik sağlanır ve doğruluk artırılır.
    "datagen = canaro.generators.imageDataGenerator()\n", -> `canaro` kütüphanesinden bir görüntü veri jeneratörü oluşturulur.
    "train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)" -> Eğitim verileri için veri jeneratörü kullanılarak bir akış oluşturulur.
  ]
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "Y8fjXBuH4S7m"
  },
  "execution_count": null,
  "source": [
    "# Create our model (returns a compiled model)\n", -> Eğitim için bir model oluşturulur (derlenmiş model döndürür).
    "model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), \n",
    "                                         loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.9,\n",
    "                                         nesterov=True)" -> Modelin parametreleri ayarlanır: giriş boyutu, kayıp fonksiyonu, öğrenme oranı, vb.
  ]
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "CepcT54J4S7t"
  },
  "execution_count": null,
  "source": [
    "model.summary()" -> Modelin katmanlarını, parametre sayılarını ve diğer bilgileri özetler.
  ]
},

{
  "cell_type": "code",
  "metadata": {
    "trusted": true,
    "id": "AknG90ch4S7-"
  },
  "execution_count": null,
  "source": [
    "# Training the model\n", -> Modelin eğitimi başlatılır.

"execution_count": null,
# -> Kod hücresinin çalıştırılmadığını gösterir. Henüz bir yürütme numarası atanmamış.

"outputs": []
# -> Kod çalıştırıldığında üretilen çıktıların yer alacağı boş bir liste. Bu durumda henüz çıktı üretilmemiş.

"metadata": {
  "trusted": true,
  "id": "AknG90ch4S7-"
},
# -> Metadata bilgisi: 
#    - "trusted": true -> Bu hücrenin güvenilir olarak işaretlendiğini gösterir.
#    - "id" -> Bu hücreye özel bir kimlik numarasıdır, notebook içinde hücreleri izlemek için kullanılır.

"cell_type": "code",
# -> Hücrenin bir kod hücresi olduğunu belirtir.

"source": [
  "characters"
],
# -> Kod hücresinin içeriğini temsil eder. Burada yalnızca "characters" adlı bir değişken belirtilmiş.

"cell_type": "markdown",
# -> Hücrenin bir Markdown (yazı) hücresi olduğunu belirtir. Markdown, başlıklar, açıklamalar ve belgeler için kullanılır.

"source": [
  "## Testing"
],
# -> Markdown hücresinde "## Testing" yazısı yer alır. Bu, "Test Etme" başlıklı bir metin oluşturur.

"test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'"
# -> `test_path` değişkenine bir görüntü dosyasının yolunu tanımlar. 
#    Bu yol, Kaggle'daki 'The Simpsons Characters Dataset' içerisindeki bir test görüntüsünü işaret ediyor.


"plt.imshow(img)\n",  
# -> `matplotlib.pyplot` kullanılarak `img` değişkenindeki görüntüyü ekranda gösterir.

"plt.show()"  
# -> Görüntüyü ekranda göstermeyi tamamlar. 

"execution_count": null,
# -> Hücrenin henüz çalıştırılmadığını gösterir.  

"outputs": []  
# -> Bu kodun çalıştırılması sonucu elde edilen çıktılar için boş bir liste. Şu an için çıktı yok.

"metadata": {  
  "trusted": true,  
  "id": "ETnmB3DC4S8M"  
},  
# -> Metadata bilgisi:  
#    - "trusted": true -> Güvenilir hücre olarak işaretlenmiş.  
#    - "id": -> Hücreye özel kimlik numarası.  

"def prepare(image):\n",  
# -> Görüntüleri işlemek için bir fonksiyon tanımlıyor.  

"image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)\n",  
# -> `cv2` kütüphanesi kullanılarak görüntüyü gri tonlamalı bir formata dönüştürür.

"image = cv.resize(image, IMG_SIZE)\n",  
# -> Görüntüyü belirtilen boyuta (`IMG_SIZE`) yeniden boyutlandırır.  

"image = caer.reshape(image, IMG_SIZE, 1)\n",  
# -> Görüntüyü `caer` kütüphanesi ile yeniden şekillendirir, tek bir kanal ekleyerek boyutları günceller.

"return image"  
# -> İşlenmiş görüntüyü döndürür.  

"predictions = model.predict(prepare(img))"  
# -> Hazırlanan görüntüyü (`prepare(img)`) modele tahmin için gönderir ve sonucu `predictions` değişkenine atar.

"# Getting class with the highest probability\n",  
# -> En yüksek olasılığa sahip sınıfı bulmayı açıklar.  

"print(characters[np.argmax(predictions[0])])"  
# -> `np.argmax` ile tahmin edilen en olası sınıfın indeksini bulur ve `characters` listesinde bu indekse karşılık gelen karakteri yazdırır.  


# pylint:disable=no-member  
# -> `cv` modülünün üyelerine erişirken oluşabilecek lint hatalarını devre dışı bırakır. 

# Installing `caer` and `canaro` since they don't come pre-installed  
# -> `caer` ve `canaro` kütüphanelerini kurmayı hatırlatır, çünkü bunlar varsayılan olarak yüklü değildir.  

# Uncomment the following line:  
# -> Yükleme komutunu çalıştırmak için ilgili satırı yorumdan çıkarmayı önerir.  

import caer  
# -> Görüntü işleme ve sınıflandırma için kullanılan `caer` kütüphanesini içe aktarır.  

import canaro  
# -> Genellikle sinir ağı modelleriyle ilgili işlemler için kullanılan `canaro` kütüphanesini içe aktarır.  

import cv2 as cv  
# -> Görüntü işleme için kullanılan OpenCV kütüphanesini içe aktarır.  

import numpy as np  
# -> Matematiksel hesaplamalar ve veri manipülasyonu için kullanılan `numpy` kütüphanesini içe aktarır.  

import matplotlib.pyplot as plt  
# -> Görüntüleri ve grafik verilerini görselleştirmek için kullanılan `matplotlib.pyplot` kütüphanesini içe aktarır.  

from tensorflow.keras.utils import to_categorical  
# -> TensorFlow ile kategorik verileri dönüştürmek için bir yardımcı fonksiyon sağlar.  

from tensorflow.keras.callbacks import LearningRateScheduler  
# -> TensorFlow'da öğrenme oranını belirli bir plana göre güncellemek için bir geri çağırma fonksiyonu.  

import sklearn.model_selection as skm  
# -> Veri setlerini eğitim ve test için bölmek veya çapraz doğrulama yapmak için kullanılan `scikit-learn` kütüphanesini içe aktarır.  

IMG_SIZE = (80,80)  
# -> Görüntülerin boyutlarını `(80,80)` piksel olarak ayarlar.  

channels = 1  
# -> Görüntülerin kanallarını tanımlar. `1` genelde gri tonlama görüntüler için kullanılır.  

Cropping işlemi (Bir fotoğrafta istenmeyen kısımları çıkarmak için)

cropped = img[50:200, 200:400]  # -> Görüntünün (img) [50:200] satırları ve [200:400] sütunlarından oluşan bir bölümünü kes (crop işlemi).
cv.imshow('Cropped', cropped)  # -> 'Cropped' başlıklı bir pencere aç ve kesilen görüntüyü (cropped) göster.
cv.waitKey(0)  # -> Bir tuşa basılmasını bekle. (0, sınırsız süre bekle anlamına gelir.)
cv.destroyAllWindows()  # -> Tüm açık OpenCV pencerelerini kapat.

# DeepStream Tracker README

Bu README, [NVIDIA DeepStream SDK](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#object-re-identification) içinde kullanılan **Gst-nvtracker** eklentisinin ve NVIDIA’nın referans olarak sunduğu “low-level tracker” kütüphanelerinin (IOU, NvSORT, NvDeepSORT, NvDCF vb.) genel işleyişini açıklamayı amaçlar. Aynı zamanda tek bir tracker ile çoklu kaynak (multi-stream) kullanımı durumunda ID’lerin global olarak artması sorununa `sub-batches` ile nasıl çözüm getirebileceğimizi ve `Re-Identification (Re-ID)` kavramını özetler. Son olarak, YOLOv8M (5 kaynak) - RTX 4090 üzerinde yaptığımız örnek FPS test sonuçlarını paylaşıyoruz.

---

## İçindekiler

1. [Tracker Türleri ve Temel Farkları](#1-tracker-turleri-ve-temel-farklari)  
   1.1. [BBox Tabanlı (Bounding Box Based) Tracker’lar](#11-bbox-tabanli-trackerlar)  
   1.2. [Görüntü Tabanlı (Image Based) Tracker’lar](#12-goruntu-tabanli-trackerlar)  
   1.3. [Genel Karşılaştırma Tablosu](#13-genel-karsilastirma-tablosu)

2. [Tek Bir Tracker ile Birden Fazla Kaynak Kullanımı ve ID Sorunu](#2-tek-bir-tracker-ile-birden-fazla-kaynak)

3. [Sub-batches (Alt-Partisyonlama) ile ID ve Paralelleştirme Çözümü](#3-sub-batches)

4. [Re-ID (Re-Identification) Nedir?](#4-re-id)

5. [Test Sonuçları (YOLOv8M - 5 Kaynak - RTX 4090)](#5-test-sonuclari)

6. [Kaynaklar ve Ek Bilgiler](#6-kaynaklar-ve-ek-bilgiler)

---

## 1. Tracker Türleri ve Temel Farkları <a name="1-tracker-turleri-ve-temel-farklari"></a>

DeepStream içindeki **Gst-nvtracker** eklentisi, altındaki “low-level tracker” kütüphanesi sayesinde obje takibini (MOT: Multi-Object Tracking) gerçekleştiren bir yapıya sahiptir. NVIDIA’nın referans olarak sunduğu toplam **6** adet tracker vardır:

1. **IOU**  
2. **NvSORT**  
3. **NvDeepSORT**  
4. **NvDCF_accuracy**  
5. **NvDCF_perf**  
6. **NvDCF_max_perf**

Bu tracker’lar iki ana kategoride değerlendirilir:

- **BBox Tabanlı (Bounding Box Based) Tracker’lar**: Görsel/piksel işlemine ihtiyaç duymadan, sadece algılanan bounding box’lar arasındaki konum ve boyut benzerliklerine (ör. IOU, Kalman filtresi vb.) dayalı eşleştirme yaparlar.
- **Görüntü Tabanlı (Image Based) Tracker’lar**: Takip edilen objenin piksel verisini işleyerek (ör. DCF, ReID ağı vs.) ek benzerlik metriği kullanırlar.

### 1.1. BBox Tabanlı (Bounding Box Based) Tracker’lar <a name="11-bbox-tabanli-trackerlar"></a>

**IOU Tracker**  
- En temel yaklaşım: İki ardışık karedeki bbox’ların **IOU** değerine bakarak ID atar.  
- Piksel işleme yapmadığı için **çok hızlı** ve **CPU tüketimi düşük**.  
- Sık veya yüksek doğruluklu bir dedektörle kullanıldığında yeterli olabilir; hızlı hareketlerde veya çoklu kapanma durumlarında ID kayıpları yaşanabilir.

**NvSORT**  
- Klasik SORT algoritmasının NVIDIA sürümü.  
- IOU + Kalman filtresi ile hareket bilgisi de katılır.  
- Piksel işlemesi olmadığı için **düşük CPU yükü** ve **hızlı**.  
- Dedektör güvenilir ise IOU’ya göre daha iyi ID stabilitesi sağlar.

### 1.2. Görüntü Tabanlı (Image Based) Tracker’lar <a name="12-goruntu-tabanli-trackerlar"></a>

**NvDeepSORT**  
- DeepSORT algoritmasının NVIDIA optimize edilmiş versiyonu.  
- Re-ID ağı + Kalman filtresi birleştirerek ID tutarlılığını korur.  
- Ekstra bir **ReID derin ağı** çalıştığı için GPU kullanımını artırır.  
- Özellikle aynı görünümlü objelerin yoğun olduğu kalabalık sahnelerde ID kararlılığını ciddi oranda iyileştirir.

**NvDCF (accuracy / perf / max_perf)**  
- Discriminative Correlation Filter (DCF) tabanlı görsel takip ailesi.  
- Dedektörün kare atladığı (ör. false negative) anlarda dahi, DCF ile hedefin konumunu tahmin etmeye devam edebilir.  
- `NvDCF_accuracy`: En yüksek doğruluk ve ek özellikler (re-association vb.).  
- `NvDCF_perf` / `NvDCF_max_perf`: Daha az kanal veya daha küçük feature haritalarıyla **daha hızlı** ama doğruluk biraz kırpılmış.  
- GPU tüketimi, IOU ve NvSORT’a göre daha yüksek ancak NvDeepSORT kadar da ek bir model çalıştırmaz (ReID isteğe bağlı).  

### 1.3. Genel Karşılaştırma Tablosu <a name="13-genel-karsilastirma-tablosu"></a>

| Tracker Adı       | BBox Tabanlı | Görüntü/Piksel Tabanlı |
|-------------------|:-----------:|:-----------------------:|
| **IOU**           | Evet        | Hayır                  |
| **NvSORT**        | Evet        | Hayır                  |
| **NvDeepSORT**    | Hayır       | Evet (ReID)            |
| **NvDCF_accuracy**| Hayır       | Evet (DCF, ReID)       |
| **NvDCF_perf**    | Hayır       | Evet (DCF, ReID)       |
| **NvDCF_max_perf**| Hayır       | Evet (DCF, ReID)       |

---

## 2. Tek Bir Tracker ile Birden Fazla Kaynak Kullanımı ve ID Sorunu <a name="2-tek-bir-tracker-ile-birden-fazla-kaynak"></a>

DeepStream’de genellikle birden fazla kameradan veya videodan gelen kareler **batch** halinde işlenir. **Gst-nvtracker** tek bir instance altında birden fazla kaynağın takibini yapabilir. Bu durumda tüm ID’ler **küresel** bir sayaçtan verilir; örneğin 5 kamera varsa, ID’ler 0,1,2,... şeklinde **kaynak ayırt etmeden** devam eder.

Bazı senaryolarda “Her kaynağın ID’leri kendi başından başlasın” veya “ID alanları çakışmasın” gibi ihtiyaçlar olabilir. Tek bir tracker instance’ı kullanıldığında, varsayılanda bu şekilde global sayaçla çalışır ve ID’ler karışabilir. Bu sorunun çözümü için **sub-batches** özelliği kullanılır.

---

## 3. Sub-batches (Alt-Partisyonlama) ile ID ve Paralelleştirme Çözümü <a name="3-sub-batches"></a>

**Sub-batching**, bir batch içindeki akışları alt parçalara bölerek, her alt parçaya **ayrı** bir low-level tracker instance’ı atayabilmenizi sağlar. Örneğin:

```bash
sub-batches=0,1;2,3;4
ll-config-file=cfg_tracker1.yml;cfg_tracker2.yml;cfg_tracker3.yml
```

1. Sub-batch: Kaynak 0 ve 1 → cfg_tracker1.yml
2. Sub-batch: Kaynak 2 ve 3 → cfg_tracker2.yml
3. Sub-batch: Kaynak 4 → cfg_tracker3.yml
Bu sayede:

- Her sub-batch kendi ID’lerini lokal olarak yönetir (0’dan başlar).
- Her sub-batch farklı tracker algoritması veya farklı parametre ayarı kullanabilir.
- Paralelleştirme artar, CPU/GPU kaynakları daha verimli kullanılabilir.

Tabloda basit bir örnekle görebilirsiniz:

| Sub-batch Index |Kaynak ID’leri | Tracker Tipi   | ll-config-file                   |
|-----------------|:-------------:|:--------------:|:--------------------------------:|
| **0**           | 0,1           | NvDCF_accuracy |config_tracker_NvDCF_accuracy.yml |
| **1**           | 2,3           | NvSORT         |config_tracker_NvSORT.yml         |
| **2**           | 4             | IOU            |config_tracker_IOU.yml            |


## 4. Re-ID (Re-Identification) Nedir? <a name="4-re-id"></a>
Re-Identification (Re-ID), objelerin derin özellik (feature) vektörlerini kullanarak onların aynı obje olup olmadığını saptamaya yarar. Özellikle kalabalık sahnelerde veya uzun süreli takipte ID kararlılığını artırır.

NvDeepSORT, Re-ID ağı zorunlu olarak çalışır (derin bir embedding çıkarır).
NvDCF’de isteğe bağlı olarak re-association için Re-ID devreye alınabilir.
Re-ID ekstra bir inference yükü getirdiği için performans maliyeti vardır.
Çok benzer objelerde ID kaymalarını en aza indirmek için önemli bir mekanizmadır.

## 5. Test Sonuçları (YOLOv8M - 5 Kaynak - RTX 4090) <a name="5-test-sonuclari"></a>
Örnek olarak, YOLOv8M (Primary GIE) + 5 kaynak + RTX 4090 ile her tracker’ı test ettiğimizde aldığımız yaklaşık FPS değerleri şu şekildedir:

| Tracker       | FPS|
|-------------------|:-----------:|
| **IOU**           | Evet        |
| **NvSORT**        | Evet        |
| **NvDeepSORT**    | Hayır       |
| **NvDCF_accuracy**| Hayır       |
| **NvDCF_perf**    | Hayır       |
| **NvDCF_max_perf**| Hayır       |
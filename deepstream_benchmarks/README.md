# DeepStream & YOLOv8 Performans Denemeleri

Bu projede, **DeepStream** ve **YOLOv8** kullanarak çok sayıda video kaynağının (source) gerçek zamanlı işlenmesi üzerine denemeler yaptık. Farklı **batch-size** değerleri ile **RTX 4060** ve **RTX 4090** GPU’ları üzerinde **FPS**, **VRAM** kullanımı ve **GPU** kullanım yüzdesi sonuçları ölçülmüştür. Ayrıca **GStreamer** tabanlı sistemlerde yüksek kaynak sayısına çıkıldığında karşılaşılan sorunlar ve çözümlerine dair notlar paylaşılmıştır.

---

## İçindekiler

1. [Problem Tanımı](#problem-tanımı)
2. [Test Ortamları](#test-ortamları)
3. [DeepStream ile Denemeler (ResNet)](#deepstream-ile-denemeler-resnet)
4. [YOLOv8 Denemeleri (RTX 4060)](#yolov8-denemeleri-rtx-4060)
5. [YOLOv8 Denemeleri (RTX 4090)](#yolov8-denemeleri-rtx-4090)
6. [Notlar ve Öneriler](#notlar-ve-öneriler)
7. [Sonuç](#sonuç)
8. [Kaynaklar](#kaynaklar)

---

## Problem Tanımı

- Çoklu kamera (30, 50, 100, 200+ video kaynağı) senaryolarında gerçek zamanlı (real-time) işlem yapmak istiyoruz.  
- **Batch-size**’ı artırmak, teorik olarak aynı GPU üzerinde daha verimli işleme olanak tanıyabilir.  
- Pratikte ise **donanım kısıtları**, **işletim sistemi limitleri** (ör. file descriptor sayısı) veya **GStreamer** pipeline kısıtları nedeniyle **FPS artışı beklenildiği kadar yüksek olmayabilir**.  
- Özellikle çok sayıda kaynaktan gelen veriler, senkronizasyon sorunlarına ve “buffer drop” uyarılarına neden olmaktadır.

---

## Test Ortamları

- **GPU Modelleri**: 
  - RTX 4060  
  - RTX 4090  
- **Model**:  
  - YOLOv8m (float16 precision, 640x640 veya 1920x1080 giriş çözünürlüğü)
  - DeepStream örnek ResNet modeli
- **Önemli Ayarlar**:  
  1. `export __GL_SYNC_TO_VBLANK=0`  
     - NVIDIA sürücüsünün dikey senkronizasyonu (VSync) kapatması için kullanılır.  
     - FPS’in monitör yenileme hızına (genelde 60 Hz) takılmasını engeller.
  2. `ulimit -n 65536`  
     - Linux’ta varsayılan `file descriptor (FD)` limiti 1024’tür. Çok sayıda GStreamer kaynağı açılacağında bu değer yeterli olmayabilir.  
     - `65536` gibi daha yüksek bir değere çekmek gerekir.

---

## DeepStream ile Denemeler (ResNet)

DeepStream’in kendi **30 kaynaklı örneği** üzerinden, **RTX 4060** kullanılarak yapılan test sonuçları aşağıdaki tabloda gösterilmiştir.

| Kaynak (Source Count) | VRAM Kullanımı (MB) | FPS   | Notlar                                                                                                 |
|:----------------------:|:-------------------:|:-----:|---------------------------------------------------------------------------------------------------------|
| 30                    | 1379                | 30    | 30 kaynak sorunsuz çalışmakta, senkronizasyon hatası görülmedi.                                        |
| 100                   | 4126                | 19.54 | GStreamer tarafında `WARNING from sink_sub_bin_sink1...` hatası oluşuyor. Donanım/codec veya pipeline bölme gerekebilir. |

> **Uyarı**: 100 ve üzeri kaynakta (donanıma göre değişmekle beraber) `gstbasesink.c(3143): gst_base_sink_is_too_late()` benzeri uyarılar/hatalar alınmaktadır.  

### Olası Çözüm Yöntemleri
1. **sync=0**:  
   - Eşzamanlılığı (senkronizasyonu) kapatarak, kuyruk yapısı üzerinden FPS droplarını hafifletebilirsiniz.  
   - FPS’te büyük bir artış görülmeyebilir, ancak buffer drop sayısı azalabilir.

2. **Donanım Yükseltmesi veya Donanım Hızlandırma**:  
   - `nvv4l2decoder` gibi donanım hızlandırma eklentileriyle CPU yükü azaltılabilir.  
   - GPU’nun encode/decode kapasitelerini etkin şekilde kullanmak, darboğazı hafifletebilir.

3. **Pipeline Bölmek**:  
   - Çok yüksek kaynak sayısı için birden fazla pipeline çalıştırıp sonuçları birleştirmek, tek bir pipeline’da yığılmayı önleyebilir.

---

## YOLOv8 Denemeleri (RTX 4060)

### Test Koşulları
- **Model**: YOLOv8m  
- **Precision**: float16  
- **Çözünürlük**: 640x640 (Test 1) ve 1920x1080 (Test 2) giriş görüntüleri  
- **FPS ve VRAM** değerleri ±2 sapma payıyla anlık değişiklik gösterebilir.

#### Test 1: 640x640 Giriş, Model 640x640

| Source Count | Batch-size | FPS    | VRAM (MB) | GPU Util (%) | Notlar                |
|:-----------:|:----------:|:------:|:---------:|:------------:|-----------------------|
| **1**       | 1          | 145    | 344       | %47          |                       |
| **2**       | 1          | 72     | 344       | %47          |                       |
| **3**       | 1          | 49     | 344       | %47          |                       |
| **30**      | 1          | 4.8    | 545       | %47          |                       |
| **1**       | 2          | 281    | 386       | %98          |                       |
| **2**       | 2          | 145    | 393       | %97          |                       |
| **3**       | 2          | 97.28  | 400       | %97          |                       |
| **30**      | 2          | 9.02   | 589       | %97          |                       |
| **1**       | 3          | 253    | 450       | -            |                       |
| **2**       | 3          | 144    | 427       | -            |                       |
| **3**       | 3          | 96     | 436       | -            |                       |
| **30**      | 3          | 9.41   | 625       | -            |                       |
| **30**      | 30         | 8.6    | 1709      | -            |                       |

#### Test 2: 1920x1080 Giriş, Model 640x640

| Source Count | Batch-size | FPS   | VRAM (MB) | Notlar                |
|:-----------:|:----------:|:-----:|:---------:|-----------------------|
| **1**       | 1          | 146   | 356       |                       |
| **2**       | 1          | 72    | 373       |                       |
| **3**       | 1          | 49    | 392       |                       |
| **1**       | 2          | 285   | 398       |                       |
| **2**       | 2          | 142   | 417       |                       |
| **3**       | 2          | 98    | 434       |                       |
| **1**       | 3          | 255   | 432       |                       |
| **2**       | 3          | 144   | 451       |                       |
| **3**       | 3          | 96    | 472       |                       |

> **Not**: Model onnx formatı 640x640’da kalırken, giriş akışı 1920x1080 olarak verilmiştir. Bu durumda iç ölçekleme yapılır.

---

## YOLOv8 Denemeleri (RTX 4090)

Aşağıdaki testlerde yine **float16** hassasiyeti ve **640x640** giriş boyutu kullanılmıştır.

| Source Count | Batch-size | FPS   | VRAM (MB) | GPU Util (%) |
|:-----------:|:----------:|:-----:|:---------:|:------------:|
| **1**       | 1          | 520   | 1258      | %96          |
| **30**      | 30         | 48    | 2610      | %99          |
| **50**      | 50         | 29    | 3521      | %99          |
| **60**      | 50         | 24    | 3551      | %99          |
| **80**      | 50         | 18    | 3676      | %99          |
| **100**     | 50         | 14    | 3840      | %99          |
| **150**     | 50         | 9.45  | 4141      | %99          |
| **200**     | 50         | 7.12  | 4456      | %99          |
| **300**     | 50         | 4.7   | 5067      | %99          |
| **400**     | 50         | 3.5   | 5692      | %99          |

### Batch-size 50 vs. 100 Karşılaştırması

| Source Count | Batch-size | FPS   | VRAM (MB) | GPU Util (%) |
|:-----------:|:----------:|:-----:|:---------:|:------------:|
| **100**     | 50         | 14    | 3840      | %99          |
| **100**     | 100        | 14    | 5785      | %99          |

> **Değerlendirme**: FPS değişmezken (14 FPS), batch-size değerinin artması VRAM kullanımını ciddi şekilde yükseltiyor. GPU kullanımı ise her iki durumda da %99 düzeylerinde seyrediyor.

---

## Notlar ve Öneriler

1. **WARNING from sink_sub_bin_sink1 Hatası**  
   - `sync=0` ile senkronizasyonu devre dışı bırakmak, buffer drop sorununu kısmen hafifletebilir.  
   - Donanım hızlandırma eklentileri (örneğin `nvv4l2decoder`) daha yüksek verim sunabilir.
   - Pipeline’ı bölmek (her bir pipeline’da daha az kaynak işlemek).

2. **FD (File Descriptor) Limiti**  
   - Linux’ta varsayılan FD limiti (1024), çok sayıda (100+) GStreamer kaynağı açıldığında yetersiz kalabilir.  
   - `ulimit -n 65536` veya benzeri bir değerle yükseltme yapılmalıdır.

3. **VSync’in Kapatılması**  
   - `export __GL_SYNC_TO_VBLANK=0` komutu ile dikey senkronizasyonu kapatarak 60 FPS üstüne çıkabilmek mümkündür.  
   - Aksi halde NVIDIA sürücüleri genelde kareleri monitör yenileme hızına göre senkronize eder (ör. 60 Hz).

4. **Batch-size ve Performans**  
   - Yüksek kaynak sayısında batch-size artırmak **VRAM kullanımını** ciddi şekilde artırabilir.  
   - **FPS** ise GPU darboğazı veya diğer bileşenlerdeki kısıtlar nedeniyle aynı kalabilir.  
   - GPU kullanımı genellikle %99 düzeyinde görülmektedir; bu da GPU’nun tam kapasitede çalıştığını ancak sistemde başka bir darboğaz (ör. I/O, bellek, CPU, decoder vb.) olduğunu gösterebilir.

---

## Sonuç

- Denemeler, **çok yüksek kaynak sayısında** (örn. 100+ kaynak) `batch-size`’ın artırılmasının **FPS artışı sağlamadığını** fakat **VRAM kullanımını** önemli ölçüde yükselttiğini göstermektedir.  
- **GPU kullanımı** çoğunlukla %99 düzeyinde seyrederken, sistemdeki **diğer bileşenlerin** de (decoder, I/O, vs.) performansı etkilediği gözlemlenmiştir.  
- **GStreamer** pipeline katmanında, **file descriptor** limiti gibi işletim sistemi parametreleri özellikle 100 üzeri kaynak sayısı için kritik hale gelir.  
- **sync=0**, `nvv4l2decoder` veya pipeline bölme gibi yöntemlerle **buffer drop** ve senkronizasyon sorunları hafifletilebilir.  

---

## Kaynaklar

- [NVIDIA DeepStream Dokümantasyonu](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)  
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  
- [GStreamer Resmi Sitesi](https://gstreamer.freedesktop.org/)


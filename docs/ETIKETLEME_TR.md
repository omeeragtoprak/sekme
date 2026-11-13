# Kendi görüntünde doğruluk ölçmek (etiketleme kılavuzu)

SEKME'nin gerçek görüntüdeki iki sayısı henüz ölçülmedi: **sekme konumunun santimetre doğruluğu** ve
**servislerin ne kadarının yakalandığı**. İkisi de senin işaretlemenle ölçülür; sistemin kendi
çıktısı bu ölçüme girmez. Bu sayfa, yaklaşık bir saatlik bir etiketleme oturumunun adımlarıdır.

## 1. Görüntü ve analiz

- Kamera dip çizginin arkasında ve yüksekte, kortun tamamı kadrajda; 60 fps tercih edilir
  (ayrıntı: README, *Limitations*). Yayın görüntüsü de olur.
- Analiz: `sekme analyze video.mp4 --out runs/<ad>`
- İnceleme Odası: `sekme serve --runs runs`, sonra tarayıcıda http://127.0.0.1:8765
- Bir vuruşu seçip denetçideki karar bölümünde **Ad** alanına adını bir kez yaz. Tarayıcı hatırlar ve
  etiketlerde "Hakem" yerine bu ad kullanılır.

## 2. Sekme noktası (santimetre doğruluğu)

1. Soldaki listeden kararı olan bir vuruş seç.
2. Denetçide **Sekme Noktasını İşaretle**'ye bas.
3. Videoyu ← → ile topun yere değdiği kareye getir (⇧ ile 10 kare).
4. Topun **yere değdiği noktaya**, topun altına tıkla; topun merkezine değil.

İşaretleme sırasında video yalnızca ham görüntüyü gösterir: sistemin sekme işareti, %95 elipsi, top izi
ve saha çizgileri gizlenir; işlenmiş bir video açıksa Kaynak'a geçilir. Zaman çizelgesindeki sekme
çentikleri, mini saha ve karar kartı ise ekranda kalır. Kareyi ve noktayı onlara bakmadan, kendi
gözünle seç; ölçümün değeri buradan gelir.

Sonuç **Analiz → Etiketli doğruluk** kartında görünür:

- ortanca ve p90 hata (cm);
- noktaların ne kadarının sistemin %95 elipsinin içinde kaldığı (iyi kalibre bir sistemde ~%95);
- kesin kararların etiketle uyuşup uyuşmadığı.

Yaklaşık **30 sekme** ilk anlamlı sayı için yeter. Çizgiye yakın sekmeler en değerlileridir.

## 3. Servis (yakalama oranı)

1. Denetçinin **Vuruş** sekmesinde **Servis etiketi** bölümünü aç.
2. Her servisi, **raketin topa değdiği karede** işaretle: **Servis Temasını İşaretle (bu kare)**.
   - Hatalı servisler de sayılır.
   - Kamera geçişinden önce atılan, görünmeyen servisleri işaretleme.
3. Yanlış işaret **Kaldır** ile geri alınır. Kayıt silinmez; yeni bir kayıt eklenir.
4. Klipteki **bütün** servisleri işaretlediysen **Bu koşudaki tüm servisleri işaretledim** kutusunu
   işaretle. Ancak o zaman fazladan servis oranı (kesinlik) hesaplanır.

Sonuç **Analiz → Servis yakalama (etiketli)** kartında görünür. Bulunan servis, işaretinden en çok
±0,5 sn uzaktaki sistem servisidir; eşleşmeler bire birdir.

## Kayıtlar

Etiketler her koşunun klasöründe, eklemeli günlükler olarak durur: `labels.jsonl` ve
`serve_labels.jsonl`. Her kayıtta zaman damgası ve etiketleyenin adı vardır. Analiz yeniden
koşulsa da etiketler kalır.

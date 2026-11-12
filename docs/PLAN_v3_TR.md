# CourtVision v3 — Dürüst Tenis VAR & Analitik Platformu: Teknik Plan (Onay Taslağı)

**Tarih:** 16 Ekim 2026 · **Durum:** ONAY BEKLİYOR — onay gelmeden kod yazılmayacak
**Kapsam:** Referans LinkedIn demosunu kopyalamak değil; ölçülebilir, test edilmiş, belirsizliğini dürüstçe raporlayan bir ürün.

> **Etiketler:** **[D]** = kaynağı açılıp doğrulandı · **[Ç]** = benim çıkarımım/tahminim · **[H]** = hedef (henüz ölçülmedi)

---

## 0. Yönetici Özeti (TL;DR)

1. **Referans** (Otman Heddouch, LinkedIn): YOLO26 ile oyuncu + top, ResNet50 ile 14 kort noktası, Roboflow veri seti, mini-kort ve hız gösterimi. Kapalı kaynak; yazarın GitHub'ında (`otman-ai`) tenis reposu yok. Mimari, abdullahtarek/tennis_analysis desenine birebir uyuyor. **[D]**
2. **Klasördeki mevcut "CourtVision Pro v2.0" tamamen sentetik bir demo:** "US Open" videosu OpenCV ile çizilmiş 6 sn'lik karikatür kort; `.pt` ağırlıkları eğitilmemiş (kort noktaları sabit kodlanmış); ses sentetik; 7 test yalnızca bu çizimin kendi içinde tutarlılığını ölçüyor. Gerçek görüntüden algılama yapan bir boru hattı **yok**. **[D]**
3. **Fiziksel gerçek:** Tek yayın kamerası (30 fps) ile milimetrik çizgi kararı **mümkün değil**. Top iki kare arasında 0.5–1.4 m yol alır; gerçekçi belirsizlik yan çizgide ~5–15 cm, dip/servis çizgisinde ~15–40 cm (1σ) **[Ç]**. Yayınlanmış en iyi gerçek-tenis sonucu 0.63 m ortalama sekme hatası **[D]**. Hawk-Eye 12 kamera + yüksek fps ile mm seviyesinde çalışır; ITF Gold ≤5 mm ortalama hata ister **[D]**.
4. **Bu yüzden ürünün özü "Dürüst VAR":** her çizgi kararı **olasılık + hata elipsi** ile verilir; belirsiz durumda sistem **karar vermez, "İNCELEME" der** ve insan onayına sunar. "IN BY 14 mm" gibi sahte hassasiyet yok.
5. **Fark yaratan yenilikler:** (a) ITF ELC protokolüne benzer ölçüm altyapısı, (b) fizik tabanlı simülatör ile **Kamera Yerleşim / Doğruluk Danışmanı**, (c) 2 telefon + **ses ile senkronizasyon** ile gerçek 3B üçgenleme modu (cm seviyesine iner), (d) yayındaki radar hızıyla **otomatik öz-doğrulama**, (e) Futbol SAOT'tan alınmış insan-onaylı inceleme odası + denetim kaydı.

---

## 1. Referans Analizi

| Öğe | Referans (LinkedIn) | Kaynak |
|---|---|---|
| Oyuncu tespiti/takibi | YOLO26 (Ultralytics) | Gönderi metni **[D]** |
| Top tespiti | YOLO26 (genel nesne dedektörü) | Gönderi metni **[D]** |
| Kort | ResNet50, 14 anahtar nokta | Gönderi metni **[D]** |
| Veri | Roboflow Universe (bağlantı gönderide kesik: `universe.roboflow.com/s`) | **[D]** |
| Çıktı | Açıklamalı video, mini-kort, oyuncu hızı/mesafe, top hızı | **[D]** |
| Kod | Yayınlanmamış | GitHub API **[D]** |

**Gönderi yorumlarındaki itirazlar tam olarak bizim gap listemiz:** "kalibrasyondaki küçük hatalar hız ve mesafe tahminlerini kaydırır", "ralli sırasında kamera açısı değişirse mini-kort ne olur?", "top nasıl takip edildi?" **[D]**

**Ekran görüntüsü notu:** Bu oturumda ekran görüntüsü bana ulaşmadı. Önceki plandaki tarif (sarı 0–13 kort noktaları, yeşil `P1 3.4 km/h` kutuları, sağ üst mini-kort, alt bar `PLAYER 1 · ID 62 | 3.4 km/h | 4 m | peak 12.1 … BALL SPEED 127.8 km/h … MATCH TIME 3.3s`) esas alındı. Piksel-sadakati isteniyorsa görüntüyü tekrar paylaşman yeterli.

---

## 2. Mevcut Durum Denetimi (As-Is)

| Varlık | Gerçek durum | Karar |
|---|---|---|
| `courtvision/core/constants.py` | ITF kort ölçüleri doğru | **Yeniden kullan** |
| `courtvision/core/homography.py`, `physics_3d.py` | Matematik iskeleti var, gerçek veride hiç denenmemiş | Referans al, yeniden yaz + test et |
| `courtvision/detectors/vision_pipeline.py` | Eğitilmemiş ağ tanımları | Gerçek ağırlıklı modellerle değiştir |
| `scripts/build_and_run.py` (949 satır) | Sentetik video/ses üreticisi | Fikir olarak **simülatöre** dönüştür |
| `weights/*.pt` | Eğitilmemiş, sabit kodlu | Kullanma |
| `output/*` | Sentetik çıktı | Arşivle; asla "sonuç" olarak gösterme |
| `SENIOR_ARCHITECTURE_PLAN.md` | Kaynakların çoğu doğru; 4 sorun (aşağıda) | Arşivle, v3 planı bunun yerine geçer |
| Testler | 7/7 geçiyor, ama sentetik tutarlılık testi | Gerçek-veri doğruluk testleriyle genişlet |

**Önceki plandaki kaynak doğrulaması:** arXiv 2607.28500 (Sandholtz vd.) **[D]**, CalTennis = arXiv 2606.20542 **[D]**, arXiv 2506.05763 **[D]**, TT3D **[D]**, F3Set = ICLR 2025 **[D]**. **Sorunlu olanlar:** "Improved ST-GCN" (ResearchGate 386523248) **bulunamadı**; arXiv 2409.11760 tenis değil **masa tenisi**; "3.6 mm" Mather 2008'e değil Hawk-Eye'ın kendi iddiasına ait; "IN BY 14 mm" türü çıktılar tek kamerayla fiziksel olarak savunulamaz.

**Ortam:** Apple M4, 16 GB, macOS 26.6.2, Python 3.13.7 (arm64), PyTorch 2.10 (MPS aktif), Ultralytics 8.4.14, onnxruntime 1.27, Node 26, uv, ffmpeg, yt-dlp. **Kısıt: diskte yalnızca ~17 GB boş (%92 dolu).** Klasör git deposu değil.

---

## 3. Gap Analizi (Rakip Haritası)

### 3.1 Açık kaynak (16 repo doğrulandı **[D]**)

| Proje | Güçlü yanı | Kritik eksik |
|---|---|---|
| abdullahtarek/tennis_analysis (897★, lisanssız) | Referansın atası; YOLO + ResNet50 kort | Kort sadece **ilk karede**; homografi yok; hız = 2B mesafe / sabit 24 fps; sekme yok; in/out yok; test yok |
| yastrebksv/TennisProject + TrackNet + CourtDetector (lisanssız) | Her karede homografi, sahne kesme tespiti, CatBoost sekme | Hız yok, in/out yok, metrik yok |
| HarshTomar1234/Tennis-Vision (MIT, Eyl 2026) | En titizi: 412 test, 3B uçuş, radar karşılaştırması | Ralli hızları doğrulanmamış; kamera kesmesi yok; **in/out yok**; yavaş |
| philippdubach/tennis-vision (MIT) | Skor tabelası OCR + skor mutabakatı | Geometrik in/out yok; layout sabit kodlu |
| nttcom/WASB-SBDT (MIT) | Tenis top modeli: F1 95.6 (TrackNet tenis seti) | Uçtan uca sistem değil |
| ArtLabss/tennis-tracking | Sekme sınıflandırıcı | "HawkEye" diyor ama **in/out mantığı yok** |
| SEU Hawkeye (MIT) | Tek gerçek in/out geometrisi | **4 kalibre kamera** gerektiriyor |

**Ortak boşluklar:** (1) Belirsizliği ölçülmüş tek-kamera in/out kararı **hiçbir yerde yok**; (2) hızlar çoğunlukla düz 2B; (3) kamera hareketi/kesme nadiren ele alınıyor; (4) eşikler kare sayısına bağlı (60 fps'de bozuluyor); (5) değerlendirme ve test neredeyse yok; (6) temel yapı taşlarının çoğu **lisanssız**.

### 3.2 Ticari ve profesyonel sistemler **[D]**

| Sistem | Donanım | Doğruluk iddiası | Not |
|---|---|---|---|
| Hawk-Eye Live (ATP 2025'ten beri tüm kortlarda) | 12 kamera/kort | "mm seviyesi" (satıcı iddiası) | Wimbledon 2025: operatör yanlışlıkla izlemeyi kapattı, 3 karar kaçtı → manuel kapatma kaldırıldı |
| ITF Gold / Silver / Bronze | — | Gold: ±100 mm bandında %100 doğru, ort. ≤5 mm · Silver: ±25 mm'de ≥%80 · Bronze: ≥%75/%65 | **Bizim ölçüm çerçevemiz olacak** |
| SwingVision | 1 iPhone | Çizgiye 10 cm içindeki toplarda %97 (satıcı) | ITF ELC sınıfı değil |
| Zenniz / PlayReplay | çoklu kamera (+mikrofon) | Silver | |
| Roland-Garros 2026 | insan hakem | "Makine %100 güvenilir değil" | Toprakta iz tartışması |

**Futbol SAOT'tan alınacak desenler:** sistem önerir → insan onaylar; karar verildiği veriden üretilen 3B animasyon; her karar için denetim kaydı (kim, ne zaman, model sürümü, güven, kullanılan kareler); sistemin sessizce devre dışı kalamaması.

---

## 4. Fiziksel Sınırlar ve "Dürüst VAR" Tasarım İlkesi

**Neden mm mümkün değil (tek kamera, 30 fps):** [Ç, kaynaklarla desteklenmiş]
- Kareler arası 33 ms; sekme anı kameraya **hiç yakalanmaz**, iki parabolün kesişiminden çıkarılır (TT3D: 25 fps'de 10 m/s'de 40 cm'e kadar sekme hatası **[D]**).
- 720p yayın görüntüsünde uzak dip çizgide 1 piksel ≈ derinlikte 5–7 cm.
- Kalibrasyon + top merkezi hatası birkaç piksel ekler.

**Tasarım ilkesi:** Sekme noktası tahmini **b̂** ve kovaryansı **Σ** (yörünge uydurma + kalibrasyon belirsizliğinin Monte Carlo yayılımı) hesaplanır. Çizginin dış kenarına işaretli mesafe **m** (top izi yarıçapı dahil; çizgiye değen top İÇERİDE sayılır), çizgi normaline göre **σₘ = √(nᵀΣn)**.

**P(IN) = Φ(m / σₘ)** → Karar katmanları:
- **P(IN) ≥ 0.95 → İÇERİDE** · **P(IN) ≤ 0.05 → DIŞARIDA** · arası → **İNCELEME GEREKLİ** (insan karar verir)
- Ekranda: tahmin edilen iz + **%95 güven elipsi** + "P(IN) = %83 · ±12 cm" — asla sahte mm yok.
- σ değerleri gerçek veriyle **kalibre edilir** (güvenilirlik diyagramı, ECE). "%90 eminim" dediğinde gerçekten ~%90 doğru olmalı.

**Doğruluğu artırmanın gerçek yolu** (Faz 7): kendi çekimin (iPhone 120–240 fps → kareler arası 4–8 ms, top 10–20 cm ilerler) ve/veya 2 telefonla stereo üçgenleme.

---

## 5. Ürün Özellikleri

### 5.1 Algılama Katmanı
- **Kort kalibrasyonu (her karede):** 14 nokta ısı-haritası modeli + çizgi iyileştirme → homografi **ve** tam kamera modeli (odak + poz). **Yenilik:** file direği tepeleri (1.07 m) ve file merkezi (0.914 m) düzlem dışı noktalar olarak kalibrasyona eklenir → 3B için gerçek kamera. Zamansal filtre + **geçerlilik kapısı** (yeniden izdüşüm hatası eşiği aşılırsa o kare "ölçülemez" işaretlenir, sessizce yanlış sayı üretilmez).
- **Sahne kesme / tekrar (replay) tespiti:** yayın kesmeleri, yakın plan ve tekrarlar ayıklanır; analiz yalnızca ana kamerada yapılır.
- **Oyuncular:** Değiştirilebilir dedektör (YOLO26 *veya* RF-DETR) + ByteTrack; yakın/uzak oyuncu kimliği, top toplayıcı/hakem/seyirci filtresi.
- **Top:** WASB (MIT, tenis ağırlığı mevcut) birincil; TrackNetV3 / TOTNet (oklüzyon) alternatif. Hareket bulanıklığı merkezi etiket kuralı.
- **Poz:** RTMPose (rtmlib, Apache-2.0) oyuncu kırpıntıları üzerinde.

### 5.2 Olay ve 3B Katmanı
- **Vuruş/sekme tespiti:** (1) 2B yörünge kırılması + (2) öğrenilmiş sınıflandırıcı + (3) **ses onset füzyonu** (yayın sesindeki raket/sekme darbesi). Eşikler **saniye cinsinden**, fps'ten bağımsız.
- **3B yörünge:** Olaylar arası parçalı fizik uydurma (yerçekimi + hava direnci, opsiyonel Magnus); gerçek 3B hız, file üstü geçiş yüksekliği, sekme noktası + kovaryans.
- **Ralli segmentasyonu:** servis tespiti, ralli başlangıç/bitiş, vuruş sırası.

### 5.3 Hakemlik (VAR) Katmanı
- **Olasılıksal çizgi kararı** (bölüm 4) + bağlama duyarlı kural bölgesi (servis kutusu / tekler / çiftler).
- **Kural motoru + skor durum makinesi:** servis hatası, çift hata, ace, winner, hata; tie-break; skor tabelası OCR ile **çapraz doğrulama** (yayında varsa).
- **"Challenge" akışı:** kullanıcı bir sekmeye itiraz eder → sistem zoom kartı, 3B tekrar, P(IN), kanıt kareleri sunar → insan onaylar/değiştirir → **denetim kaydı**.
- **Sağlık göstergesi:** kalibrasyon kaybı, top kaybı, kesme sırasında "SİSTEM KÖR" bandı — Wimbledon dersi.

### 5.4 Analitik
- Oyuncu: mesafe, hız/ivme profili, ısı haritası, toparlanma pozisyonu, yön değiştirme sayısı.
- Vuruş: 3B hız, derinlik, açı, file geçişi, vuruş tipi (FH/BH/servis/vole — poz + yörünge).
- Servis haritaları, sekme dağılımı, ralli uzunluğu istatistikleri.
- Kinematik kort hakimiyeti (ulaşma süresi tabanlı) — deneysel olarak etiketlenir.

### 5.5 Arayüz: "İnceleme Odası" (Web)
- Kare-doğru video oynatıcı, olay zaman çizelgesi, çizgi kararı kartları, filtreler.
- **three.js 3B tekrar** (karar verildiği aynı veriden), mini-kort, ısı haritaları, grafikler.
- İnsan onayı / düzeltme → düzeltmeler **etiket olarak** saklanır (veri çarkı / aktif öğrenme).
- Dışa aktarma: annotasyonlu video (Klasik HUD = referans görünümü, Pro HUD), JSON/Parquet/CSV, HTML maç raporu.
- **LLM açıklaması (opsiyonel):** yapılandırılmış olay JSON'u + birkaç kare → küçük bir dil modeliyle doğal dil özeti (~$0.001/itiraz **[D]**). **LLM asla karar vermez**, yalnızca açıklar.

### 5.6 Yenilikçi Özellikler (rakiplerde yok)
1. **Ölçüm altyapısı (ITF-benzeri protokol):** her modül için metrikler + çizgi kararları için "±25 mm / ±100 mm bandında doğru karar oranı", ECE, karar vermeme oranı. Rapor dürüstçe "bu kurulum ITF Bronze eşdeğerinin altında/üstünde" der.
2. **Fizik simülatörü + Kamera Yerleşim Danışmanı:** bilinen 3B gerçeklikli sentetik ralliler (kamera modeli, bulanıklık, gürültü) → "kameranı şuraya, şu fps'le koyarsan dip çizgide beklenen belirsizlik ±X cm". Eski sentetik kod burada değer kazanır.
3. **Çift telefon + ses senkronu:** iki kayıt, darbe seslerinin çapraz korelasyonu ile ms seviyesinde senkronlanır → stereo üçgenleme.
4. **Radar ile öz-doğrulama:** yayındaki servis hızı göstergesi OCR'la okunur, sistemin 3B hızıyla otomatik karşılaştırılıp hata istatistiği tutulur.
5. **Yayın Hawk-Eye grafiklerinden etiket hasadı:** yayında gösterilen resmi itiraz sonuçları değerlendirme etiketi olarak toplanabilir (araştırma amaçlı).

---

## 6. Mimari

```
Video (dosya / kendi çekimin / 2 telefon)
   │  PyAV: kare-doğru zaman damgası + ses kanalı
   ▼
[io] sahne kesme & replay tespiti ──► yalnızca ana kamera segmentleri
   ▼
[calib] kort noktaları → homografi + tam kamera (PnP+odak, file direkleri) → zamansal filtre → geçerlilik kapısı
   ▼
[detect] oyuncu (YOLO26|RF-DETR) · top (WASB) · poz (RTMPose)      ← değiştirilebilir arayüzler (Protocol)
   ▼
[track] ByteTrack oyuncular · top iz bağlama + boşluk doldurma
   ▼
[events] vuruş/sekme: yörünge kırılması + sınıflandırıcı + ses onset füzyonu
   ▼
[physics] parçalı 3B balistik uydurma → hız, file geçişi, sekme noktası + Σ
   ▼
[officiating] P(IN) çizgi kararı · kural motoru · skor FSM · OCR mutabakatı
   ▼
[analytics] oyuncu/vuruş/servis istatistikleri, hakimiyet
   ▼
[store] Pydantic şemalar · Parquet (kare-izleri) · SQLite (maç/olay/denetim kaydı)
   ▼
[api] FastAPI + WebSocket (iş kuyruğu, ilerleme)  ──►  [web] React + Vite + three.js İnceleme Odası
[render] Klasik HUD (referans) · Pro HUD · 3B tekrar verisi
[eval] metrikler, benchmark, güvenilirlik diyagramları   [sim] fizik simülatörü + yerleşim danışmanı
```

**Klasör yapısı (öneri):**
```
tenis/
  legacy/v2_synthetic/      # mevcut her şey (taşınmadan önce git ile anlık görüntüsü alınır)
  src/courtvision/{io,calib,detect,track,events,physics,officiating,analytics,store,render,api,eval,sim}/
  web/                      # React + Vite + three.js
  configs/  docs/  tests/{unit,integration,golden}/
  data/ (gitignore)  weights/ (gitignore + sha256 manifest)
```

**Mühendislik kuralları:** tipli kod (pydantic v2, mypy), yapılandırma YAML + doğrulama, deterministik çalıştırma (seed, model sürümü, sha256), her çıktıda **kaynak bilgisi** (hangi model/hangi kareler), ruff + pytest, golden-test videoları.

---

## 7. Teknoloji Yığını (Ekim 2026 sürümleri **[D]**)

| Bileşen | Seçim | Lisans | Neden / Not |
|---|---|---|---|
| Top | WASB (tenis ağırlığı) → gerekirse ince ayar | MIT | TrackNet tenis setinde F1 95.6 |
| Kort | TennisCourtDetector mimarisi (14 nokta ısı haritası) | **Repo lisanssız** | Medyan 1.83 px; ticari için kendi verimizle yeniden eğitim |
| Oyuncu | RF-DETR (S/M) **veya** YOLO26 | Apache-2.0 / **AGPL-3.0** | Lisans kararına bağlı (soru 2) |
| Takip | `trackers` (ByteTrack/BoT-SORT) 2.6.1 | Apache-2.0 | BoxMOT AGPL olduğu için değil |
| Poz | rtmlib 0.0.16 (RTMPose) | Apache-2.0 | ONNX Runtime, mmcv gerektirmez |
| Çıkarım | PyTorch MPS (geliştirme) → ONNX Runtime CoreML EP (üretim) | BSD/MIT | M4 Neural Engine |
| Video I/O | PyAV 19 + OpenCV | BSD/Apache | Kare-doğru zaman damgası |
| Backend | FastAPI + uvicorn + WebSocket | MIT/BSD | |
| Frontend | React 19 + Vite 8 + three.js (R3F) | MIT | |
| Kare-doğru oynatıcı | WebCodecs + Mediabunny / requestVideoFrameCallback | MPL-2.0 | |
| Veri | Pydantic v2, Parquet (pyarrow), SQLite | | |
| LLM (ops.) | küçük bir ticari dil modeli | ticari API | yalnızca açıklama |
| Ortam | `uv` ile izole venv, sabitlenmiş sürümler | | global Python'u kirletmeden |

**Lisans riskleri:** Ultralytics YOLO26 AGPL-3.0 (ağ üzerinden sunulan kapalı ürün = kaynak açma yükümlülüğü veya ücretli lisans); yastrebksv/abdullahtarek/F3Set **lisanssız** (yalnızca araştırma); SAM 3 özel lisans; DEIMv2 ticari kullanım yasak; "Hawk-Eye"/"VAR" marka adları ürün adında kullanılmamalı; "CourtVision" adı NBA'deki Clippers CourtVision ile çakışabilir (yayın öncesi kontrol).

---

## 8. Veri ve Değerlendirme Stratejisi

| Veri | Kullanım | Lisans / Not |
|---|---|---|
| TrackNet tenis seti (~19.8k kare, 1280×720, 30 fps) | Top modeli testi | Lisans belirtilmemiş → araştırma |
| TennisCourtDetector seti (8.841 görüntü) | Kort modeli testi/eğitimi | Lisans belirtilmemiş → araştırma |
| Wikimedia Commons Davis Cup 2018 klipleri (6 adet, ~97 sn, 720p) | Gerçek-dünya dayanıklılık (toprak, yan açı, pan yapan kamera) | **CC BY-SA 4.0** **[D]** |
| Senin videoların / çekimlerin | Ana değerlendirme + ince ayar | Tam hak |
| Fizik simülatörü | 3B ve çizgi kararı için **kesin** gerçeklik | Kendi üretimimiz |
| Yayın videoları (yt-dlp) | Yalnızca yerel araştırma, dağıtım yok | Telif riski — tavsiye: kendi hakkın olan görüntü |

**Disk planı:** 17 GB boş → veri setlerinin yalnızca alt kümeleri indirilir, ara çıktılar sıkıştırılır, büyük setler için harici disk önerilir.

**Hedef metrikler [H] — ölçülmeden "başarıldı" denmeyecek:**

| Modül | Metrik | Hedef |
|---|---|---|
| Kort | Medyan anahtar-nokta hatası (720p) | ≤ 3 px; geçerlilik kapısı yanlış-kabul < %1 |
| Top | F1 @ 4 px (TrackNet tenis testi) | ≥ 0.90 |
| Oyuncu | Ana iki oyuncu kimlik tutarlılığı | ≥ %99 kare |
| Olaylar | Sekme/vuruş F1 (±2 kare) | ≥ 0.85 |
| Hız | Servis hızı MAE (yayın radarına karşı) | ≤ 8 km/h |
| Çizgi kararı | "Kesin" kararlarda doğruluk / ECE / karar-vermeme oranı | ≥ %98 / ≤ 0.05 / raporlanır |
| Sistem | M4'te 720p30 işleme hızı | v1: ≥ 0.5× gerçek zaman; optimizasyon sonrası ≥ 1× |

---

## 9. Fazlar ve Kabul Kriterleri

| Faz | İçerik | Kabul kriteri (kanıt) |
|---|---|---|
| **F0 Temel** | `git init` + mevcut durumun ilk commit'i; `legacy/` arşivi; `uv` venv; paket iskeleti; ruff/mypy/pytest; config sistemi | Eski 7 test arşivde hâlâ çalışıyor; yeni iskelet testleri yeşil |
| **F1 Veri + Değerlendirme + Simülatör** | Veri indirme betikleri (sha256), etiket formatları, metrik kütüphanesi, fizik simülatörü | Simülatör bilinen 3B gerçeklik üretir; metrik birim testleri |
| **F2 Algılama v1** | Kort kalibrasyonu + kesme tespiti + oyuncu + top + poz, gerçek ağırlıklarla | Hedef metrikler gerçek veri setlerinde ölçülüp raporlanır; **gerçek videoda referans görünümü (Klasik HUD)** |
| **F3 Olaylar + 3B** | Vuruş/sekme, ses füzyonu, 3B yörünge, hızlar | Simülatörde 3B hata raporu; radar karşılaştırmalı servis hızı MAE |
| **F4 Hakemlik** | P(IN) motoru, kural motoru, skor FSM, OCR mutabakatı | Simülatörde ITF-benzeri bant doğruluğu + ECE; gerçek klipte el etiketli sekmelerle kontrol |
| **F5 İnceleme Odası** | FastAPI + React + three.js, challenge akışı, denetim kaydı | Uçtan uca: video yükle → işle → incele → onayla; tarayıcıda doğrulanmış |
| **F6 Analitik + Rapor** | İstatistikler, ısı haritaları, HTML/PDF rapor, LLM açıklaması | Rapor alanları test verisiyle tutarlı |
| **F7 Gelişmiş (ops.)** | Çift telefon + ses senkronu, yüksek fps modu, yerleşim danışmanı | Simülatör + gerçek çekimde belirsizlik karşılaştırması |
| **F8 Sağlamlaştırma** | ONNX/CoreML hızlandırma, paketleme, dokümantasyon | Performans hedefi ölçülmüş; kurulum kılavuzu ile temiz kurulum |

**MVP sınırı = F0–F4 + temel F5.** Her faz sonunda: testler + ölçüm raporu + sana kısa durum özeti. Çalışmayan veya hedefin altında kalan her şey açıkça belirtilir.


---

## 10. Riskler ve Önlemler

| Risk | Etki | Önlem |
|---|---|---|
| Tek kamera fiziksel sınırı | Çizgi kararı hassasiyeti | Olasılıksal karar + karar vermeme + F7 çok kamera/yüksek fps |
| Lisanslar (AGPL, lisanssız veri/ağırlık) | Ticari kullanım engeli | Değiştirilebilir model arayüzü; "yalnızca araştırma" bayrağı; kendi verimizle yeniden eğitim yolu |
| Yayın görüntüsü telif | Hukuki | Senin görüntülerin + CC klipler; yerel kullanım, dağıtım yok |
| Alan kayması (toprak/çim, amatör açı) | Doğruluk düşer | Alan bazında ayrı metrik; ince ayar; kalibrasyon kapısı |
| Kamera kesmesi/zoom/replay | Yanlış ölçüm | Kesme tespiti + geçerlilik kapısı; "ölçülemez" etiketleri |
| M4 16 GB bellek/hız | Yavaş işleme | Toplu işleme, oyuncu tespitinde kare atlama + takip, ONNX/CoreML |
| Disk (17 GB boş) | İndirme/eğitim kısıtı | Alt küme indirme, temizlik, harici disk |
| Gerçeklik verisi kıtlığı | Ölçüm zayıf | Simülatör + yayın itiraz grafikleri + İnceleme Odası düzeltmeleri |
| Kapsam şişmesi | Hiçbir şeyin bitmemesi | Faz kapıları; MVP önce |

---

## 11. Açık Kararlar (onayınla netleşecek)
1. **Hedef video türü:** yayın (TV) / kendi çekimlerin / ikisi.
2. **Kullanım amacı → lisans:** kişisel/araştırma/açık kaynak (YOLO26 AGPL sorun değil) mı, ticari kapalı ürün mü (Apache yığını)?
3. **Mevcut sentetik kod:** `legacy/` altına arşiv (önerilen) / olduğu gibi bırak.
4. **Etiketleme katkısı:** yaklaşık 150–300 sekmeyi İnceleme Odası'nda doğrulamaya zaman ayırabilir misin? (Gerçek doğruluk ölçümü için en değerli girdi.)

---

## Ek: Başlıca Kaynaklar
- Referans gönderi: https://www.linkedin.com/feed/update/urn:li:activity:7515821228201336832/
- ITF ELC Değerlendirme Prosedürü v2.8.1 (Ocak 2026): https://www.itftennis.com/media/15990/elc-evaluation-procedures-v281-jan-2026.pdf
- WASB (BMVC 2023): https://arxiv.org/abs/2311.05237 · https://github.com/nttcom/WASB-SBDT
- TennisCourtDetector: https://github.com/yastrebksv/TennisCourtDetector
- Where Is The Ball (CVPRW 2025): https://arxiv.org/abs/2506.05763 · TT3D: https://arxiv.org/abs/2504.10035
- TOTNet: https://arxiv.org/abs/2508.09650 · TrackNetV4: https://arxiv.org/abs/2409.14543
- F3Set (ICLR 2025): https://arxiv.org/abs/2504.08222 · Santra vd. (CVPR 2025): https://arxiv.org/abs/2503.00147
- Sandholtz vd. 2026 servis hedefleme: https://arxiv.org/abs/2607.28500 · CalTennis: https://arxiv.org/abs/2606.20542
- Collins & Evans 2008 (Hawk-Eye belirsizliği): https://doi.org/10.1177/0963662508093370 · Mather 2008: https://pmc.ncbi.nlm.nih.gov/articles/PMC2602819/
- Rod Cross, Hawk-Eye fiziği: https://www.physics.sydney.edu.au/~cross/HAWKEYE.htm
- Wimbledon 2025 ELC olayları: https://www.skysports.com/tennis/news/32498/13393602/ · https://www.enca.com/sport/wimbledon-changes-line-calling-system-after-embarrassing-blunder
- FIFA SAOT: https://inside.fifa.com/innovation/world-cup-2022/semi-automated-offside-technology
- HarshTomar1234/Tennis-Vision: https://github.com/HarshTomar1234/Tennis-Vision · abdullahtarek/tennis_analysis: https://github.com/abdullahtarek/tennis_analysis
- RF-DETR: https://github.com/roboflow/rf-detr · trackers: https://github.com/roboflow/trackers · rtmlib: https://github.com/Tau-J/rtmlib
- Ultralytics lisansı: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Wikimedia Davis Cup klipleri (CC BY-SA 4.0): https://commons.wikimedia.org/wiki/File:2018_Davis_Cup_Americas_Zone_-_Uruguay_vs_Mexico_-_01.webm

import os
import pandas as pd
import librosa
from tqdm import tqdm

# --- YOLLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "processed_labels.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")

def clean_and_verify_data():
    print("="*55)
    print(" VERİ TEMİZLİĞİ VE DOĞRULAMA OPERASYONU ")
    print("="*55)

    if not os.path.exists(CSV_PATH):
        print(f"[HATA] CSV dosyası bulunamadı: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    initial_count = len(df)
    to_drop = []

    print(f"Toplam {initial_count} kayıt taranıyor...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Dosya adını oluştur (label_processor'daki mantıkla aynı)
        ep_id = str(row['EpId']).zfill(3) if "fluencybank" in str(row['Show']).lower() else str(row['EpId'])
        filename = f"{row['Show']}_{ep_id}_{row['ClipId']}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)

        # 1. Dosya Fiziksel Olarak Var mı?
        if not os.path.exists(filepath):
            to_drop.append(idx)
            continue

        # 2. Dosya Boyutu 0 mı? (Fiziksel Boşluk)
        if os.path.getsize(filepath) == 0:
            print(f"\n[SİLİNİYOR] Boş dosya (0 byte): {filename}")
            os.remove(filepath) # Fiziksel olarak sil
            to_drop.append(idx)
            continue

        # 3. İçerik Analizi (Sinyal Var mı?)
        try:
            # Sadece çok kısa bir kısmını yükleyerek kontrol et (Hız için)
            y, sr = librosa.load(filepath, sr=16000, duration=0.1)
            
            # Eğer ses 0.1 saniyeden kısaysa veya tamamen sessizse (tümü 0 ise)
            if len(y) < 160 or (y == 0).all():
                print(f"\n[SİLİNİYOR] Sessiz veya çok kısa: {filename}")
                if os.path.exists(filepath): os.remove(filepath)
                to_drop.append(idx)
        except:
            # Dosya bozuksa ve okunmuyorsa
            print(f"\n[SİLİNİYOR] Bozuk dosya: {filename}")
            if os.path.exists(filepath): os.remove(filepath)
            to_drop.append(idx)

    # Veri setinden hatalıları çıkar
    df_clean = df.drop(to_drop).reset_index(drop=True)
    
    # 4. Kaydet
    df_clean.to_csv(CSV_PATH, index=False)

    print("\n" + "="*55)
    print(f"TEMİZLİK TAMAMLANDI")
    print(f"İlk Kayıt Sayısı    : {initial_count}")
    print(f"Silinen Kayıt Sayısı: {len(to_drop)}")
    print(f"Kalan Temiz Kayıt   : {len(df_clean)}")
    print("="*55)

if __name__ == "__main__":
    clean_and_verify_data()
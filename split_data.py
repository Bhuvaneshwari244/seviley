import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "processed_labels.csv")
SEED = 42

def deterministic_group_split():
    print("="*55)
    print(" SIZINTISIZ VERİ BÖLME (MUTLAK KİŞİ İZOLASYONU) ")
    print("="*55)

    df = pd.read_csv(CSV_PATH)
    
    # 1. KİŞİ (SPEAKER) KİMLİĞİ OLUŞTURMA
    # İstatistiksel Kural: FluencyBank'te her EpId farklı bir insandır. 
    # Podcast serilerinde ise (HeStutters vb.) tüm şov aynı insana aittir.
    def get_speaker_id(row):
        show = str(row['Show']).strip()
        epid = str(row['EpId']).strip()
        
        if "fluency" in show.lower():
            return f"FluencyBank_Speaker_{epid}"
        else:
            return f"Podcast_Speaker_{show}"
            
    df['Speaker_ID'] = df.apply(get_speaker_id, axis=1)
    
    groups = df['Speaker_ID'].values
    y = df['is_stutter'].values

    # 2. ÖNCE TEST SETİNİ AYIR (%15)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss_test.split(df, y, groups))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # 3. KALANI TRAIN VE VAL OLARAK BÖL
    # Tüm verinin %15'i val olacak şekilde (0.15 / 0.85 = ~0.176)
    val_ratio = 0.15 / 0.85
    groups_train_val = df_train_val['Speaker_ID'].values
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    train_idx, val_idx = next(gss_val.split(df_train_val, df_train_val['is_stutter'], groups_train_val))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # 4. MUTLAK SIZINTI KONTROLÜ (KANIT)
    train_spk = set(df_train['Speaker_ID'])
    val_spk = set(df_val['Speaker_ID'])
    test_spk = set(df_test['Speaker_ID'])
    
    print("\n─── Konuşmacı Sızıntısı Raporu ───")
    print(f"Train ∩ Val   : {len(train_spk & val_spk)} ortak kişi")
    print(f"Train ∩ Test  : {len(train_spk & test_spk)} ortak kişi")
    print(f"Val ∩ Test    : {len(val_spk & test_spk)} ortak kişi")
    
    # Sızıntı varsa kırmızı alarm
    if len(train_spk & val_spk) > 0 or len(train_spk & test_spk) > 0:
        print("\n[KRİTİK HATA] İzolasyon başarısız oldu. Kodu durdurun.")
        return
    else:
        print("-> İzolasyon BAŞARILI. Sızıntı SIFIR.")
    
    # 5. DOSYALARI KAYDET
    # Kaydederken Speaker_ID sütununu çıkarıyoruz (Wav2Vec ve XGBoost'a sadece is_stutter lazım)
    df_train.drop(columns=['Speaker_ID']).to_csv("train_split.csv", index=False)
    df_val.drop(columns=['Speaker_ID']).to_csv("val_split.csv", index=False)
    df_test.drop(columns=['Speaker_ID']).to_csv("test_split.csv", index=False)
    
    print("\n─── Dağılım Özeti ───")
    print(f"Train : {len(df_train):>5} klip | Kişi: {len(train_spk):>3} | Kekeme Oranı: %{round(df_train['is_stutter'].mean()*100, 1)}")
    print(f"Val   : {len(df_val):>5} klip | Kişi: {len(val_spk):>3} | Kekeme Oranı: %{round(df_val['is_stutter'].mean()*100, 1)}")
    print(f"Test  : {len(df_test):>5} klip | Kişi: {len(test_spk):>3} | Kekeme Oranı: %{round(df_test['is_stutter'].mean()*100, 1)}")
    print("="*55)

if __name__ == "__main__":
    deterministic_group_split()
import pandas as pd
import numpy as np
import os

def create_unified_label(row, dataset="SEP-28k"):
    """
    Hem SEP-28k hem de FluencyBank için rater-based (çoğunluk oylaması) mantığı uygular.
    Eğer gelen satırda doğrudan ikili etiket varsa korur, yoksa oylama hesaplar.
    """
    stutter_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep']

    # -------------------------
    # FLUENCYBANK DIRECT LABEL CHECK
    # -------------------------
    # Eğer oylama sütunları yoksa ve doğrudan hazır etiket varsa onu kullanır
    if "fluency" in dataset.lower() or "fluency" in str(row.get("Show", "")).lower():
        # Eğer çoklu etiket sütunları yoksa hazır binary etiketleri kontrol et
        if not any(c in row for c in stutter_cols):
            if "label" in row:
                return int(row["label"])
            if "is_stutter" in row:
                return int(row["is_stutter"])
            return 0  # fallback

    # -------------------------
    # MAJORITY VOTING (Hem SEP-28k hem FluencyBank oylama sütunları için)
    # -------------------------
    total_stutter = sum(row[c] for c in stutter_cols if c in row and pd.notna(row[c]))
    total_raters = sum(row[c] for c in stutter_cols + ['NoStutteredWords'] if c in row and pd.notna(row[c]))

    if total_raters == 0:
        return 0

    ratio = total_stutter / total_raters

    return 1 if ratio >= 0.5 else 0


def process_labels_v2():
    sep_df = pd.read_csv("SEP-28k_labels.csv")
    flu_df = pd.read_csv("FluencyBank_labels.csv")

    # --- FluencyBank formatını SEP-28k ile uyumlu yap ---
    flu_df = flu_df.rename(columns={
        "show": "Show",
        "ep_id": "EpId",
        "clip_id": "ClipId",
        "start": "Start",
        "stop": "Stop"
    })

    # EPID padding
    flu_df["EpId"] = flu_df["EpId"].apply(lambda x: str(x).zfill(3))

    # Veri setlerinin kökenini ayırt edebilmek için geçici bir kaynak sütunu ekliyoruz
    sep_df["dataset_origin"] = "SEP-28k"
    flu_df["dataset_origin"] = "FluencyBank"

    # label uyumu (varsayım: varsa)
    if "is_stutter" in flu_df.columns:
        flu_df["is_stutter"] = flu_df["is_stutter"].astype(int)

    # --- BİRLEŞTİR ---
    df = pd.concat([sep_df, flu_df], ignore_index=True)
    
    # 1. Tanımlamalar
    stutter_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep']

    # 2. Hesaplamalar
    # Sütunların varlığını kontrol ederek güvenli toplama yapıyoruz (FluencyBank'te eksik sütun olma ihtimaline karşı)
    existing_stutter_cols = [c for c in stutter_cols if c in df.columns]
    df['total_stutter_votes'] = df[existing_stutter_cols].sum(axis=1)
    
    # En sağlıklı rater sayısı tahmini
    rater_counting_cols = ['NoStutteredWords'] + stutter_cols
    existing_rater_cols = [c for c in rater_counting_cols if c in df.columns]
    df['calculated_raters'] = df[existing_rater_cols].sum(axis=1)
    
    # 3. Oransal Eşikleme (Ratio)
    df['stutter_ratio'] = df['total_stutter_votes'] / df['calculated_raters']
    
    # 4. Veri Kalitesi Filtresi
    # Rater oylama sütunları içeren (calculated_raters >= 3) satırları VEYA 
    # oylama sütunu içermeyip doğrudan hazır binary etiketi olan FluencyBank satırlarını koru
    if "is_stutter" in df.columns or "label" in df.columns:
        df_clean = df[(df['calculated_raters'] >= 3) | (df['calculated_raters'] == 0)].copy()
    else:
        df_clean = df[df['calculated_raters'] >= 3].copy()
    
    # 5. Nihai Etiketleme (Decision)
    # HATA DÜZELTMESİ: Sabit "SEP-28k" göndermek yerine satırın gerçek kökenini gönderiyoruz.
    df_clean['is_stutter'] = df_clean.apply(
        lambda row: create_unified_label(row, dataset=row["dataset_origin"]),
        axis=1
    )

    # --- İSTATİSTİKSEL RAPOR ---
    total_raw = len(df)
    total_filtered = len(df_clean)
    stutter_count = df_clean['is_stutter'].sum()
    fluent_count = total_filtered - stutter_count

    print(f"\n" + "="*45)
    print(f" NİHAİ ETİKETLEME VE KALİTE RAPORU ")
    print(f"="*45)
    print(f"Ham Veri Sayısı          : {total_raw}")
    print(f"Filtrelenmiş (Rater>=3)  : {total_filtered} (Eleme: {total_raw - total_filtered})")
    print(f"Nihai Kekeme (Label 1)   : {stutter_count} (%{round(stutter_count/total_filtered*100, 1)})")
    print(f"Nihai Akıcı  (Label 0)   : {fluent_count} (%{round(fluent_count/total_filtered*100, 1)})")
    print("-" * 45)

    # 6. Kayıt (Eklenen geçici sütunu çıktıda temizliyoruz)
    output_cols = ['Show', 'EpId', 'ClipId', 'Start', 'Stop', 'is_stutter', 'stutter_ratio', 'calculated_raters']
    df_clean[output_cols].to_csv('processed_labels.csv', index=False)
    print(f"[BAŞARILI] Koordinatlar (Start/Stop) işlendi. Toplam: {len(df_clean)} klip.")

if __name__ == "__main__":
    process_labels_v2()
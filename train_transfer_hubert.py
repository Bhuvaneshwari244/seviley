import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2FeatureExtractor,  # HuBERT da girişleri işlemek için Wav2Vec2 feature extractor kullanır
    HubertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. AYARLAR VE YOL TANIMLAMALARI
# ==========================================
# Get paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Audio clips folder
AUDIO_DIR = os.path.join(BASE_DIR, "processed_clips")

# CSV files
TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV   = os.path.join(BASE_DIR, "val_split.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test_split.csv")

# HuBERT Temel Modeli (Kekemelik için base model oldukça yeterlidir)
MODEL_NAME = "facebook/hubert-base-ls960"
BATCH_SIZE = 8
EPOCHS = 5

# ==========================================
# 2. VERİ ÖN İŞLEME VE METRİKLER
# ==========================================
# HuBERT modeli de ham ses sinyallerini 16kHz frekansında kabul eder
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def load_local_dataset(csv_path):
    """ 
    Senin CSV sütun yapına (Show, EpId, ClipId, is_stutter) 
    %100 uyumlu akıllı veri yükleyici.
    """
    df = pd.read_csv(csv_path)
    
    file_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        show = str(row['Show']).strip()
        ep_id = str(row['EpId']).strip()
        clip_id = str(row['ClipId']).strip()
        
        # FluencyBank şovunda klasördeki eşleşme için EpId'yi 3 haneli (000) yapıyoruz
        if "fluencybank" in show.lower():
            save_ep_id = ep_id.zfill(3)
        else:
            save_ep_id = ep_id
            
        # label_processor ve temizlik kodundaki standart .wav dosya adını oluşturuyoruz
        filename = f"{show}_{save_ep_id}_{clip_id}.wav"
        full_path = os.path.join(AUDIO_DIR, filename)
        
        # Sadece bilgisayarında fiziksel olarak var olan sesleri eğitime dahil et (Sağlamlık kontrolü)
        if os.path.exists(full_path):
            file_paths.append(full_path)
            labels.append(int(row['is_stutter']))  # Etiket sütunun: is_stutter

    print(f"📈 {os.path.basename(csv_path)} içinden {len(file_paths)} geçerli ses dosyası başarıyla yüklendi.")
    
    # HuggingFace Dataset formatına dönüştürme
    data_dict = {
        "file": file_paths,
        "label": labels
    }
    return Dataset.from_dict(data_dict)

def preprocess_audio_data(examples):
    audio_inputs = []
    for path in examples["file"]:
        try:
            speech, sr = sf.read(path)
            if len(speech.shape) > 1:
                speech = speech[:, 0]  # Stereo ise Mono'ya çevir
        except:
            speech = np.zeros(16000 * 3) # Bozuk dosya toleransı
            
        audio_inputs.append(speech)
        
    # Sesleri 3 saniyeye (16000 * 3) eşitle, eksikse doldur (padding), uzunsa kes (truncation)
    inputs = feature_extractor(
        audio_inputs, 
        sampling_rate=16000, 
        max_length=int(16000 * 3.0), 
        truncation=True, 
        padding="max_length",
        return_attention_mask=True

    )
    inputs["label"] = examples["label"]
    return inputs

# Başarı ölçüm metrikleri
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = metric_accuracy.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="binary")
    return {**acc, **f1}

# ==========================================
# 3. TRANSFER LEARNING VE EĞİTİM
# ==========================================
def main():
    print("🔄 Veri setleri HuBERT için yükleniyor...")
    raw_datasets = DatasetDict({
        "train": load_local_dataset(TRAIN_CSV),
        "validation": load_local_dataset(VAL_CSV),
        "test": load_local_dataset(TEST_CSV)
    })

    print("📊 Ses dalgaları HuBERT formatına dönüştürülüyor...")
    encoded_datasets = raw_datasets.map(
        preprocess_audio_data, 
        batched=True, 
        batch_size=16,
        remove_columns=["file"]
    )

    print("🏗️ Önceden eğitilmiş HuBERT modeli sınıflandırma katmanıyla kuruluyor...")
    # 2 Sınıf: 0 (Normal), 1 (Kekemelik)
    model = HubertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🚀 Eğitim donanımı: {device.upper()}")

    # TRANSFER LEARNING İÇİN KRİTİK ADIM: 
    # Özellik çıkaran alt katmanları dondur, sadece üst sınıflama katmanını eğit.
    model.freeze_feature_encoder()
    # Son 2 transformer katmanını tekrar eğitime aç
    for param in model.hubert.encoder.layers[-4:].parameters():
        param.requires_grad = True
    print("HuBERT son 2 encoder katmanı fine-tuning için açıldı.")

    # Eğitim argümanları
    # Eğitim argümanları (Güncel Transformers sürümleriyle uyumlu)
    training_args = TrainingArguments(
        output_dir="./hubert-stuttering-model",
        eval_strategy="epoch",       # 'evaluation_strategy' yerine güncel olan 'eval_strategy' yazıldı
        save_strategy="epoch",
        learning_rate=1e-5,  
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=5,          # CPU'da çok uzun sürmemesi için başlangıçta 1 epoch deneyelim
        weight_decay=0.01,
        logging_steps=50,            # Her 50 adımda bir ekrana ilerleme durumunu bassın
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,                  # CPU'da çalışırken fp16 (yarım duyarlılık) False olmalıdır
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    print("\n🏋️ HuBERT Fine-Tuning Başlıyor...")
    trainer.train()

    print("\n🧪 Test Seti Üzerinde Son Değerlendirme...")
    test_results = trainer.evaluate(encoded_datasets["test"])
    print(f"\n🎯 HUBERT MODELİ TEST SONUÇLARI:")
    print(f"Test Doğruluğu (Accuracy): {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Skoru           : {test_results['eval_f1']:.4f}")

    # Modeli diske kaydetme
    trainer.save_model("./best_hubert_stuttering_model")
    feature_extractor.save_pretrained("./best_hubert_stuttering_model")
    print("💾 Model 'best_hubert_stuttering_model' klasörüne başarıyla kaydedildi!")

if __name__ == "__main__":
    main()

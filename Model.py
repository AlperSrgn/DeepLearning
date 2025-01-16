import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pickle

# Veri setini yükleyin
file_path = 'train.csv'  # Yüklediğiniz dosyanın yolu
train_df = pd.read_csv(file_path)

# Eksik id veya comment_text içeren satırları temizle
train_df = train_df.dropna(subset=['id', 'comment_text'])

# Etiket sütunlarını bir liste olarak hazırlayın
label_columns = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_df['label'] = train_df[label_columns].values.tolist()

# Sadece tek bir sınıf için etiketle
train_df['label'] = train_df['label'].apply(lambda x: x.index(1) if 1 in x else -1)
train_df = train_df[train_df['label'] != -1]  # Geçersiz etiketleri çıkar

# Eğitim ve doğrulama veri kümelerine bölün
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Pandas veri çerçevelerini Hugging Face Dataset formatına dönüştürün
train_dataset = Dataset.from_pandas(train_data[['id', 'comment_text', 'label']])
val_dataset = Dataset.from_pandas(val_data[['id', 'comment_text', 'label']])

# Tokenizer'ı yükleyin
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenizasyon fonksiyonu
def tokenize_function(examples):
    return tokenizer(examples['comment_text'], padding="max_length", truncation=True)

# Eğitim ve doğrulama veri kümelerini tokenleştirme
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Modeli yükleyin
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_columns))


# Özel bir accuracy hesaplama fonksiyonu
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc}


# Eğitim argümanları
training_args = TrainingArguments(
    num_train_epochs=50,
    output_dir='./results',  # Model çıktı dosyalarının saklanacağı yer
    learning_rate=3e-5,      #3e-5
    per_device_train_batch_size=8,  # 8
    per_device_eval_batch_size=8,  # 8
    evaluation_strategy='epoch',  # Her epoch sonunda doğrulama
    logging_dir='./logs',  # Logların saklanacağı yer
    save_strategy='epoch',  # Her epoch sonunda model kaydet
    load_best_model_at_end=True  # En iyi doğrulama performansına sahip modeli yükle
)

# Trainer tanımlama
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Eğitim boyunca metrikleri tutacak bir liste
epoch_metrics = []

for epoch in range(int(training_args.num_train_epochs)):
    print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")

    # Epoch başlama zamanını kaydet
    epoch_start_time = time.time()

    # Eğitim adımı (train_loss ve train_accuracy hesaplama)
    train_output = trainer.train(resume_from_checkpoint=False)

    # Eğitim tahminlerini almak için predict() fonksiyonunu kullan
    train_preds_output = trainer.predict(tokenized_train_dataset)
    train_preds = np.argmax(train_preds_output.predictions, axis=1)  # Predicted class labels
    train_labels = train_preds_output.label_ids  # Gerçek etiketler
    train_acc = accuracy_score(train_labels, train_preds)

    # Değerlendirme adımı (val_loss ve val_accuracy hesaplama)
    metrics = trainer.evaluate()

    # Doğrulama tahminlerini almak için predict() fonksiyonunu kullan
    val_preds_output = trainer.predict(tokenized_val_dataset)
    val_preds = val_preds_output.predictions  # Predicted class probabilities
    val_labels = val_preds_output.label_ids  # Gerçek etiketler

    # Epoch bitiş zamanını kaydet
    epoch_end_time = time.time()
    epoch_training_time = epoch_end_time - epoch_start_time  # Epoch eğitim süresi

    # Çıkarım süresi (inference time) hesaplama
    inference_start_time = time.time()

    # Modelin eğitim ve doğrulama veri setleri üzerinde tahmin yapma sürelerini ölç
    _ = trainer.predict(tokenized_train_dataset)
    _ = trainer.predict(tokenized_val_dataset)

    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time  # Çıkarım süresi

    # Epoch verisini kaydetme
    epoch_data = {
        'epoch': epoch + 1,
        'train_loss': train_output.training_loss,
        'train_accuracy': train_acc,  # Eğitim doğruluğunu ekle
        'val_loss': metrics['eval_loss'],
        'val_accuracy': metrics['eval_accuracy'],
        'val_preds': val_preds.tolist(),  # Değerlendirme tahminlerini ekle
        'val_labels': val_labels.tolist(),  # Gerçek etiketleri ekle
        'epoch_training_time': epoch_training_time,  # Eğitim süresi
        'total_inference_time': total_inference_time  # Çıkarım süresi
    }

    epoch_metrics.append(epoch_data)

# Dosya mevcut mu kontrol et
file_path = 'roberta_training_metrics.pkl'

if os.path.exists(file_path):
    # Dosya varsa, mevcut verileri okuyun ve yeni verilerle birleştirin
    with open(file_path, 'rb') as f:
        existing_data = pickle.load(f)
    existing_data.extend(epoch_metrics)  # Yeni metrikleri mevcut verilere ekle

    # Güncellenmiş verileri tekrar dosyaya yaz
    with open(file_path, 'wb') as f:
        pickle.dump(existing_data, f)
    print("Dosya var, yeni metrikler mevcut verilere eklendi.")
else:
    # Dosya yoksa, yeni dosya oluştur ve verileri kaydet
    with open(file_path, 'wb') as f:
        pickle.dump(epoch_metrics, f)
    print("Dosya oluşturuldu ve metrikler kaydedildi.")

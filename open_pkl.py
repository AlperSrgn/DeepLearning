import pickle
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize


# .pkl dosyasının yolu
file_path = 'roberta_training_metrics.pkl'  # Dosya yolu

# Dosyayı yükle
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# İçeriği yazdır
print(data)


# Loss grafik çizim fonksiyonu
def plot_losses_from_pkl(pkl_file_path, output_image_path='roberta_loss_plot.png'):
    # 1. pkl dosyasını yükle
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)

    # 2. Epoch, train_loss ve val_loss değerlerini çıkar
    epochs = [entry['epoch'] for entry in data]
    train_losses = [entry['train_loss'] for entry in data]
    val_losses = [entry['val_loss'] for entry in data]

    # 3. Grafik oluşturma
    plt.figure(figsize=(8, 6))

    # Train loss ve val loss için çizgi grafiği
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o', color='orange')

    # Başlık ve etiketler
    plt.title('Train Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 4. Grafik kaydetme
    plt.savefig(output_image_path)
    print(f"Grafik başarıyla kaydedildi: {output_image_path}")


# Acc grafik çizim fonksiyonu
def plot_accuracy_from_pkl(pkl_file_path, output_image_path='roberta_accuracy_plot.png'):
    # 1. pkl dosyasını yükle
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)

    # 2. Epoch, train_accuracy ve val_accuracy değerlerini çıkar
    epochs = [entry['epoch'] for entry in data]
    train_accs = [entry['train_accuracy'] for entry in data]
    val_accs = [entry['val_accuracy'] for entry in data]

    # 3. Grafik oluşturma
    plt.figure(figsize=(8, 6))

    # Train accuracy ve val accuracy için çizgi grafiği
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o', color='blue')
    plt.plot(epochs, val_accs, label='Val Accuracy', marker='o', color='orange')

    # Başlık ve etiketler
    plt.title('Train Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 4. Grafik kaydetme
    plt.savefig(output_image_path)
    print(f"Grafik başarıyla kaydedildi: {output_image_path}")


# ROC eğrisi fonksiyonu
def plot_roc_curve_from_pkl(pkl_file_path, output_image_path='roberta_roc_curve.png'):
    # 1. pkl dosyasını yükle
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)

    # 2. Değerlendirme tahminlerini ve etiketleri çıkar
    val_preds = []  # Modelin tahmin ettiği olasılıkları depolamak için
    val_labels = []  # Gerçek etiketler

    for entry in data:
        val_preds.append(entry['val_preds'])
        val_labels.append(entry['val_labels'])

    # 3. ROC eğrisini çizmek için gerekli işlemler
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # 4. Çok sınıflı etiketleri ikili (binary) formatına çevirme
    # Burada label_binarize fonksiyonunu kullanarak etiketleri ikili hale getiriyoruz
    val_labels_bin = label_binarize(val_labels, classes=np.unique(val_labels))

    # 5. Sınıf isimlerini almak
    # Eğer sınıf isimleri dosyada bir yerde yer alıyorsa, onları almak için veri setini kontrol etmeliyiz.
    class_names = np.unique(val_labels)  # Sınıf isimlerini alıyoruz

    # **Opsiyonel:** Eğer veri setinde 'class_names' gibi bir anahtar varsa
    # ve sınıf isimlerini almak istiyorsanız şu şekilde de ekleyebilirsiniz:
    # class_names = data.get('class_names', np.unique(val_labels))

    # 6. ROC eğrisini çizmek için her sınıfı tek tek işleme
    n_classes = val_labels_bin.shape[1]  # Sınıf sayısını öğren

    plt.figure(figsize=(8, 6))

    # ROC eğrisini her sınıf için çizme
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(val_labels_bin[:, i], val_preds[:, i])  # Her sınıf için ROC hesaplama
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')  # class_names ile sınıf ismini ekle

    # Diagonal çizgiyi (rasgele tahminleri temsil eder) ekleyelim
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # 7. Grafik başlıkları ve etiketler
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # 8. Grafik kaydetme
    plt.savefig(output_image_path)
    print(f"ROC eğrisi başarıyla kaydedildi: {output_image_path}")


# Zaman grafiği fonksiyonu
def plot_training_inference_time(pkl_file_path, output_image_path='roberta_training_inference_time.png'):
    # .pkl dosyasını yükleyelim
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    # Epoch numaralarını alalım
    epochs = [entry['epoch'] for entry in data]

    # Eğitim süresi ve çıkarım süresini alalım
    training_times = [entry['epoch_training_time'] for entry in data]
    inference_times = [entry['total_inference_time'] for entry in data]

    # Grafik oluşturma
    plt.figure(figsize=(10, 6))

    # Eğitim süresi ve çıkarım süresi için çizimler
    plt.plot(epochs, training_times, label='Training Time (s)', marker='o', color='blue')
    plt.plot(epochs, inference_times, label='Inference Time (s)', marker='o', color='orange')

    # Grafik başlıkları ve etiketler
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training and Inference Time per Epoch')
    plt.legend()

    # Grafiği kaydet
    plt.savefig(output_image_path)
    print(f"Grafik başarıyla kaydedildi: {output_image_path}")
    plt.close()



# Zaman grafiği
plot_training_inference_time('roberta_training_metrics.pkl')

# ROC eğrisi
plot_roc_curve_from_pkl('roberta_training_metrics.pkl')

# Acc grafik
plot_accuracy_from_pkl('roberta_training_metrics.pkl')

#Loss grafik
plot_losses_from_pkl('roberta_training_metrics.pkl')




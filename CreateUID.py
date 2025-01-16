import random
import string
import pandas as pd

def generate_dynamic_prefix_id(prefix_start=0000, length=16):

    # Prefix artacak sayıyı takip etmek için bir sayaç kullanıyoruz
    current_prefix = prefix_start

    # Rastgele hex karakterlerden oluşan bir kısım oluştur
    random_part = ''.join(random.choices(string.hexdigits.lower(), k=length - len(str(current_prefix))))

    # Prefix ve rastgele kısmı birleştir
    custom_id = f"{current_prefix}{random_part}"

    # Prefix değerini artır
    prefix_start += 1

    return custom_id, prefix_start


# 8800 adet ID oluştur
prefix_value = 0000
id_list = []

for _ in range(9642):
    custom_id, prefix_value = generate_dynamic_prefix_id(prefix_start=prefix_value, length=16)
    id_list.append(custom_id)

#****************** EXCEL DOSYASINA YAZDIR ****************************
file_path = "generated_ids.xlsx"  # Dosya adını ve yolunu belirleyin

# pandas DataFrame ile listeyi Excel dosyasına yazıyoruz
df = pd.DataFrame(id_list, columns=["Generated IDs"])

# Excel dosyasına yazma
df.to_excel(file_path, index=False)

print(f"{len(id_list)} adet ID başarıyla '{file_path}' dosyasına kaydedildi.")
#*********************************************************************

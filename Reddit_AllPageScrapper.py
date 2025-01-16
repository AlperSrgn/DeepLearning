import requests
import pandas as pd
import os


# Reddit API üzerinden ana sayfa (örneğin, popüler) gönderilerine erişim
url = "https://www.reddit.com/r/all/top/.json?limit=10"  # Popüler gönderiler (top) için URL
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Yorumların saklanacağı liste
comments = []

# Reddit API'den gönderi verilerini çekme
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()

    # Gönderiler üzerinde döngü
    for post in data['data']['children']:
        post_id = post['data']['id']
        post_title = post['data']['title']
        post_url = f"https://www.reddit.com/r/all/comments/{post_id}.json"

        # Yorumları almak için her gönderi için yorum URL'sine istek yapıyoruz
        post_response = requests.get(post_url, headers=headers)
        if post_response.status_code == 200:
            post_data = post_response.json()

            # Yorumları ekle
            for comment in post_data[1]['data']['children']:
                body = comment['data'].get('body', '').strip()  # Yorumun "body" kısmını al
                if body:  # Yorum boş değilse
                    comments.append(body)

    # Yorumları pandas DataFrame'e aktar
    new_df = pd.DataFrame(comments, columns=["comment"])

    # Klasör yolu
    folder_path = r"C:\Users\alper\OneDrive\Masaüstü\Comments"

    # Klasörün varlığını kontrol et ve oluştur
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Dosya yolu
    base_filename = "reddit_comments.xlsx"
    file_path = os.path.join(folder_path, base_filename)

    # Eğer dosya varsa, mevcut verileri oku ve yeni verileri ekle
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path, engine='openpyxl')
        # Yeni verileri var olan DataFrame'e ekle
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Eğer dosya yoksa, yeni DataFrame kullan
        combined_df = new_df

    # DataFrame'i Excel dosyasına kaydet
    combined_df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"{len(comments)} yorum başarıyla '{file_path}' dosyasına eklendi.")
else:
    print(f"Veri çekilemedi. Hata kodu: {response.status_code}")

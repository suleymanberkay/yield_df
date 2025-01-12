import serial
import csv
import time

# Seri portu başlat
ser = serial.Serial('COM8', 115200)  # 'COM8' yerine doğru portu yazın
time.sleep(2)  # Arduino'nun bağlanabilmesi için 2 saniye bekleyelim

# Sabit veriler (örneğin)
area = 'Turkey'
items = ['Maize', 'Potatoes', 'Rice', 'Paddy', 'Sorghum', 'Soybeans', 'Wheat']  # Ürünler listesi (Rice ve Paddy ayrı ürünler)
year = 2010
hg_ha_yield = [72614, 323303, 86899, 86899, 70968, 36869, 24400]  # Ürün verim değerleri
pesticides_tonnes = 38554.69  # Sabit pesticides değeri
average_rain_fall_mm_per_year = 593.0  # Sabit yağış miktarı (bu da değişebilir)

# CSV dosyasına yazma işlemi için dosya aç
with open('sensor_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'Area', 'Item', 'Year', 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])  # CSV başlıkları

    counter = 0  # Sayaç başlat (sıfırdan başlıyor)
    item_counter = 0  # İlk üründen başla (0: Maize, 1: Potatoes, vb.)

    while year <= 2015:
        # Seri porttan veri oku
        line = ser.readline().decode('utf-8').strip()  # Satırdaki veriyi oku
        print(f"Received: {line}")  # Seri porttan gelen ham veriyi yazdırarak kontrol et

        if "Temp:" in line and "Humidity:" in line:  # Veri "Temp:" ve "Humidity:" içeriyorsa işlem yap
            try:
                # Sıcaklık ve nem verilerini ayıralım
                temp_str = line.split('Temp:')[1].split('Humidity:')[0].strip()  # Temp değeri
                humidity_str = line.split('Humidity:')[1].strip()  # Humidity değeri

                # Nem yüzdesini al ve 100 ile çarp
                humidity_value = float(humidity_str.replace('%', '')) * 10

                # Sıcaklık değerindeki 'C' harfini kaldır
                temp_value = float(temp_str.replace(' C', '').strip())  # 'C'yi kaldırıyoruz

                # Şu anki ürün ve yıl bilgileri
                current_item = items[item_counter]  # Güncel ürün
                current_hg_ha_yield = hg_ha_yield[item_counter]  # Güncel verim
                current_year = year  # Şu anki yıl
                current_avg_rain_fall = average_rain_fall_mm_per_year  # Sabit yağış miktarı

                # Sabit veriler ve sensörden alınan verilerle CSV'ye yazdıralım
                writer.writerow([counter, area, current_item, current_year, current_hg_ha_yield, current_avg_rain_fall, pesticides_tonnes, temp_value])  # CSV dosyasına yaz
                print(f"Temp: {temp_value} C, Humidity: {humidity_value} %")  # Seri monitöre yazdır

                counter += 1  # Sayaç bir arttır

                # Ürünler bitince bir sonraki yıla geç
                item_counter += 1

                if item_counter == len(items):  # Eğer bütün ürünler tamamlandıysa
                    item_counter = 0  # Yeniden ilk ürüne dön

                    # Yıl güncelle
                    year += 1  # Yıl arttırılır

            except Exception as e:
                print(f"Error parsing data: {line}. Error: {e}")  # Hata mesajı
        time.sleep(2)  # 2 saniye bekle

print("Veri kaydı tamamlandı.")

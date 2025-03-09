import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_images_in_folder(folder_path):
    # Klasördeki tüm dosyaları al
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('bmp', 'jpg', 'jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"İşleniyor: {image_file}")
        
        # 1. Görüntüyü yükleme ve griye çevirme
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. CLAHE ile kontrast artırma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        
        # 3. Gürültü azaltma (Median Blur + Bilateral Filtering)
        blurred = cv2.medianBlur(img_clahe, 5)
        bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        # 4. Thresholding ile objeyi belirgin hale getirme
        _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Kenar algılama (Canny)
        edges_processed = cv2.Canny(thresh, 50, 150)
        
        # 6. Sonuçları gösterme
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1), plt.imshow(img_clahe, cmap='gray'), plt.title("CLAHE ile Kontrast Artırılmış")
        plt.subplot(1, 3, 2), plt.imshow(thresh, cmap='gray'), plt.title("Threshold Uygulanmış Görüntü")
        plt.subplot(1, 3, 3), plt.imshow(edges_processed, cmap='gray'), plt.title("Yeni Kenar Algılama Sonucu")
        plt.show()
        
        # 7. En yoğun konturu bulma
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = img[y:y+h, x:x+w]
            
            # 8. En yoğun konturu içeren bölümü gösterme
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.title("En Yoğun Kontur Bölgesi")
            plt.show()

# Örnek Kullanım
folder_path = "hatali-isik"  # Buraya klasör yolunu yazın
process_images_in_folder(folder_path)

import cv2
import numpy as np
import os

def select_golden_template(image_path):
    """ Kullanıcının Golden Template üzerinden bir bölge seçmesini sağlar. """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    roi = cv2.selectROI("Golden Template Seçimi", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

def normalized_cross_correlation(template, search_img):
    """ Normalized grayscale correlation ile template'in arama görüntüsündeki konumunu bulur. """
    result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc

def process_images_in_folder(folder_path, golden_template_path):
    """ Tüm resimlerde golden template üzerinde seçilen alanı arar ve eşleşmeleri gösterir. """
    # Kullanıcıdan golden template üzerinde bir alan seçmesini iste
    roi = select_golden_template(golden_template_path)
    
    # Golden template'in belirlenen bölgesini kesip griye çevir
    golden_img = cv2.imread(golden_template_path, cv2.IMREAD_COLOR)
    golden_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    template = golden_gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('bmp', 'jpg', 'jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"İşleniyor: {image_file}")
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE ile kontrast artırma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        
        # Gürültü azaltma
        blurred = cv2.medianBlur(img_clahe, 5)
        bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        # Thresholding
        _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # En büyük konturu bul ve kırp
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = img[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Golden template'teki bölgeyi bu görüntüde ara
            match_loc = normalized_cross_correlation(template, cropped_gray)
            
            # Kapağın bulunup bulunmadığını kontrol et
            if match_loc:
                cx, cy = match_loc
                found_x, found_y = x + cx, y + cy
                
                # Eşleşen bölgeyi işaretle
                result_img = img.copy()
                cv2.rectangle(result_img, (found_x, found_y), 
                              (found_x + roi[2], found_y + roi[3]), (0, 255, 0), 3)
                
                # Sonucu göster
                cv2.imshow("Kapak Tespit Edildi", result_img)
                print(f"Kapak bulundu: {image_file} - Konum: ({found_x}, {found_y})")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Kapak bulunamadı: {image_file}")
    
# Kullanım
folder_path = r"C:\Users\Hakan\Desktop\sleepy-last\hatali-isik"  # Klasör yolu
golden_template_path = r"C:\Users\Hakan\Desktop\sleepy-last\dogru-isik\Basler_acA4096-30uc__40279615__20250307_164844388_0000.bmp"  # Golden template için resim yolu
process_images_in_folder(folder_path, golden_template_path)

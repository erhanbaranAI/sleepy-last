import cv2
import numpy as np
import os

def select_golden_template_regions(template_folder, scale=0.5):
    """ Kullanıcının golden template üzerindeki bölgeleri seçmesini sağlar. """
    template_files = [f for f in os.listdir(template_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    all_rois = []
    
    for template_file in template_files:
        template_path = os.path.join(template_folder, template_file)
        img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        while True:
            print(f"Golden Template Seçimi - {template_file}. Seçimi tamamladıktan sonra ESC tuşuna basın.")
            rois = cv2.selectROIs(f"Golden Template: {template_file}", resized_img, showCrosshair=True)
            cv2.destroyAllWindows()
            
            if len(rois) > 0:
                all_rois.extend([(int(r[0] / scale), int(r[1] / scale), int(r[2] / scale), int(r[3] / scale)) for r in rois])
                break
            else:
                print("Hiç ROI seçilmedi! Lütfen en az bir bölge seçin.")
    
    return all_rois

def compute_average_roi(rois):
    """ Seçilen golden template bölgelerinin ortalamasını hesaplar. """
    if len(rois) == 0:
        raise ValueError("Golden template işaretlemeleri bulunamadı! Lütfen en az bir işaretleme yapın.")
    
    rois = np.array(rois)
    avg_x, avg_y, avg_w, avg_h = np.mean(rois, axis=0).astype(int)
    return avg_x, avg_y, avg_w, avg_h

def process_images_in_folder(folder_path, template_folder, scale=0.5):
    """ Golden template'ten öğrenilen konumu kullanarak yeni görüntülerde karşılaştırma yapar. """
    all_rois = select_golden_template_regions(template_folder, scale)
    if not all_rois:
        print("Golden template işaretlemeleri bulunamadı. İşlem iptal edildi.")
        return
    
    avg_roi = compute_average_roi(all_rois)
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"İşleniyor: {image_file}")
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing ekleme
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        blurred = cv2.medianBlur(img_clahe, 5)
        bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
        _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # En büyük konturu bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = img[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Preprocessing aynı şekilde uygulanıyor
            cropped_clahe = clahe.apply(cropped_gray)
            cropped_blurred = cv2.medianBlur(cropped_clahe, 5)
            cropped_bilateral = cv2.bilateralFilter(cropped_blurred, 9, 75, 75)
            
            # Template matching düzeltildi
            result = cv2.matchTemplate(img_gray, cropped_bilateral, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            if max_loc:
                cx, cy = max_loc
                found_x, found_y = cx, cy
                found_w, found_h = avg_roi[2], avg_roi[3]
                
                error_x = abs(avg_roi[0] - found_x) / avg_roi[2] * 100
                error_y = abs(avg_roi[1] - found_y) / avg_roi[3] * 100
                error_percentage = (error_x + error_y) / 2
                
                result_img = img.copy()
                
                # Kırmızı dikdörtgen: Olması gereken yer
                cv2.rectangle(result_img, (avg_roi[0], avg_roi[1]), 
                              (avg_roi[0] + avg_roi[2], avg_roi[1] + avg_roi[3]), (0, 0, 255), 3)
                
                # Yeşil dikdörtgen: Algılanan yer
                cv2.rectangle(result_img, (found_x, found_y), 
                              (found_x + found_w, found_y + found_h), (0, 255, 0), 3)
                
                h, w = result_img.shape[:2]
                resized_result = cv2.resize(result_img, (int(w * scale), int(h * scale)))
                cv2.imshow("Kapak Tespit Edildi", resized_result)
                print(f"Kapak bulundu: {image_file} - Konum: ({found_x}, {found_y})")
                print(f"Hata Yüzdesi: %{error_percentage:.2f}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Kapak bulunamadı: {image_file}")
    


folder_path = r"C:\Users\Hakan\Desktop\sleepy-last\hatali-isik"  # Test edilecek klasör yolu
template_folder = r"C:\Users\Hakan\Desktop\sleepy-last\dogru-isik"  # Golden template klasör yolu
process_images_in_folder(folder_path, template_folder, scale=0.5)

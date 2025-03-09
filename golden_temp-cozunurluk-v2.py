import cv2
import numpy as np
import os

def select_golden_template(image_path, scale=0.5):
    """ Kullanıcının Golden Template üzerinden bir bölge seçmesini sağlar. """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    roi = cv2.selectROI("Golden Template Seçimi", resized_img, fromCenter=False, showCrosshair=True)
    
    # Orijinal boyutlara geri döndür
    roi = (int(roi[0] / scale), int(roi[1] / scale), int(roi[2] / scale), int(roi[3] / scale))
    
    cv2.destroyAllWindows()
    return roi

def template_matching_fixed_scale(template, search_img):
    """ Template matching kullanarak en iyi konumu bulur (sabit ölçekte). """
    result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc

def process_images_in_folder(folder_path, golden_template_path, scale=0.5):
    """ Tüm resimlerde golden template üzerinde seçilen alanı arar ve eşleşmeleri gösterir. """
    roi = select_golden_template(golden_template_path, scale)
    
    golden_img = cv2.imread(golden_template_path, cv2.IMREAD_COLOR)
    golden_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    template = golden_gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('bmp', 'jpg', 'jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"İşleniyor: {image_file}")
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        
        blurred = cv2.medianBlur(img_clahe, 5)
        bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = img[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            match_loc = template_matching_fixed_scale(template, cropped_gray)
            
            if match_loc:
                cx, cy = match_loc
                found_x, found_y = x + cx, y + cy
                found_w, found_h = roi[2], roi[3]
                
                result_img = img.copy()
                cv2.rectangle(result_img, (found_x, found_y), 
                              (found_x + found_w, found_y + found_h), (0, 255, 0), 3)
                
                h, w = result_img.shape[:2]
                resized_result = cv2.resize(result_img, (int(w * scale), int(h * scale)))
                cv2.imshow("Kapak Tespit Edildi", resized_result)
                print(f"Kapak bulundu: {image_file} - Konum: ({found_x}, {found_y})")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Kapak bulunamadı: {image_file}")
    
folder_path = r"C:\Users\Hakan\Desktop\sleepy-last\hatali-isik"  # Klasör yolu
golden_template_path = r"C:\Users\Hakan\Desktop\sleepy-last\dogru-isik\Basler_acA4096-30uc__40279615__20250307_164844388_0000.bmp"  # Golden template için resim yolu
process_images_in_folder(folder_path, golden_template_path, scale=0.5)

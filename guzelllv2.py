import cv2
import numpy as np
import os

def process_images_in_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('bmp', 'jpg', 'jpeg'))]
    valid_selections = []
    
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
            
            crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, crop_thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_contours, _ = cv2.findContours(crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = cropped.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            crop_contours = sorted(crop_contours, key=lambda cnt: cv2.pointPolygonTest(cnt, (center_x, center_y), True), reverse=True)
            index = 0
            
            while index < len(crop_contours):
                cx, cy, cw, ch = cv2.boundingRect(crop_contours[index])
                area = cw * ch
                
                if area > 192.500:
                    display_img = cropped.copy()
                    cv2.rectangle(display_img, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 3)
                    
                    cv2.imshow("Kapak Seçimi", display_img)
                    print(f"Kısa Kenar: {ch}, Uzun Kenar: {cw}, Alan: {area}")
                    key = cv2.waitKey(0)
                    
                    if key == ord('s'):
                        index += 1
                    elif key == ord('a'):
                        valid_selections.append((cx, cy, cw, ch))
                        break
                else:
                    index += 1
            
            cv2.destroyAllWindows()
    
    print("Seçilen Kapağın Ölçümleri:")
    for selection in valid_selections:
        print(selection)

folder_path = "dogru-isik"
process_images_in_folder(folder_path)

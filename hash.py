import sensor, image, time, lcd
import gc
from maix import KPU
from maix import GPIO, utils
from fpioa_manager import fm
from board import board_info

from modules import ybserial
import time
import math

serial = ybserial()

lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 100)
clock = time.clock()

feature_img = image.Image(size=(64,64), copy_to_fb=False)
feature_img.pix_to_ai()

FACE_PIC_SIZE = 64
dst_point =[(int(38.2946 * FACE_PIC_SIZE / 112), int(51.6963 * FACE_PIC_SIZE / 112)),
            (int(73.5318 * FACE_PIC_SIZE / 112), int(51.5014 * FACE_PIC_SIZE / 112)),
            (int(56.0252 * FACE_PIC_SIZE / 112), int(71.7366 * FACE_PIC_SIZE / 112)),
            (int(41.5493 * FACE_PIC_SIZE / 112), int(92.3655 * FACE_PIC_SIZE / 112)),
            (int(70.7299 * FACE_PIC_SIZE / 112), int(92.2041 * FACE_PIC_SIZE / 112)) ]

anchor = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875, 0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875, 0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)
kpu = KPU()
kpu.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
kpu.init_yolo2(anchor, anchor_num=9, img_w=320, img_h=240, net_w=320 , net_h=240 ,layer_w=10 ,layer_h=8, threshold=0.7, nms_value=0.2, classes=1)

ld5_kpu = KPU()
print("ready load model")
ld5_kpu.load_kmodel("/sd/KPU/face_recognization/ld5.kmodel")

fea_kpu = KPU()
print("ready load model")
fea_kpu.load_kmodel("/sd/KPU/face_recognization/feature_extraction.kmodel")

start_processing = False
BOUNCE_PROTECTION = 50

fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
def set_key_state(*_):
    global start_processing
    start_processing = True
    time.sleep_ms(BOUNCE_PROTECTION)
key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)

# ============ Optimized Signature Scheme ============
face_signatures = []  # Store face signatures (mean, variance, trend_hash)

def create_face_signature(feature):
    """Create a more stable face signature"""
    # 1. Calculate mean and standard deviation
    total = 0
    for val in feature:
        total += val
    mean = total / len(feature)
    
    variance = 0
    for val in feature:
        diff = val - mean
        variance += diff * diff
    variance = variance / len(feature)
    std_dev = math.sqrt(variance) if variance > 0 else 0.001
    
    # 2. Create trend hash (based on relative magnitude of feature values)
    trend_hash = ""
    # Take the first 10 feature values, compare adjacent values
    for i in range(min(9, len(feature)-1)):
        if feature[i] > feature[i+1]:
            trend_hash += "1"  # Decreasing trend
        else:
            trend_hash += "0"  # Increasing trend
    
    return (mean, std_dev, trend_hash)

def compare_signatures(sig1, sig2):
    """Compare signatures, return a score from 0-100"""
    mean1, std1, hash1 = sig1
    mean2, std2, hash2 = sig2
    
    # 1. Mean similarity (40% weight)
    mean_diff = abs(mean1 - mean2)
    # Normalization: assume mean difference in range 0-0.5
    mean_similarity = max(0, 100 - (mean_diff * 200))
    
    # 2. Standard deviation similarity (30% weight)
    if std1 == 0 or std2 == 0:
        std_similarity = 0
    else:
        std_ratio = min(std1, std2) / max(std1, std2)
        std_similarity = std_ratio * 100
    
    # 3. Trend hash matching (30% weight)
    hash_length = min(len(hash1), len(hash2))
    if hash_length == 0:
        hash_similarity = 0
    else:
        matches = 0
        for i in range(hash_length):
            if hash1[i] == hash2[i]:
                matches += 1
        hash_similarity = (matches / hash_length) * 100
    
    # Combined score (weighted average)
    total_score = (mean_similarity * 0.4 + 
                   std_similarity * 0.3 + 
                   hash_similarity * 0.3)
    
    return total_score

THRESHOLD = 95.0  # Below 95 points considered unrecognized - This is the only modified part
recog_flag = False

def extend_box(x, y, w, h, scale):
    x1_t = x - scale*w
    x2_t = x + w + scale*w
    y1_t = y - scale*h
    y2_t = y + h + scale*h
    x1 = int(x1_t) if x1_t>1 else 1
    x2 = int(x2_t) if x2_t<320 else 319
    y1 = int(y1_t) if y1_t>1 else 1
    y2 = int(y2_t) if y2_t<240 else 239
    cut_img_w = x2-x1+1
    cut_img_h = y2-y1+1
    return x1, y1, cut_img_w, cut_img_h

msg_=""
while True:
    gc.collect()
    clock.tick()
    img = sensor.snapshot()
    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2()
    fps = clock.fps()
    
    if len(dect) > 0:
        for l in dect :
            x1, y1, cut_img_w, cut_img_h= extend_box(l[0], l[1], l[2], l[3], scale=0)
            face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
            face_cut_128 = face_cut.resize(128, 128)
            face_cut_128.pix_to_ai()
            out = ld5_kpu.run_with_output(face_cut_128, getlist=True)
            face_key_point = []
            for j in range(5):
                x = int(KPU.sigmoid(out[2 * j])*cut_img_w + x1)
                y = int(KPU.sigmoid(out[2 * j + 1])*cut_img_h + y1)
                face_key_point.append((x,y))
            T = image.get_affine_transform(face_key_point, dst_point)
            image.warp_affine_ai(img, feature_img, T)
            feature = fea_kpu.run_with_output(feature_img, get_feature = True)
            del face_key_point
            
            # ============ Face Recognition Logic ============
            # Create signature for current face
            current_sig = create_face_signature(feature)
            
            # Find best match
            best_score = 0
            best_id = -1
            
            for i, saved_sig in enumerate(face_signatures):
                score = compare_signatures(current_sig, saved_sig)
                if score > best_score:
                    best_score = score
                    best_id = i
            
            # Strict threshold: do not recognize if below 95 points
            if best_score >= THRESHOLD and best_id != -1:
                # Recognition successful
                img.draw_string(0, 195, "ID:%d score:%2.1f" % (best_id, best_score), 
                                color=(0, 255, 0), scale=2)
                recog_flag = True
                index = best_id
                box_color = (0, 255, 0)  # Green box
                msg_ = "Y%02d" % best_id
            else:
                # Recognition failed (insufficient score or not registered)
                if best_score > 0:
                    status_text = "low score:%2.1f" % best_score
                else:
                    status_text = "unregistered"
                
                img.draw_string(0, 195, status_text, color=(255, 0, 0), scale=2)
                recog_flag = False
                box_color = (255, 255, 255)  # White box
                msg_ = "N"
            
            # Draw face bounding box
            img.draw_rectangle(l[0], l[1], l[2], l[3], color=box_color)
            
            # Button press to register new face
            if start_processing:
                # Check if too similar to existing signatures (avoid duplicate registration)
                is_duplicate = False
                for saved_sig in face_signatures:
                    if compare_signatures(current_sig, saved_sig) > 90:
                        is_duplicate = True
                        print("Face already registered (similarity > 90%)")
                        break
                
                if not is_duplicate and len(face_signatures) < 10:
                    face_signatures.append(current_sig)
                    new_id = len(face_signatures) - 1
                    print("Registered new face ID:%d" % new_id)
                    print("Signature: mean=%.3f, std=%.3f" % 
                          (current_sig[0], current_sig[1]))
                    # Temporarily display registration success
                    img.draw_string(0, 175, "Registered ID:%d" % new_id, 
                                    color=(0, 255, 255), scale=2)
                
                start_processing = False
            
            del (face_cut_128)
            del (face_cut)
    
    if len(dect) > 0:
        send_data = "$08" + msg_ + ",#"
        time.sleep_ms(5)
        serial.send(send_data)
    else:
        serial.send("#")

    # Display information
    img.draw_string(0, 0, "FPS:%2.1f" % fps, color=(0, 60, 255), scale=2.0)
    img.draw_string(0, 215, "BOOT: Register", color=(255, 100, 0), scale=2.0)
    img.draw_string(250, 0, "F:%d" % len(face_signatures), color=(255, 255, 0), scale=2)
    
    # Display current threshold
    img.draw_string(150, 0, "T:%d" % int(THRESHOLD), color=(255, 100, 255), scale=2)
    
    lcd.display(img)

kpu.deinit()
ld5_kpu.deinit()
fea_kpu.deinit()
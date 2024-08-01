import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import os
from PIL import Image


autoencoder_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'autoencoder_model.h5'))
autoencoder_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
encoder_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'encoder_model.h5'))
kde = joblib.load(os.path.join(os.getcwd(), 'AI_Models', 'kde_model.joblib'))
text_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'text.h5'))
sign_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'sign.h5'))

model_elevation_color = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'elevation_color_model.h5'))

corneal_color_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'corneal_color_model.h5'))

line_PTI_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'PTI_model.h5'))
line_CTSP_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'AI_Models', 'CTSP_model.h5'))

def check_anomaly(img):
    density_threshold_1 = -20
    density_threshold_2 = 5
    reconstruction_error_threshold1 = 0.05
    reconstruction_error_threshold2 = 0.07 
    img = np.array(Image.fromarray(img).resize((16, 16), Image.Resampling.LANCZOS))
    # plt.imshow(img)
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]
    
    encoded_img = encoder_model.predict([img])
    encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]
    density = kde.score_samples(encoded_img)[0]

    reconstruction = autoencoder_model.predict([img])
    reconstruction_error = autoencoder_model.evaluate([reconstruction], [img], batch_size=1)[0]

    
    if density_threshold_1 <= density <= density_threshold_2 and reconstruction_error_threshold1 <= reconstruction_error <= reconstruction_error_threshold2:
        return 0
    else:
        return 1
    

text_class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def prediction(pil_image): 
    
    resized_image = pil_image.resize((32, 32))
    new_img_array = img_to_array(resized_image)
    new_img_array = np.expand_dims(new_img_array, axis=0)
    new_img_array /= 255.0
    prediction = text_model.predict(new_img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = text_class_labels[predicted_class_index]

    return predicted_class_label


sign_class_labels = ['-', '+', '+']

def sign_prediction(pil_image): 
    
    resized_image = pil_image.resize((32, 32))
    new_img_array = img_to_array(resized_image)
    new_img_array = np.expand_dims(new_img_array, axis=0)
    new_img_array /= 255.0
    prediction = sign_model.predict(new_img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = sign_class_labels[predicted_class_index]

    return predicted_class_label


elevation_color_class_names = ['+15', '+25', '+35', '+45', '+5', '+55', '+65', '+75', '-15', '-25', '-35', '-45', '-5', '-55', '-65', '-75']


def elevation_color(x1, mask_radius=135, threshold_value=235, window_size=17, threshold_mean=12):
    
    h, w, _ = x1.shape
    center = (int(w / 2), int(h / 2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= mask_radius

    x2 = cv2.bitwise_and(x1, x1, mask=mask.astype(np.uint8))
    x2_gray = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
    _, x3 = cv2.threshold(x2_gray, threshold_value, 255, cv2.THRESH_BINARY)

    def calculate_mean_intensity(window):
        return np.mean(window)

    cropped_image_saved = False  

    for y in range(1, x3.shape[0] - window_size + 1):
        if cropped_image_saved:  
            break
        for x in range(1, x3.shape[1] - window_size + 1):
            window = x3[y:y+window_size, x:x+window_size]
            mean_intensity = calculate_mean_intensity(window)
            if mean_intensity > threshold_mean:
                cropped_image_saved = True 
                break

    q1 = y 
    q2 = q1 + 16
    q3 = x 
    q4 = q3 + 16

    x10 = x1[q1:q2, q3:q4, :]
    pixels = x10.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common_color = unique_colors[most_common_index]
    x20 = np.ones((24, 24, 3), dtype=np.uint8) * most_common_color
    
    cv2.imwrite('image.jpg', x20)

    new_image_path = 'image.jpg' 
    new_image = image.load_img(new_image_path, target_size=(24, 24))
    new_image_array = image.img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)
    new_image_array /= 255.0

    predictions = model_elevation_color.predict(new_image_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = elevation_color_class_names[predicted_class_index]
    os.remove('image.jpg')
    return predicted_class



def four_Map_Refractive_RightEye(image_path):
    img_maps_od = cv2.imread(image_path)

    time = img_maps_od[141:160, 250:320, :]

    regions_time = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26),
        (4, 15, 25, 32),
        (4, 15, 34, 41),
        (4, 15, 40, 47)]

    text_time = []

    for region in regions_time:
        a, b, c, d = region
        roi = time[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_time.append(predicted_digit)

    res_time = ''.join(text_time[:2]) + ':' + ''.join(text_time[2:4]) + ':' + ''.join(text_time[4:6])
    result_time = res_time

    exam_date = img_maps_od[141:160, 111:181, :]

    regions_exam_date = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 21, 28),
        (4, 15, 27, 34),
        (4, 15, 38, 45),
        (4, 15, 44, 51),
        (4, 15, 50, 57),
        (4, 15, 56, 63)]

    text_exam_date = []

    for region in regions_exam_date:
        a, b, c, d = region
        roi = exam_date[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_exam_date.append(predicted_digit)

    res_exam_date = ''.join(text_exam_date[:2]) + '/' + ''.join(text_exam_date[2:4]) + '/' + ''.join(text_exam_date[4:8])
    result_exam_date = res_exam_date

    birth_date = img_maps_od[118:181, 111:181, :]

    regions_birth_date = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 21, 28),
        (4, 15, 27, 34),
        (4, 15, 38, 45),
        (4, 15, 44, 51),
        (4, 15, 50, 57),
        (4, 15, 56, 63)]

    text_birth_date = []

    for region in regions_birth_date:
        a, b, c, d = region
        roi = birth_date[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_birth_date.append(predicted_digit)

    res_birth_date = ''.join(text_birth_date[:2]) + '/' + ''.join(text_birth_date[2:4]) + '/' + ''.join(text_birth_date[4:8])
    result_birth_date = res_birth_date

    # =============================================================================

    QS_value_maps_od = img_maps_od[319:338, 54:94, :]

    # =============================================================================
    k1_od = img_maps_od[226:246, 255:320, :]

    regions_k1_od = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26)]

    text_k1_od = []

    for region in regions_k1_od:
        a, b, c, d = region
        roi = k1_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_k1_od.append(predicted_digit)

    res_k1_od = ''.join(text_k1_od[:2]) + '.' + text_k1_od[2]
    result_k1_od = ''.join(text_k1_od[:2]) + '.' + text_k1_od[2] + ' D'

    # =============================================================================
    k2_od = img_maps_od[257:277, 255:320, :]

    regions_k2_od = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26)]

    text_k2_od = []

    for region in regions_k2_od:
        a, b, c, d = region
        roi = k2_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_k2_od.append(predicted_digit)

    res_k2_od = ''.join(text_k2_od[:2]) + '.' + text_k2_od[2]
    result_k2_od = ''.join(text_k2_od[:2]) + '.' + text_k2_od[2] + ' D'

    # =============================================================================
    km_od = img_maps_od[288:308, 255:320, :]

    regions_km_od = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26)]

    text_km_od = []

    for region in regions_km_od:
        a, b, c, d = region
        roi = km_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_km_od.append(predicted_digit)

    res_km_od = ''.join(text_km_od[:2]) + '.' + text_km_od[2]
    result_km_od = ''.join(text_km_od[:2]) + '.' + text_km_od[2] + ' D'

    # =============================================================================

    Q_value_od = img_maps_od[350:369, 54:94, :]

    regions_sign_Q_value_od = [
        (4, 15, 4, 8)]

    for region in regions_sign_Q_value_od:
        a, b, c, d = region
        roi = Q_value_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)
    
    res_Q_value_od, result_Q_value_od = None, None

    if predicted_sign == "-": 
        regions_Q_value_od = [
            (4, 15, 7, 14),
            (4, 15, 16, 23),
            (4, 15, 22, 29)]

        text_Q_value_od = []

        for region in regions_Q_value_od:
            a, b, c, d = region
            roi = Q_value_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_Q_value_od.append(predicted_digit)

        res_Q_value_od = text_Q_value_od[0] + '.' + ''.join(text_Q_value_od[1:])
        result_Q_value_od = " -"+ text_Q_value_od[0] + '.' + ''.join(text_Q_value_od[1:])
    else: 
        regions_Q_value_od = [
            (4, 15, 10, 17),
            (4, 15, 19, 26),
            (4, 15, 25, 32)]

        text_Q_value_od = []

        for region in regions_Q_value_od:
            a, b, c, d = region
            roi = Q_value_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_Q_value_od.append(predicted_digit)

        res_Q_value_od = text_Q_value_od[0] + '.' + ''.join(text_Q_value_od[1:])
        result_Q_value_od = text_Q_value_od[0] + '.' + ''.join(text_Q_value_od[1:])

    # =============================================================================

    astigmatism_od = img_maps_od[319:339, 255:320, :]

    regions_sign_astigmatism_od = [(4, 15, 4, 8)]

    for region in regions_sign_astigmatism_od:
        a, b, c, d = region
        roi = astigmatism_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)

    res_astigmatism_od, result_astigmatism_od = None, None

    if predicted_sign == "-": 
        regions_astigmatism_od = [
                    (4, 15, 7, 14),
                    (4, 15, 16, 23)]

        text_astigmatism_od = []

        for region in regions_astigmatism_od:
            a, b, c, d = region
            roi = astigmatism_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_astigmatism_od.append(predicted_digit)

        res_astigmatism_od = text_astigmatism_od[0] + '.' + ''.join(text_astigmatism_od[1:])
        result_astigmatism_od = " -"+ text_astigmatism_od[0] + '.' + ''.join(text_astigmatism_od[1:]) +" D"
    else: 
        regions_astigmatism_od = [
                    (4, 15, 10, 17),
                    (4, 15, 19, 26)]

        text_astigmatism_od = []

        for region in regions_astigmatism_od:
            a, b, c, d = region
            roi = astigmatism_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_astigmatism_od.append(predicted_digit)

        res_astigmatism_od = text_astigmatism_od[0] + '.' + ''.join(text_astigmatism_od[1:])
        result_astigmatism_od = " +"+ text_astigmatism_od[0] + '.' + ''.join(text_astigmatism_od[1:]) +" D"

    # =============================================================================

    a, b, c, d = 122, 433, 802, 1130
    x0 = img_maps_od[a:b, c:d, :]

    elevation_front_od =  elevation_color(x0)

    print("The elevation of the cornea at the fixation target on the elevated front is ", elevation_front_od, "µm")

    a, b, c, d = 500, 810, 802, 1130
    x = img_maps_od[a:b, c:d, :]

    elevation_back_od =  elevation_color(x)

    print("The elevation of the cornea at the fixation target on the elevated back is ", elevation_back_od, "µm")

    return [result_time, result_exam_date, result_birth_date, QS_value_maps_od, res_k1_od, result_k1_od, res_k2_od, result_k2_od, res_km_od, result_km_od, res_Q_value_od, result_Q_value_od, res_astigmatism_od, result_astigmatism_od, elevation_front_od, elevation_back_od]



def four_Map_Refraction_LeftEye(image_path):
    img_maps_os = cv2.imread(image_path)

    # =============================================================================

    QS_value_maps_os = img_maps_os[319:338, 54:94, :]

    # =============================================================================
    k1_os = img_maps_os[226:246, 255:320, :]

    regions_k1_os = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26)]

    text_k1_os = []

    for region in regions_k1_os:
        a, b, c, d = region
        roi = k1_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_k1_os.append(predicted_digit)

    res_k1_os = ''.join(text_k1_os[:2]) + '.' + text_k1_os[2] 
    result_k1_os = ''.join(text_k1_os[:2]) + '.' + text_k1_os[2] + ' D'

    # =============================================================================
    k2_os = img_maps_os[257:277, 255:320, :]

    regions_k2_os = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26)]

    text_k2_os  = []

    for region in regions_k2_os:
        a, b, c, d = region
        roi = k2_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_k2_os.append(predicted_digit)


    res_k2_os = ''.join(text_k2_os[:2]) + '.' + text_k2_os[2] 
    result_k2_os = ''.join(text_k2_os[:2]) + '.' + text_k2_os[2] + ' D'

    # =============================================================================
    km_os = img_maps_os[288:308, 255:320, :]

    regions_km_os = [
        (4, 15, 4, 11),
        (4, 15, 10, 17),
        (4, 15, 19, 26)]

    text_km_os = []

    for region in regions_km_os:
        a, b, c, d = region
        roi = km_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_km_os.append(predicted_digit)

    res_km_os = ''.join(text_km_os[:2]) + '.' + text_km_os[2]
    result_km_os = ''.join(text_km_os[:2]) + '.' + text_km_os[2] + ' D'

    # =============================================================================

    Q_value_os = img_maps_os[350:369, 54:94, :]

    regions_sign_Q_value_os = [
        (4, 15, 4, 8)]

    for region in regions_sign_Q_value_os:
        a, b, c, d = region
        roi = Q_value_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)
    
    res_Q_value_os, result_Q_value_os = None, None

    if predicted_sign == "-": 
        regions_Q_value_os = [
            (4, 15, 7, 14),
            (4, 15, 16, 23),
            (4, 15, 22, 29)]

        text_Q_value_os = []

        for region in regions_Q_value_os:
            a, b, c, d = region
            roi = Q_value_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_Q_value_os.append(predicted_digit)

        res_Q_value_os = text_Q_value_os[0] + '.' + ''.join(text_Q_value_os[1:])
        result_Q_value_os = " -"+ text_Q_value_os[0] + '.' + ''.join(text_Q_value_os[1:])
    else: 
        regions_Q_value_os = [
            (4, 15, 10, 17),
            (4, 15, 19, 26),
            (4, 15, 25, 32)]

        text_Q_value_os = []

        for region in regions_Q_value_os:
            a, b, c, d = region
            roi = Q_value_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_Q_value_os.append(predicted_digit)

        res_Q_value_os = text_Q_value_os[0] + '.' + ''.join(text_Q_value_os[1:])
        result_Q_value_os = text_Q_value_os[0] + '.' + ''.join(text_Q_value_os[1:])

    # =============================================================================

    astigmatism_os = img_maps_os[319:339, 255:320, :]

    regions_sign_astigmatism_os = [(4, 15, 4, 8)]

    for region in regions_sign_astigmatism_os:
        a, b, c, d = region
        roi = astigmatism_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)

    res_astigmatism_os, result_astigmatism_os = None, None

    if predicted_sign == "-": 
        regions_astigmatism_os = [
                    (4, 15, 7, 14),
                    (4, 15, 16, 23)]

        text_astigmatism_os = []

        for region in regions_astigmatism_os:
            a, b, c, d = region
            roi = astigmatism_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_astigmatism_os.append(predicted_digit)

        res_astigmatism_os = text_astigmatism_os[0] + '.' + ''.join(text_astigmatism_os[1:])
        result_astigmatism_os = " -"+ text_astigmatism_os[0] + '.' + ''.join(text_astigmatism_os[1:]) +" D"
    else: 
        regions_astigmatism_os = [
                    (4, 15, 10, 17),
                    (4, 15, 19, 26)]

        text_astigmatism_os = []

        for region in regions_astigmatism_os:
            a, b, c, d = region
            roi = astigmatism_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_astigmatism_os.append(predicted_digit)

        res_astigmatism_os = text_astigmatism_os[0] + '.' + ''.join(text_astigmatism_os[1:])
        result_astigmatism_os = " +"+ text_astigmatism_os[0] + '.' + ''.join(text_astigmatism_os[1:]) +" D"

    # =============================================================================

    a, b, c, d = 122, 433, 802, 1130
    x0 = img_maps_os[a:b, c:d, :]

    elevation_front_os =  elevation_color(x0)

    print("The elevation of the cornea at the fixation target on the elevated front is ", elevation_front_os, "µm")

    a, b, c, d = 500, 810, 802, 1130
    x = img_maps_os[a:b, c:d, :]

    elevation_back_os =  elevation_color(x)

    print("The elevation of the cornea at the fixation target on the elevated back is ", elevation_back_os, "µm")

    return [QS_value_maps_os, res_k1_os, result_k1_os, res_k2_os, result_k2_os, res_km_os, result_km_os, res_Q_value_os, result_Q_value_os, res_astigmatism_os, result_astigmatism_os, elevation_front_os, elevation_back_os]



def coroneal_Color_Thickness_BothEyes(image_path_right, image_path_left):
    img_maps_od = cv2.imread(image_path_right)
    img_maps_os = cv2.imread(image_path_left)


    class_names = ['300', '340', '380', '420', '460', '500', '540', '580', '620', '660', '700', '740', '780', '820', '860', '900']

    def corneal_color(x1, mask_radius=135, threshold_value=235, window_size=17, threshold_mean=12):
        
        h, w, _ = x1.shape
        center = (int(w / 2), int(h / 2))
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= mask_radius

        x2 = cv2.bitwise_and(x1, x1, mask=mask.astype(np.uint8))
        x2_gray = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
        _, x3 = cv2.threshold(x2_gray, threshold_value, 255, cv2.THRESH_BINARY)

        def calculate_mean_intensity(window):
            return np.mean(window)

        cropped_image_saved = False  

        for y in range(1, x3.shape[0] - window_size + 1):
            if cropped_image_saved:  
                break
            for x in range(1, x3.shape[1] - window_size + 1):
                window = x3[y:y+window_size, x:x+window_size]
                mean_intensity = calculate_mean_intensity(window)
                if mean_intensity > threshold_mean:
                    cropped_image_saved = True 
                    break

        q1 = y 
        q2 = q1 + 16
        q3 = x 
        q4 = q3 + 16

        x10 = x1[q1:q2, q3:q4, :]
        pixels = x10.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_common_index = np.argmax(counts)
        most_common_color = unique_colors[most_common_index]
    
        x20 = np.ones((24, 24, 3), dtype=np.uint8) * most_common_color

    
        cv2.imwrite('image.jpg', x20)

        new_image_path = 'image.jpg' 
        new_image = image.load_img(new_image_path, target_size=(24, 24))
        new_image_array = image.img_to_array(new_image)
        new_image_array = np.expand_dims(new_image_array, axis=0)
        new_image_array /= 255.0

        predictions = corneal_color_model.predict(new_image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        os.remove('image.jpg')
        return predicted_class


    a, b, c, d = 500, 810, 429, 756
    y = img_maps_od[a:b, c:d, :]

    corneal_thickness_od =  corneal_color(y)

    a, b, c, d = 500, 810, 429, 756
    y = img_maps_os[a:b, c:d, :]

    corneal_thickness_os =  corneal_color(y)

    return [corneal_thickness_od, corneal_thickness_os]



def fourier_Analysis_RightEye(image_path):
    img_fourier_od = cv2.imread(image_path)

    # =============================================================================
    QS_value_fourier_od = img_fourier_od[250:269, 268:333, :]

    # =============================================================================
    spherical_rmin_od = img_fourier_od[387:407, 606:671, :]

    regions_spherical_rmin_od = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_spherical_rmin_od = []

    for region in regions_spherical_rmin_od:
        a, b, c, d = region
        roi = spherical_rmin_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_spherical_rmin_od.append(predicted_digit)

    res_spherical_rmin_od = text_spherical_rmin_od[0] + '.' + ''.join(text_spherical_rmin_od[1:])
    result_spherical_rmin_od = text_spherical_rmin_od[0] + '.' + ''.join(text_spherical_rmin_od[1:]) + ' mm'

    # =============================================================================
    spherical_ecc_od = img_fourier_od[411:431, 606:671, :]

    regions_spherical_ecc_od = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_spherical_ecc_od = []

    for region in regions_spherical_ecc_od:
        a, b, c, d = region
        roi = spherical_ecc_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_spherical_ecc_od.append(predicted_digit)

    res_spherical_ecc_od = text_spherical_ecc_od[0] + '.' + ''.join(text_spherical_ecc_od[1:]) 
    result_spherical_ecc_od = text_spherical_ecc_od[0] + '.' + ''.join(text_spherical_ecc_od[1:]) 

    # =============================================================================
    max_decentration_od = img_fourier_od[387:407, 958:1006, :]

    regions_max_decentration_od = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_max_decentration_od = []

    for region in regions_max_decentration_od:
        a, b, c, d = region
        roi = max_decentration_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_max_decentration_od.append(predicted_digit)

    res_max_decentration_od = text_max_decentration_od[0] + '.' + ''.join(text_max_decentration_od[1:])
    result_max_decentration_od = text_max_decentration_od[0] + '.' + ''.join(text_max_decentration_od[1:]) + ' mm'


    # =============================================================================
    astigmatism_center_value_od = img_fourier_od[785:805, 606:654, :]

    regions_astigmatism_center_value_od = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_astigmatism_center_value_od = []

    for region in regions_astigmatism_center_value_od:
        a, b, c, d = region
        roi = astigmatism_center_value_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_astigmatism_center_value_od.append(predicted_digit)

    res_astigmatism_center_value_od = text_astigmatism_center_value_od[0] + '.' + ''.join(text_astigmatism_center_value_od[1:])
    result_astigmatism_center_value_od = text_astigmatism_center_value_od[0] + '.' + ''.join(text_astigmatism_center_value_od[1:]) + ' mm'

    # =============================================================================
    astigmatism_periph_value_od = img_fourier_od[809:829, 606:654, :]
    regions_astigmatism_periph_value_od = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_astigmatism_periph_value_od = []

    for region in regions_astigmatism_periph_value_od:
        a, b, c, d = region
        roi = astigmatism_periph_value_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_astigmatism_periph_value_od.append(predicted_digit)

    res_astigmatism_periph_value_od = text_astigmatism_periph_value_od[0] + '.' + ''.join(text_astigmatism_periph_value_od[1:]) 
    result_astigmatism_periph_value_od = text_astigmatism_periph_value_od[0] + '.' + ''.join(text_astigmatism_periph_value_od[1:]) + ' mm'

    # =============================================================================
    irregularity_od = img_fourier_od[785:805, 958:1006, :]

    regions_irregularity_od = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26),
        (4, 15, 25, 32)]

    text_irregularity_od = []

    for region in regions_irregularity_od:
        a, b, c, d = region
        roi = irregularity_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_irregularity_od.append(predicted_digit)

    res_irregularity_od = text_irregularity_od[0] + '.' + ''.join(text_irregularity_od[1:]) 
    result_irregularity_od = text_irregularity_od[0] + '.' + ''.join(text_irregularity_od[1:]) 

    return [QS_value_fourier_od, res_spherical_rmin_od, result_spherical_rmin_od, res_spherical_ecc_od, result_spherical_ecc_od, res_max_decentration_od, result_max_decentration_od, res_astigmatism_center_value_od, result_astigmatism_center_value_od, res_astigmatism_periph_value_od, result_astigmatism_periph_value_od, res_irregularity_od, result_irregularity_od]
    # =============================================================================



def fourier_Analysis_LeftEye(image_path):
    img_fourier_os = cv2.imread(image_path)

    # =============================================================================
    QS_value_fourier_os = img_fourier_os[250:269, 268:333, :]

    # =============================================================================
    spherical_rmin_os = img_fourier_os[387:407, 606:671, :]

    regions_spherical_rmin_os = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_spherical_rmin_os = []

    for region in regions_spherical_rmin_os:
        a, b, c, d = region
        roi = spherical_rmin_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_spherical_rmin_os.append(predicted_digit)

    res_spherical_rmin_os = text_spherical_rmin_os[0] + '.' + ''.join(text_spherical_rmin_os[1:])
    result_spherical_rmin_os = text_spherical_rmin_os[0] + '.' + ''.join(text_spherical_rmin_os[1:]) + ' mm'

    # =============================================================================
    spherical_ecc_os = img_fourier_os[411:431, 606:671, :]

    regions_spherical_ecc_os = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_spherical_ecc_os = []

    for region in regions_spherical_ecc_os:
        a, b, c, d = region
        roi = spherical_ecc_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_spherical_ecc_os.append(predicted_digit)

    res_spherical_ecc_os = text_spherical_ecc_os[0] + '.' + ''.join(text_spherical_ecc_os[1:]) 
    result_spherical_ecc_os = text_spherical_ecc_os[0] + '.' + ''.join(text_spherical_ecc_os[1:]) 

    # =============================================================================
    max_decentration_os = img_fourier_os[387:407, 958:1006, :]

    regions_max_decentration_os = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_max_decentration_os = []

    for region in regions_max_decentration_os:
        a, b, c, d = region
        roi = max_decentration_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_max_decentration_os.append(predicted_digit)

    res_max_decentration_os = text_max_decentration_os[0] + '.' + ''.join(text_max_decentration_os[1:])
    result_max_decentration_os = text_max_decentration_os[0] + '.' + ''.join(text_max_decentration_os[1:]) + ' mm'

    # =============================================================================
    astigmatism_center_value_os = img_fourier_os[785:805, 606:654, :]

    regions_astigmatism_center_value_os = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_astigmatism_center_value_os = []

    for region in regions_astigmatism_center_value_os:
        a, b, c, d = region
        roi = astigmatism_center_value_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_astigmatism_center_value_os.append(predicted_digit)

    res_astigmatism_center_value_os = text_astigmatism_center_value_os[0] + '.' + ''.join(text_astigmatism_center_value_os[1:])
    result_astigmatism_center_value_os = text_astigmatism_center_value_os[0] + '.' + ''.join(text_astigmatism_center_value_os[1:]) + ' mm'

    # =============================================================================
    astigmatism_periph_value_os = img_fourier_os[809:829, 606:654, :]

    regions_astigmatism_periph_value_os = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26)]

    text_astigmatism_periph_value_os = []

    for region in regions_astigmatism_periph_value_os:
        a, b, c, d = region
        roi = astigmatism_periph_value_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_astigmatism_periph_value_os.append(predicted_digit)

    res_astigmatism_periph_value_os = text_astigmatism_periph_value_os[0] + '.' + ''.join(text_astigmatism_periph_value_os[1:]) 
    result_astigmatism_periph_value_os = text_astigmatism_periph_value_os[0] + '.' + ''.join(text_astigmatism_periph_value_os[1:]) + ' mm'

    # =============================================================================
    irregularity_os = img_fourier_os[785:805, 958:1006, :]

    regions_irregularity_os = [
        (4, 15, 4, 11),
        (4, 15, 13, 20),
        (4, 15, 19, 26),
        (4, 15, 25, 32)]

    text_irregularity_os = []

    for region in regions_irregularity_os:
        a, b, c, d = region
        roi = irregularity_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_digit = prediction(pil_roi)
        text_irregularity_os.append(predicted_digit)

    res_irregularity_os = text_irregularity_os[0] + '.' + ''.join(text_irregularity_os[1:]) 
    result_irregularity_os = text_irregularity_os[0] + '.' + ''.join(text_irregularity_os[1:]) 


    return [QS_value_fourier_os, res_spherical_rmin_os, result_spherical_rmin_os, res_spherical_ecc_os, result_spherical_ecc_os, res_max_decentration_os, result_max_decentration_os, res_astigmatism_center_value_os, result_astigmatism_center_value_os, res_astigmatism_periph_value_os, result_astigmatism_periph_value_os, res_irregularity_os, result_irregularity_os]
    # =============================================================================



def zernik_Analysis_Wavefront_Aberration_Cornea_Back_RightEye(image_path):
    img_zernike_od = cv2.imread(image_path)

    QS_value_zernike_od = img_zernike_od[339:358, 923:988, :]

    res_zernike_od, result_zernike_od = None, None

    zernike_od = img_zernike_od[339:358, 1110:1175, :]

    regions_sign_zernike_od = [
        (4, 15, 4, 8)]

    for region in regions_sign_zernike_od:
        a, b, c, d = region
        roi = zernike_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)

    if predicted_sign == "-":  
        regions_zernike_od = [
            (4, 15, 7, 14),
            (4, 15, 16, 23),
            (4, 15, 22, 29),
            (4, 15, 28, 35)]

        text_zernike_od = []

        for region in regions_zernike_od:
            a, b, c, d = region
            roi = zernike_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike_od.append(predicted_digit)
        res_zernike_od = "-" + text_zernike_od[0] + '.' + ''.join(text_zernike_od[1:]) 
        result_zernike_od = "-" + text_zernike_od[0] + '.' + ''.join(text_zernike_od[1:]) + " µm"

        
    else:
        regions_zernike_od = [
                (4, 15, 4, 11),
                (4, 15, 13, 20),
                (4, 15, 19, 26),
                (4, 15, 25, 32)]

        text_zernike_od = []

        for region in regions_zernike_od:
            a, b, c, d = region
            roi = zernike_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike_od.append(predicted_digit)
        res_zernike_od = text_zernike_od[0] + '.' + ''.join(text_zernike_od[1:]) 
        result_zernike_od = text_zernike_od[0] + '.' + ''.join(text_zernike_od[1:]) + " µm"

    return [QS_value_zernike_od, res_zernike_od, result_zernike_od]



def zernik_Analusis_Wavefront_Aberration_Cornea_Front_RightEye(image_path):
    img_zernike1_od = cv2.imread(image_path)

    # =============================================================================
    QS_value_zernike1_od = img_zernike1_od[339:358, 923:988, :]

    res_zernike1_od, result_zernike1_od = None, None
    # =============================================================================

    zernike1_od = img_zernike1_od[339:358, 1110:1175, :]

    regions_sign_zernike1_od = [
        (4, 15, 4, 8)]

    for region in regions_sign_zernike1_od:
        a, b, c, d = region
        roi = zernike1_od[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)

    if predicted_sign == "-":  
        regions_zernike1_od = [
            (4, 15, 7, 14),
            (4, 15, 16, 23),
            (4, 15, 22, 29),
            (4, 15, 28, 35)]

        text_zernike1_od = []

        for region in regions_zernike1_od:
            a, b, c, d = region
            roi = zernike1_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike1_od.append(predicted_digit)
        res_zernike1_od = "-" + text_zernike1_od[0] + '.' + ''.join(text_zernike1_od[1:]) 
        result_zernike1_od = "-" + text_zernike1_od[0] + '.' + ''.join(text_zernike1_od[1:]) + " µm"
        
    else:
        regions_zernike1_od = [
                (4, 15, 4, 11),
                (4, 15, 13, 20),
                (4, 15, 19, 26),
                (4, 15, 25, 32)]

        text_zernike1_od = []

        for region in regions_zernike1_od:
            a, b, c, d = region
            roi = zernike1_od[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike1_od.append(predicted_digit)
        res_zernike1_od = text_zernike1_od[0] + '.' + ''.join(text_zernike1_od[1:]) 
        result_zernike1_od = text_zernike1_od[0] + '.' + ''.join(text_zernike1_od[1:]) + " µm"

    
    return [QS_value_zernike1_od, res_zernike1_od, result_zernike1_od]
    # =============================================================================


def zernik_Analysis_Wavefront_Aberration_Cornea_Back_LeftEye(image_path):
    img_zernike_os = cv2.imread(image_path)

    # =============================================================================
    QS_value_zernike_os = img_zernike_os[339:358, 923:988, :]

    res_zernike_os, result_zernike_os = None, None

    # =============================================================================

    zernike_os = img_zernike_os[339:358, 1110:1175, :]

    regions_sign_zernike_os = [
        (4, 15, 4, 8)]

    for region in regions_sign_zernike_os:
        a, b, c, d = region
        roi = zernike_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)

    if predicted_sign == "-":  
        regions_zernike_os = [
            (4, 15, 7, 14),
            (4, 15, 16, 23),
            (4, 15, 22, 29),
            (4, 15, 28, 35)]

        text_zernike_os = []

        for region in regions_zernike_os:
            a, b, c, d = region
            roi = zernike_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike_os.append(predicted_digit)
        res_zernike_os = "-" + text_zernike_os[0] + '.' + ''.join(text_zernike_os[1:]) 
        result_zernike_os = "-" + text_zernike_os[0] + '.' + ''.join(text_zernike_os[1:]) + " µm"
        
    else:
        regions_zernike_os = [
                (4, 15, 4, 11),
                (4, 15, 13, 20),
                (4, 15, 19, 26),
                (4, 15, 25, 32)]

        text_zernike_os = []

        for region in regions_zernike_os:
            a, b, c, d = region
            roi = zernike_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike_os.append(predicted_digit)
        res_zernike_os = text_zernike_os[0] + '.' + ''.join(text_zernike_os[1:]) 
        result_zernike_os = text_zernike_os[0] + '.' + ''.join(text_zernike_os[1:]) + " µm"


    return [QS_value_zernike_os, res_zernike_os, result_zernike_os]
    # =============================================================================



def zernik_Analysis_WaveFront_Aberration_Cornea_Front_LeftEye(image_path):
    img_zernike1_os = cv2.imread(image_path)

    # =============================================================================
    QS_value_zernike1_os = img_zernike1_os[339:358, 923:988, :]

    # =============================================================================

    zernike1_os = img_zernike1_os[339:358, 1110:1175, :]
    res_zernike1_os, result_zernike1_os = None, None

    regions_sign_zernike1_os = [
        (4, 15, 4, 8)]

    for region in regions_sign_zernike1_os:
        a, b, c, d = region
        roi = zernike1_os[a:b, c:d, :]
        pil_roi = Image.fromarray(roi)
        predicted_sign = sign_prediction(pil_roi)

    if predicted_sign == "-":  
        regions_zernike1_os = [
            (4, 15, 7, 14),
            (4, 15, 16, 23),
            (4, 15, 22, 29),
            (4, 15, 28, 35)]

        text_zernike1_os = []

        for region in regions_zernike1_os:
            a, b, c, d = region
            roi = zernike1_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike1_os.append(predicted_digit)
        res_zernike1_os = "-" + text_zernike1_os[0] + '.' + ''.join(text_zernike1_os[1:]) 
        result_zernike1_os = "-" + text_zernike1_os[0] + '.' + ''.join(text_zernike1_os[1:]) + " µm"
        
    else:
        regions_zernike1_os = [
                (4, 15, 4, 11),
                (4, 15, 13, 20),
                (4, 15, 19, 26),
                (4, 15, 25, 32)]

        text_zernike1_os = []

        for region in regions_zernike1_os:
            a, b, c, d = region
            roi = zernike1_os[a:b, c:d, :]
            pil_roi = Image.fromarray(roi)
            predicted_digit = prediction(pil_roi)
            text_zernike1_os.append(predicted_digit)
        res_zernike1_os = text_zernike1_os[0] + '.' + ''.join(text_zernike1_os[1:]) 
        result_zernike1_os = text_zernike1_os[0] + '.' + ''.join(text_zernike1_os[1:]) + " µm"


    return [QS_value_zernike1_os, res_zernike1_os, result_zernike1_os]
    # =============================================================================



PTI_class_labels = ['normal', 'abnormal']

def prediction_PTI(pil_image): 
    
    resized_image = pil_image.resize((256, 256))
    new_img_array = img_to_array(resized_image)
    new_img_array = np.expand_dims(new_img_array, axis=0)
    new_img_array /= 255.0
    prediction = line_PTI_model.predict(new_img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = PTI_class_labels[predicted_class_index]

    return predicted_class_label



CSTP_class_labels = ['normal', 'abnormal']

def prediction_CSTP(pil_image): 
    
    resized_image = pil_image.resize((256, 256))
    new_img_array = img_to_array(resized_image)
    new_img_array = np.expand_dims(new_img_array, axis=0)
    new_img_array /= 255.0
    prediction = line_CTSP_model.predict(new_img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = CSTP_class_labels[predicted_class_index]

    return predicted_class_label



def enchanced_Ectasia_RightEye(image_path):
    img_enhanced_od = cv2.imread(image_path)

    image_PTI_od = img_enhanced_od[606:745, 658:1185, :]
    pil_roi = Image.fromarray(image_PTI_od)
    prediction_pti_od = prediction_PTI(pil_roi)

    image_CSTP_od = img_enhanced_od[423:562, 658:1185, :]

    pil_roi = Image.fromarray(image_CSTP_od)
    prediction_cstp_od = prediction_CSTP(pil_roi)

    return [prediction_pti_od, prediction_cstp_od]



def enchanced_Ectasia_LeftEye(image_path):
    img_enhanced_os = cv2.imread(image_path)

    image_PTI_os = img_enhanced_os[606:745, 658:1185, :]
    pil_roi = Image.fromarray(image_PTI_os)
    prediction_pti_os = prediction_PTI(pil_roi)

    image_CSTP_os = img_enhanced_os[423:562, 658:1185, :]

    pil_roi = Image.fromarray(image_CSTP_os)
    prediction_cstp_os = prediction_CSTP(pil_roi)

    return [prediction_pti_os, prediction_cstp_os]



import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import time

# 학습된 모델 로드
right_model = tf.keras.models.load_model('best_right_model.keras')
left_model = tf.keras.models.load_model('best_left_model.keras')

# 인식할 손동작 리스트
actions_right = ['meet', 'nice', 'hello', 'you', 'name', 'what']
actions_left = ['meet', 'nice', 'hello', 'you', 'name', 'what']
actions_both = ['meet', 'nice', 'hello']

seq_length = 30

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 최대 2개의 손을 인식
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 열기
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

sequence = {'left': [], 'right': []}
action_seq = []

# 한글 폰트 설정 - 절대 경로 사용
fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
font = ImageFont.truetype(fontpath, 32)

# 자막 출력 시간 초기화
last_action_time = 0
this_action = ''
last_saved_action = ''
file_index = 0

# 텍스트 파일 초기화
output_file = open(f'output_{file_index}.txt', 'w')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    current_time = time.time()
    
    if result.multi_hand_landmarks is not None:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            dot_product = np.einsum('nt,nt->n', v, v)
            dot_product = np.clip(dot_product, -1.0, 1.0)

            angle = np.arccos(dot_product)
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])

            hand_label = hand_info.classification[0].label
            if hand_label == 'Right':
                sequence['right'].append(d)
                if len(sequence['right']) > seq_length:
                    sequence['right'].pop(0)
            else:
                sequence['left'].append(d)
                if len(sequence['left']) > seq_length:
                    sequence['left'].pop(0)

        # 양손 모델 예측
        if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
            input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
            input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)
            
            y_pred_right = right_model.predict(input_data_right).squeeze()
            y_pred_left = left_model.predict(input_data_left).squeeze()
            
            i_pred_right = int(np.argmax(y_pred_right))
            i_pred_left = int(np.argmax(y_pred_left))
            
            conf_right = y_pred_right[i_pred_right]
            conf_left = y_pred_left[i_pred_left]

            if i_pred_right < len(actions_right):
                print(f"Right hand prediction: {actions_right[i_pred_right]} ({conf_right:.2f})")
            else:
                print(f"Right hand prediction index {i_pred_right} is out of range for actions_right")

            if i_pred_left < len(actions_left):
                print(f"Left hand prediction: {actions_left[i_pred_left]} ({conf_left:.2f})")
            else:
                print(f"Left hand prediction index {i_pred_left} is out of range for actions_left")

            if conf_right > 0.5 and conf_left > 0.5:
                if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
                    action = actions_both[i_pred_right]
                    action_seq.append(action)

                    if len(action_seq) > 3:
                        action_seq = action_seq[-3:]

                    if action_seq.count(action) > 1:
                        this_action = action
                    else:
                        this_action = ' '

                    last_action_time = current_time

                    # 동작 인식 후 시퀀스 초기화
                    sequence = {'left': [], 'right': []}

        # 오른손 모델 예측
        elif len(sequence['right']) == seq_length:
            input_data = np.expand_dims(np.array(sequence['right']), axis=0)
            y_pred = right_model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if i_pred < len(actions_right):
                print(f"Right hand prediction: {actions_right[i_pred]} ({conf:.2f})")
                
                if conf > 0.5:
                    action = actions_right[i_pred]
                    action_seq.append(action)

                    if len(action_seq) > 3:
                        action_seq = action_seq[-3:]

                    if action_seq.count(action) > 1:
                        this_action = action
                    else:
                        this_action = ' '

                    last_action_time = current_time

                    # 동작 인식 후 시퀀스 초기화
                    sequence = {'left': [], 'right': []}
            else:
                print(f"Right hand prediction index {i_pred} is out of range for actions_right")

        # 왼손 모델 예측
        elif len(sequence['left']) == seq_length:
            input_data = np.expand_dims(np.array(sequence['left']), axis=0)
            y_pred = left_model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if i_pred < len(actions_left):
                print(f"Left hand prediction: {actions_left[i_pred]} ({conf:.2f})")

                if conf > 0.5:
                    action = actions_left[i_pred]
                    action_seq.append(action)

                    if len(action_seq) > 3:
                        action_seq = action_seq[-3:]

                    if action_seq.count(action) > 1:
                        this_action = action
                    else:
                        this_action = ' '

                    last_action_time = current_time

                    # 동작 인식 후 시퀀스 초기화
                    sequence = {'left': [], 'right': []}
            else:
                print(f"Left hand prediction index {i_pred} is out of range for actions_left")

        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 자막 출력 (1초 동안 유지)
    if current_time - last_action_time < 1:
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
        img = np.array(img_pil)

        # 자막을 텍스트 파일로 저장
        if this_action != last_saved_action:
            output_file.write(f'{this_action}\n')
            last_saved_action = this_action

    cv2.imshow('Hand Gesture Recognition', img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        file_index += 1
        output_file.close()
        output_file = open(f'output_{file_index}.txt', 'w')
        last_saved_action = ''

cap.release()
cv2.destroyAllWindows()
output_file.close()
# <수정 코드>
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['meet', 'nice', 'Hello']
seq_length = 30

model = load_model('models/model_a.keras')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)  # 웹캠 인덱스 (0 또는 1 등)

seq_left = []
seq_right = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            if res.landmark:
                if len(res.landmark) >= 21:  # 손의 랜드마크가 최소 21개 이상 감지된 경우
                    if res.landmark[0].x < res.landmark[20].x:  # 왼손인 경우
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
                        seq_left.append(landmarks)
                    else:  # 오른손인 경우
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
                        seq_right.append(landmarks)

                    if len(seq_left) >= seq_length and len(seq_right) >= seq_length:
                        input_data_left = np.array(seq_left[-seq_length:], dtype=np.float32)  # 왼손 시퀀스 데이터
                        input_data_right = np.array(seq_right[-seq_length:], dtype=np.float32)  # 오른손 시퀀스 데이터

                        # 입력 데이터 형태 변환: (1, 30, 21, 3) -> (1, 30, 63)
                        input_data_left = input_data_left.reshape(1, seq_length, -1)  # 왼손 데이터 변환
                        input_data_right = input_data_right.reshape(1, seq_length, -1)  # 오른손 데이터 변환

                        # 모델 예측
                        y_pred_left = model.predict(input_data_left).squeeze()
                        y_pred_right = model.predict(input_data_right).squeeze()

                        action_idx_left = np.argmax(y_pred_left)
                        action_idx_right = np.argmax(y_pred_right)

                        if action_idx_left == action_idx_right:  # 두 손의 동작이 모델 예측과 동일한 경우
                            action_text = actions[action_idx_left]
                            cv2.putText(img, f'{action_text.upper()}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#<포즈 추가 코드>
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # from tensorflow.keras.models import load_model

# # actions = ['meet', 'nice', 'Hello']
# # seq_length = 30

# # model = load_model('models/modelv3.keras')

# # # MediaPipe hands model
# # mp_hands = mp.solutions.hands
# # mp_pose = mp.solutions.pose
# # mp_drawing = mp.solutions.drawing_utils

# # hands = mp_hands.Hands(
# #     max_num_hands=2,
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5)

# # pose = mp_pose.Pose(
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5)

# # cap = cv2.VideoCapture(1)  # 웹캠 인덱스 (0 또는 1 등)

# # seq_left = []
# # seq_right = []

# # while cap.isOpened():
# #     ret, img = cap.read()
# #     img0 = img.copy()

# #     img = cv2.flip(img, 1)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# #     # 손과 포즈 감지
# #     result_hands = hands.process(img)
# #     result_pose = pose.process(img)

# #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# #     # 손 랜드마크 그리기
# #     if result_hands.multi_hand_landmarks:
# #         for res in result_hands.multi_hand_landmarks:
# #             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

# #     # 포즈 랜드마크 그리기
# #     if result_pose.pose_landmarks:
# #         mp_drawing.draw_landmarks(img, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# #     if result_hands.multi_hand_landmarks:
# #         for res in result_hands.multi_hand_landmarks:
# #             if res.landmark:
# #                 if len(res.landmark) >= 21:  # 손의 랜드마크가 최소 21개 이상 감지된 경우
# #                     if res.landmark[0].x < res.landmark[20].x:  # 왼손인 경우
# #                         landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
# #                         seq_left.append(landmarks)
# #                     else:  # 오른손인 경우
# #                         landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
# #                         seq_right.append(landmarks)

# #                     if len(seq_left) >= seq_length and len(seq_right) >= seq_length:
# #                         input_data_left = np.array(seq_left[-seq_length:], dtype=np.float32)  # 왼손 시퀀스 데이터
# #                         input_data_right = np.array(seq_right[-seq_length:], dtype=np.float32)  # 오른손 시퀀스 데이터

# #                         # 입력 데이터 형태 변환: (1, 30, 21, 3) -> (1, 30, 63)
# #                         input_data_left = input_data_left.reshape(1, seq_length, -1)  # 왼손 데이터 변환
# #                         input_data_right = input_data_right.reshape(1, seq_length, -1)  # 오른손 데이터 변환

# #                         # 모델 예측
# #                         y_pred_left = model.predict(input_data_left).squeeze()
# #                         y_pred_right = model.predict(input_data_right).squeeze()

# #                         action_idx_left = np.argmax(y_pred_left)
# #                         action_idx_right = np.argmax(y_pred_right)

# #                         if action_idx_left == action_idx_right:  # 두 손의 동작이 모델 예측과 동일한 경우
# #                             action_text = actions[action_idx_left]
# #                             cv2.putText(img, f'{action_text.upper()}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

# #     cv2.imshow('img', img)
# #     if cv2.waitKey(1) == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()




#<처음 코드>
# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model

# actions = ['meet', 'nice', 'Hello']
# seq_length = 30

# model = load_model('models/modelv5.keras')

# # MediaPipe hands model
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(1)  # 웹캠 인덱스 (0 또는 1 등)

# seq_left = []
# seq_right = []

# while cap.isOpened():
#     ret, img = cap.read()
#     img0 = img.copy()

#     img = cv2.flip(img, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     if result.multi_hand_landmarks:
#         for res in result.multi_hand_landmarks:
#             if res.landmark:
#                 if len(res.landmark) >= 21:  # 손의 랜드마크가 최소 21개 이상 감지된 경우
#                     if res.landmark[0].x < res.landmark[20].x:  # 왼손인 경우
#                         landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
#                         seq_left.append(landmarks)
#                     else:  # 오른손인 경우
#                         landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
#                         seq_right.append(landmarks)

#                     if len(seq_left) >= seq_length and len(seq_right) >= seq_length:
#                         input_data_left = np.array(seq_left[-seq_length:], dtype=np.float32)  # 왼손 시퀀스 데이터
#                         input_data_right = np.array(seq_right[-seq_length:], dtype=np.float32)  # 오른손 시퀀스 데이터

#                         # 입력 데이터 형태 변환: (1, 30, 21, 3) -> (1, 30, 63)
#                         input_data_left = input_data_left.reshape(1, seq_length, -1)  # 왼손 데이터 변환
#                         input_data_right = input_data_right.reshape(1, seq_length, -1)  # 오른손 데이터 변환

#                         # 모델 예측
#                         y_pred_left = model.predict(input_data_left).squeeze()
#                         y_pred_right = model.predict(input_data_right).squeeze()

#                         action_idx_left = np.argmax(y_pred_left)
#                         action_idx_right = np.argmax(y_pred_right)

#                         if action_idx_left == action_idx_right:  # 두 손의 동작이 모델 예측과 동일한 경우
#                             action_text = actions[action_idx_left]
#                             cv2.putText(img, f'{action_text.upper()}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

#                     mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow('img', img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# #<자막 초화하는 코드>
# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model

# actions = ['meet', 'nice', 'Hello']
# seq_length = 30

# model = load_model('models/modelv5.keras')

# # MediaPipe hands model
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(1)  # 웹캠 인덱스 (0 또는 1 등)

# seq_left = []
# seq_right = []

# prev_action_text = ""  # 이전 텍스트 초기화

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         continue

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)

#     if result.multi_hand_landmarks:
#         for res in result.multi_hand_landmarks:
#             if len(res.landmark) >= 21:
#                 if res.landmark[0].x < res.landmark[20].x:
#                     landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
#                     seq_left.append(landmarks)
#                 else:
#                     landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
#                     seq_right.append(landmarks)

#                 if len(seq_left) >= seq_length and len(seq_right) >= seq_length:
#                     input_data_left = np.array(seq_left[-seq_length:], dtype=np.float32).reshape(1, seq_length, -1)
#                     input_data_right = np.array(seq_right[-seq_length:], dtype=np.float32).reshape(1, seq_length, -1)

#                     y_pred_left = model.predict(input_data_left).squeeze()
#                     y_pred_right = model.predict(input_data_right).squeeze()

#                     action_idx_left = np.argmax(y_pred_left)
#                     action_idx_right = np.argmax(y_pred_right)

#                     if action_idx_left == action_idx_right:
#                         action_text = actions[action_idx_left]

#                         if action_text != prev_action_text:
#                             cv2.putText(img, f'{action_text.upper()}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                                         fontScale=1, color=(255, 255, 255), thickness=2)
#                             prev_action_text = action_text

#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow('Hand Gesture Recognition', img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# # cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model

# actions = ['meet', 'nice', 'hello']
# seq_length = 30

# model = load_model('models/modelv6.keras')

# # MediaPipe hands 모델 초기화
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# # 웹캠 열기
# cap = cv2.VideoCapture(1)

# # 각 손의 랜드마크 시퀀스를 저장할 리스트 초기화
# seq_left = []
# seq_right = []

# # 이전에 출력된 동작을 저장할 변수 초기화
# prev_action_text = ""

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         continue

#     # 이미지 좌우 반전
#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 미디어파이프 손 모델로 이미지 처리
#     results = hands.process(img_rgb)

#     if results.multi_hand_landmarks:
#         # 각 손의 랜드마크를 순회하면서 처리
#         for hand_landmarks in results.multi_hand_landmarks:
#             # 손 랜드마크를 배열로 변환
#             landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

#             # 손의 인덱스를 기반으로 왼손 또는 오른손 시퀀스에 추가
#             if hand_landmarks.landmark[0].x < hand_landmarks.landmark[20].x:  # 왼손
#                 seq_left.append(landmarks)
#             else:  # 오른손
#                 seq_right.append(landmarks)

#             # 이미지에 손 랜드마크 그리기
#             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         # 양 손의 시퀀스 데이터가 충분히 쌓였는지 확인
#         if len(seq_left) >= seq_length and len(seq_right) >= seq_length:
#             # 왼손과 오른손 시퀀스 데이터 추출
#             input_data_left = np.array(seq_left[-seq_length:], dtype=np.float32)
#             input_data_right = np.array(seq_right[-seq_length:], dtype=np.float32)

#             # 모델 입력 형식으로 변환
#             input_data_left = input_data_left.reshape(1, seq_length, -1)
#             input_data_right = input_data_right.reshape(1, seq_length, -1)

#             # 모델로 예측
#             y_pred_left = model.predict(input_data_left).squeeze()
#             y_pred_right = model.predict(input_data_right).squeeze()

#             # 각 손의 예측된 동작 인덱스
#             action_idx_left = np.argmax(y_pred_left)
#             action_idx_right = np.argmax(y_pred_right)

#             # 각 손의 동작 텍스트
#             action_text_left = actions[action_idx_left]
#             action_text_right = actions[action_idx_right]

#             # 두 손의 동작이 모델의 동작과 일치하는지 확인
#             if action_text_left == action_text_right and action_text_left in actions:
#                 current_action_text = action_text_left
#             else:
#                 current_action_text = None

#             # 현재 동작이 이전 동작과 동일하지 않고 유효한 경우
#             if current_action_text is not None and current_action_text != prev_action_text:
#                 print(f'현재 동작: {current_action_text.upper()}')
#                 # 화면에 동작 자막 표시
#                 cv2.putText(img, f'{current_action_text.upper()}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
#                 prev_action_text = current_action_text

#     # 업데이트된 정보가 표시된 이미지 보기
#     cv2.imshow('손 추적', img)

#     # 'q' 키를 눌러 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 웹캠 해제 및 모든 OpenCV 창 닫기
# cap.release()
# cv2.destroyAllWindows()

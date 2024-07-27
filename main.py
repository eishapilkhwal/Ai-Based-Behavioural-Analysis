import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
import math
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
from moviepy.editor import VideoFileClip


class_names = ['Gloves', 'No Gloves']

left_eye = None
right_eye = None
left_shoulder = None
right_shoulder = None
left_elbow = None
right_elbow = None
left_wrist = None
right_wrist = None
left_waist = None
right_waist = None
left_knee = None
right_knee = None
left_foot_1 = None
right_foot_1 = None

y_left_wrist = None
y_left_knee = None
y_right_wrist = None
y_right_knee = None
y_left_bar = None
y_right_bar = None

angle_degrees_left_waist = None

video_path = '../Video/VideoInput.mp4'
cap = cv2.VideoCapture(video_path)

model_2_object = YOLO('../Model/bestVersion_6.pt')
model_1_pose = YOLO('../Model/yolov8m-pose.pt')
fontpath = '../Model/sfpro.ttf'
font = ImageFont.truetype(fontpath, 32)
frames = 0

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Configure the TTS engine
#engine.setProperty('rate', 150) 



output_filename = '../Video/VideoOutput.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fbs = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter(output_filename, fourcc, fbs, (frame_width, frame_height))

status = None

left_angle_count = 0
right_angle_count = 0
prev_distance = None
all_dist = []
all_dist_var = []

prev_distance_left = None
prev_distance_right = None
all_dist_left = []
all_dist_right = []

sub_status = []
frames = 0

threshold = 0.1

 

while True:
    success, img = cap.read()
    height, width, _ = img.shape

    if not success:
        print("Error reading frame. Exiting loop.")
        break

    frames += 1
    print(frames)
    if frames > 100:
        break
    angle_count = 0

    img_copy = img.copy()

    #results = model_pose.track(img, persist=True)
    

    results_2 = model_2_object(img_copy)

    for r in results_2:

        for box in r.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = box

            class_id = int(class_id)

            item = class_names[class_id]
            print(item, score)

            if score > threshold:
                print("Working")
                #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                action = item
                '''cv2.putText(img, f'{item}', (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)'''



    result_pose = model_1_pose(img_copy)

    for r in result_pose:
        keypoints = r.keypoints
        for keypoint in keypoints:
            keypoint_xy = keypoint.xy[0]
            keypoints_data = np.array(keypoint_xy)
            # cv2.circle(img_2, tuple(map(int, keypoints_data[1])), 7, (0, 255, 0), -1)
            if len(keypoints_data) > 16:
                right_foot_1 = tuple(map(int, keypoints_data[16]))

                x_right_foot, y_right_foot = right_foot_1
                #cv2.circle(img, right_foot_1, 5, (0, 0, 255), - 1)


                left_eye = tuple(map(int, keypoints_data[1]))



                left_eye = tuple(map(int, keypoints_data[1]))
                # cv2.circle(img, left_eye, 5, (255, 255, 255), -1)

                right_eye = tuple(map(int, keypoints_data[2]))
                # cv2.circle(img, right_eye, 5, (255, 255, 255), - 1)

                left_shoulder = tuple(map(int, keypoints_data[5]))
                # cv2.circle(img, left_shoulder, 5, (255, 255, 0), - 1)

                right_shoulder = tuple(map(int, keypoints_data[6]))
                # cv2.circle(img, right_shoulder, 5, (255, 255, 0), - 1)

                left_elbow = tuple(map(int, keypoints_data[7]))
                # cv2.circle(img, left_elbow, 5, (255, 255, 0), - 1)

                right_elbow = tuple(map(int, keypoints_data[8]))
                # cv2.circle(img, right_elbow, 5, (255, 255, 0), - 1)

                left_wrist = tuple(map(int, keypoints_data[9]))
                # cv2.circle(img, left_wrist, 5, (255, 255, 0), - 1)

                right_wrist = tuple(map(int, keypoints_data[10]))
                # cv2.circle(img, right_wrist, 5, (255, 255, 0), - 1)

                left_waist = tuple(map(int, keypoints_data[11]))
                #cv2.circle(img, left_waist, 5, (0, 255, 255), - 1)

                right_waist = tuple(map(int, keypoints_data[12]))
                #cv2.circle(img, right_waist, 5, (0, 255, 255), - 1)

                left_knee = tuple(map(int, keypoints_data[13]))
                #cv2.circle(img, left_knee, 5, (0, 255, 255), - 1)

                right_knee = tuple(map(int, keypoints_data[14]))
                #cv2.circle(img, right_knee, 5, (0, 255, 255), - 1)

                left_foot_1 = tuple(map(int, keypoints_data[15]))
                x_left_foot, y_left_foot = left_foot_1
                #cv2.circle(img, left_foot_1, 5, (0, 255, 255), - 1)


                x_right_foot, y_right_foot = right_foot_1
                #cv2.circle(img, right_foot_1, 5, (0, 255, 255), - 1)

                if left_wrist is not None:
                    x_left_wrist, y_left_wrist = left_wrist
                    # y_left_wrist_points.append(y_left_wrist)
                if right_wrist is not None:
                    x_right_wrist, y_right_wrist = right_wrist
                    # y_right_wrist_points.append(y_right_wrist)
                if left_knee is not None:
                    x_left_knee, y_left_knee = left_knee
                if right_knee is not None:
                    x_right_knee, y_right_knee = right_knee

                if left_waist is not None:
                    x_left_waist, y_left_waist = left_waist
                if right_waist is not None:
                    x_right_waist, y_right_waist = right_waist

                if left_eye is not None:
                    x_left_eye, y_left_eye = left_eye
                if right_eye is not None:
                    x_right_eye, y_right_eye = right_eye

                if left_shoulder is not None:
                    x_left_shoulder, y_left_shoulder = left_shoulder
                if right_shoulder is not None:
                    x_right_shoulder, y_right_shoulder = right_shoulder

                if right_elbow is not None:
                    x_right_elbow, y_right_elbow = right_elbow
                if left_elbow is not None:
                    x_left_elbow, y_left_elbow = left_elbow

                if left_waist is not None:
                    if left_knee is not None:
                        if left_foot_1 is not None:
                            print("Working")
                            #cv2.line(img, (left_waist), (left_knee), (125, 125, 0), 3)
                            #cv2.line(img, (left_knee), (left_foot_1), (255, 0, 0), 3)

                if right_waist is not None:
                    if right_knee is not None:
                        if right_foot_1 is not None:
                            print("Working")
                            #cv2.line(img, (right_waist), (right_knee), (125, 125, 0), 3)
                            #cv2.line(img, (right_knee), (right_foot_1), (255, 0, 0), 3)

                # if left_shoulder is not None:
                #     if left_elbow is not None:
                #         if left_wrist is not None:
                #             cv2.line(img, (left_shoulder), (left_elbow), (255,255,0), 3)
                #             cv2.line(img, (left_elbow), (left_wrist), (255, 0, 255), 3)
                if right_shoulder is not None:
                    if right_elbow is not None:
                        if right_wrist is not None:
                            print("Working")
                            # cv2.line(img, (right_shoulder), (right_elbow), (255,255,0), 3)
                            # cv2.line(img, (right_elbow), (right_wrist), (255, 0, 255), 3)

                            #cv2.line(img, (right_waist), (right_shoulder), (0, 255, 0), 3)
                            #cv2.line(img, (left_waist), (left_shoulder), (0, 255, 0), 3)

                #cv2.line(img, (left_foot_1), (right_foot_1), (0, 0, 255), 3)

                angle_radians_left_waist = math.atan2(left_shoulder[1] - left_waist[1],
                                                        left_shoulder[0] - left_waist[0]) - math.atan2(
                    left_knee[1] - left_waist[1],
                    left_knee[0] - left_waist[0])
                angle_degrees_left_waist = math.degrees(angle_radians_left_waist)
                if angle_degrees_left_waist < 0:
                    angle_degrees_left_waist += 360

                if 160 < angle_degrees_left_waist < 200:
                    print("Working")

                    '''cv2.putText(img, f"{angle_degrees_left_waist:.2f} degrees", (x_left_waist + 30, y_left_waist),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)'''
                else:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_left_waist:.2f} degrees",
                                (x_left_waist + 30, y_left_waist), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)'''

                angle_radians_right_waist = math.atan2(right_shoulder[1] - right_waist[1],
                                                        right_shoulder[0] - right_waist[0]) - math.atan2(
                    right_knee[1] - right_waist[1],
                    right_knee[0] - right_waist[0])
                angle_degrees_right_waist = math.degrees(angle_radians_right_waist)
                if angle_degrees_right_waist < 0:
                    angle_degrees_right_waist += 360

                if 160 < angle_degrees_right_waist < 200:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_right_waist:.2f} ", (x_right_waist - 120, y_right_waist),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)'''
                else:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_right_waist:.2f} ",
                                (x_right_waist - 120, y_right_waist), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)'''

                angle_radians_left_knee = math.atan2(left_waist[1] - left_knee[1],
                                                        left_waist[0] - left_knee[0]) - math.atan2(
                    left_foot_1[1] - left_knee[1],
                    left_foot_1[0] - left_knee[0])
                angle_degrees_left_knee = math.degrees(angle_radians_left_knee)
                if angle_degrees_left_knee < 0:
                    angle_degrees_left_knee += 360

                if 160 < angle_degrees_left_knee < 200:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_left_waist:.2f} degrees", (x_left_knee + 30, y_left_knee),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)'''
                else:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_left_waist:.2f} degrees",
                                (x_left_knee + 30, y_left_knee), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)'''

                angle_radians_right_knee = math.atan2(right_waist[1] - right_knee[1],
                                                        right_waist[0] - right_knee[0]) - math.atan2(
                    right_foot_1[1] - right_knee[1],
                    right_foot_1[0] - right_knee[0])
                angle_degrees_right_knee = math.degrees(angle_radians_right_knee)
                if angle_degrees_right_knee < 0:
                    angle_degrees_right_knee += 360

                if 160 < angle_degrees_right_knee < 200:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_right_knee:.2f} ", (x_right_knee - 120, y_right_knee),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)'''
                else:
                    print("Working")
                    '''cv2.putText(img, f"{angle_degrees_right_knee:.2f} ",
                                (x_right_knee - 120, y_right_knee), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)'''

                delta_x_foot = x_left_foot - x_right_foot
                delta_y_foot = y_left_foot - y_right_foot
                distance = np.sqrt(delta_x_foot ** 2 + delta_y_foot ** 2)
                rounded_distance = round(distance, 2)
                all_dist.append(rounded_distance)

                '''cv2.putText(img, f'{rounded_distance}', (x_left_foot + 50, y_left_foot), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)'''

                if left_foot_1 is not None:
                    if frames % 15 == 0:
                        all_dist_left.append(left_foot_1)

                if right_foot_1 is not None:
                    if frames % 15 == 0:
                        all_dist_right.append(right_foot_1)

            else:
                print("Error: keypoints_data does not have enough elements.")
                # Add appropriate handling or break out of the loop/program as needed


    if angle_degrees_left_waist is not None:

        if (angle_degrees_left_waist < 160 or angle_degrees_left_waist > 200):
            angle_count += 1
        if (angle_degrees_right_waist < 160 or angle_degrees_right_waist > 200):
            angle_count += 1
        if (angle_degrees_left_knee < 160 or angle_degrees_left_knee > 200):
            angle_count += 1
        if (angle_degrees_right_knee < 160 or angle_degrees_right_knee > 200):
            angle_count += 1

        if angle_count >= 3:

            status = 'Processing'
            satsus_2 = 'sitting'

            sub_status.append(satsus_2)
            if len(sub_status) > 5:
                if sub_status[-1] == 'sitting' and sub_status[-2] == 'sitting' and sub_status[-3] == 'sitting' \
                        and sub_status[-4] == 'sitting' and sub_status[-5] == 'sitting':
                    status = 'sitting'
                else:
                    status = 'Processing'


        elif angle_count == 2:

            angle_count = 0

            status = 'Processing'
            if (angle_degrees_left_waist < 130 or angle_degrees_left_waist > 225):
                angle_count += 1
                print(f'angle_degrees_left_waist: {angle_degrees_left_waist}')
            if (angle_degrees_right_waist < 130 or angle_degrees_right_waist > 225):
                angle_count += 1
                print(f'angle_degrees_right_waist: {angle_degrees_right_waist}')
            if (angle_degrees_left_knee < 130 or angle_degrees_left_knee > 225):
                angle_count += 1
                print(f'angle_degrees_left_knee: {angle_degrees_left_knee}')
            if (angle_degrees_right_knee < 130 or angle_degrees_right_knee > 225):
                angle_count += 1
                print(f'angle_degrees_right_knee: {angle_degrees_right_knee}')

            if angle_count >= 2:

                status = 'sitting'

            else:
                status = 'walking or standing'

        else:

            status = 'walking or standing'

        if status == 'walking or standing':

            if len(all_dist_left) > 1 and len(all_dist_right) > 1:
                x1, y1 = all_dist_left[-1]
                x2, y2 = all_dist_left[-2]
                delta_distance_left = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                print(delta_distance_left)

                x1, y1 = all_dist_right[-1]
                x2, y2 = all_dist_right[-2]
                delta_distance_right = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                print(delta_distance_right)

                if delta_distance_left > 12.5 and delta_distance_right > 12.5:
                    status = 'walking'

                else:
                    if delta_distance_left > 15 or delta_distance_right > 15:
                        status = 'only one foot move'
                    else:
                        status = 'standing'

            else:
                status = 'Processing'

    if status is not None:
        if status == 'Processing':
            description = f"Action is {status} but the person is {action}"
            cv2.line(img, ((width - (width - 50)), 25), ((width - (width - 1000)), 25), [85, 45, 255], 40)
            cv2.putText(img, description, ((width - (width - 50)), 35), 0, 1, [225, 225, 225], thickness = 3, lineType = cv2.LINE_AA)
        else:
            description = f"A person is {status} and {action}"
            cv2.line(img, ((width - (width - 50)), 25), ((width - (width - 1000)), 25), [85, 45, 255], 40)
            cv2.putText(img, description , ((width - (width - 50)), 35), 0, 1, [225, 225, 225], thickness = 3, lineType = cv2.LINE_AA)
        
    if frames % 6 == 0:
        print(description)
        engine.say(description)
        engine.runAndWait()
    
    if img is not None:
        final_2 = cv2.resize(img, (800, 600))
    else:
        print("Warning: Empty frame. Skipping to the next frame.")
        continue

    cv2.imshow('Image', final_2)
    output_video.write(img)
    cv2.waitKey(1)


output_video.release()
cv2.destroyAllWindows()
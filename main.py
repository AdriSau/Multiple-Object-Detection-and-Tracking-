import os
import random
from collections import deque
import cv2
from ultralytics import YOLO
import pandas as pd
from tracker import Tracker
import numpy as np
from shapely.geometry import Point, Polygon
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

video_path = "Datasets_and_testfootage/testingFootage/ETSISI_AUP.mp4"
video_path_movement = "Datasets_and_testfootage/testingFootage/prueba.mp4"
video_out_path = "Datasets_and_testfootage/testingFootage/out.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
frame_height, frame_width, nChannnel = frame.shape
print(str( cap.get(cv2.CAP_PROP_FPS)))
# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))
xvalue_div_line = 591

model =  YOLO('Models/Y8_500_HW_Etsisi.pt')
#model =  YOLO('Models/best.pt')
tracker = Tracker()
data_deque = {}
speed_ds = {}
count_cars = [0, 0]
_,fixed_img_dots = cap.read();
_,fixed_img_lines = cap.read();
dict = {'ID':[], 'x':[], 'y':[],'speed':[]}
speed_hist_dict = {'ID':[], 'State':[], 'Cont':[],'Result':[]}
df_output = pd.DataFrame(dict)
sp_f = pd.DataFrame(speed_hist_dict)

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(50)]
detection_threshold = 0.2


def inside_main_det_zone (x_value, y_value):
    zone = [(0,377),(285,273),(862,273),(1279,421),(frame_width,frame_height),(0,frame_height)]
    return Polygon(zone).contains(Point(x_value, y_value))

def inside_speed_det_zone (x_value, y_value):
    print("SPEED ZONE parameters : width: " + str(frame_width) + " high: "+ str(frame_height))
    zone_s = [(780,478),(1185,446),(frame_width,492),(frame_width,frame_height),(928,frame_height)]
    return Polygon(zone_s).contains(Point(x_value, y_value))
def update_cars_count(center_x, frame_width):
    if center_x < (xvalue_div_line):
        count_cars[0]=count_cars[0]+1
    else:
        count_cars[1]=count_cars[1]+1


def draw_permanents():
    cv2.putText(frame, "Last detection: ", (int(frame_width - 395), int(22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
    cv2.line(frame, (int(xvalue_div_line), 0), (int(xvalue_div_line), frame_height), (0,0,0), thickness=2)

    cv2.rectangle(frame, (int(xvalue_div_line - 120), int(0)), (int(xvalue_div_line-23), int(39)), (14, 26, 71), thickness=-1)
    cv2.rectangle(frame, (int(xvalue_div_line + 28), int(0)), (int(xvalue_div_line + 125), int(39)), (14, 26, 71), thickness=-1)

    #cont Izq
    cv2.putText(frame, "Left-line" , (int(xvalue_div_line-100), int(15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
    cv2.putText(frame, "total cars: " + str(count_cars[0]), (int(xvalue_div_line-115), int(27)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
    #cont Derecha
    cv2.putText(frame, "Right-line", (int(xvalue_div_line + 40), int(15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
    cv2.putText(frame, "total cars: " + str(count_cars[1]), (int(xvalue_div_line + 33), int(27)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)



def draw_permanent_tarces(data_deque,track_id,fixed_img_dots,color):
    x1 = data_deque[track_id][-1][0]
    y1 = data_deque[track_id][-1][1]
    x2 = data_deque[track_id][-1][0]
    y2 = data_deque[track_id][-1][1]
    cv2.line(fixed_img_dots, (int(x1), int(y1)), (int(x2), int(y2)),
             color, 2)

def update_output_df(df_out,idd,x_udf,y_udf,speed):
   dic_aux = {'ID':[idd], 'x':[x_udf], 'y':[y_udf],'speed':[speed]}
   df_out = pd.concat([df_out,pd.DataFrame(dic_aux)],ignore_index=True)
   return df_out
def speed_mang(track_id,speed_ds,Bot_cent_X,Bot_cent_Y,limit_x):
    #deteccion nueva, se añade al seguimiento
    cv2.line(frame, (int(limit_x), 0), (int(limit_x), frame_height), (0,255,0), thickness=2)
    if track_id not in speed_ds:
        speed_ds [track_id] = []
        if object_center_X > limit_x and inside_speed_det_zone(Bot_cent_X, Bot_cent_Y):
            speed_ds[track_id].append(True)
            speed_ds[track_id].append(0)
            speed_ds[track_id].append("...")
        else:
            speed_ds[track_id].append(False)
            speed_ds[track_id].append(0)
            speed_ds[track_id].append("NA")
    elif speed_ds[track_id][0] == True:
        if speed_ds[track_id][1] == 0 and Bot_cent_Y <= -0.063107 * Bot_cent_X + 494.5:
            print("ENTRO: ID" + str(track_id) + " con estado: " + str(speed_ds[track_id][0]))
            speed_ds[track_id][1] = 1
            cv2.line(frame, (721, 449), (1133, 423), (0,255,0), thickness=2)
        elif speed_ds[track_id][1] >= 1:
            if Bot_cent_Y <= -0.019934 * Bot_cent_X + 364.38:
                speed_ds[track_id][2] = round(0.011 / ((speed_ds[track_id][1] * (1.0/30.0))/3600.0),2)
                speed_ds[track_id][0] = False
                cv2.line(frame, (671, 351), (972, 345), (0,0,255), thickness=2)
                print("SALIO: ID" + str(track_id) + " con estado: " + str(speed_ds[track_id][0])+ " ticks = " +str(speed_ds[track_id][1])+ " vel: " + str(speed_ds[track_id][2]))
            else:
                speed_ds[track_id][1] +=1
def draw_guidelines():
    #speed lines
    cv2.line(frame, (671, 351), (972, 345), (0,0,0), thickness=1)
    cv2.line(frame, (721, 449), (1133, 423), (0,0,0), thickness=1)
    #speed detection zone
    zone_s = np.array([[780, 478], [1185, 446], [frame_width, 492], [frame_width, frame_height], [928, frame_height]])
    zone_s = zone_s.reshape((-1,1,2))
    cv2.polylines(frame, [zone_s], True, (252, 255, 97), thickness=1)
    #main detection zone
    zone = np.array([[0, 377], [285, 273], [862, 273], [1279, 421], [frame_width, frame_height], [0, frame_height]])
    zone = zone.reshape((-1,1,2))
    cv2.polylines(frame, [zone],True, (252, 255, 97), thickness = 1)


while ret:
    results = model(frame)
    aux = 0
    ex = 0
    for result in results:
        detections = []
        Speed_line_top = (255, 255, 255)
        Speed_line_bot = (255, 255, 255)
        for r in result.boxes.data.tolist():
            x_top_L, y_top_L, x_bot_R, y_bot_R, conf_value, class_id = r
            x_top_L = int(x_top_L)
            x_bot_R = int(x_bot_R)
            y_top_L = int(y_top_L)
            y_bot_R = int(y_bot_R)
            class_id = int(class_id)
            bot_x_value = (x_top_L + x_bot_R) / 2
            if conf_value > detection_threshold and inside_main_det_zone(bot_x_value,y_bot_R):
                detections.append([x_top_L, y_top_L, x_bot_R, y_bot_R, conf_value])
        tracker.update(frame, detections)
        #frame_height, frame_width, nChannnel = frame.shape
        draw_guidelines()
        print("altura: " + str(frame_height) +" anchura: "+ str(frame_width))
        cv2.rectangle(frame, (int(frame_width - 400), int(0)), (int(frame_width), int(28)), (0, 0, 0), thickness=-1)
        for track in tracker.tracks:
            aux += 1
            bbox = track.bbox
            x_top_L, y_top_L, x_bot_R, y_bot_R = bbox
            track_id = track.track_id
            object_center_X = (int(x_top_L) + int(x_bot_R)) / 2
            object_center_Y = (int(y_top_L) + int(y_bot_R)) / 2
            #print(
            #    'car detected: Id: ' + str(track_id) + ' Detected in cordinates: (' + str(object_center_X) + ',' + str(
            #        object_center_Y) + ')')
            cv2.rectangle(frame, (int(x_top_L), int(y_top_L)), (int(x_bot_R), int(y_bot_R)),
                          (colors[track_id % len(colors)]), 3)
            cv2.circle(frame, (int(object_center_X), int(object_center_Y)), 2, (colors[track_id % len(colors)]),
                       thickness=-1)
            cv2.rectangle(frame, (int(x_top_L), int(y_top_L - 15)), (int(x_top_L + 120), int(y_top_L)),
                          (colors[track_id % len(colors)]), thickness=-1)
            if (aux > 0) & (aux == len(tracker.tracks)):
                cv2.putText(frame,
                            "Last detection: Id:" + str(track_id) + " posX: " + str(object_center_X) + " posY: " + str(
                                object_center_Y), (int(frame_width - 395), int(22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
            speed_mang(track_id,speed_ds,object_center_X,y_bot_R,591)
            # creo lista
            if track_id not in data_deque:
                data_deque[track_id] = deque(maxlen=15)
                update_cars_count(object_center_X, frame_width)
            # añado el centro
            if len(data_deque[track_id]) >= 15:
                data_deque[track_id].pop()
                data_deque[track_id].appendleft((object_center_X, object_center_Y))
            else:
                data_deque[track_id].append((object_center_X, object_center_Y))
            for i in range(1, len(data_deque[track_id])):
                punto1_x = data_deque[track_id][i - 1][0]
                punto1_y = data_deque[track_id][i - 1][1]
                punto2_x = data_deque[track_id][i][0]
                punto2_y = data_deque[track_id][i][1]
                cv2.line(frame, (int(punto1_x), int(punto1_y)), (int(punto2_x), int(punto2_y)),
                         (colors[track_id % len(colors)]), 2)
                cv2.line(fixed_img_lines, (int(punto1_x), int(punto1_y)), (int(punto2_x), int(punto2_y)),
                         (colors[track_id % len(colors)]), 2)
            if len(data_deque[track_id]) >= 2:
                draw_permanent_tarces(data_deque,track_id,fixed_img_dots,colors[track_id % len(colors)])
            if track_id in speed_ds:
                cv2.putText(frame, "id:" + str(track_id) + " Km,h:" + str(speed_ds[track_id][2]), (int(x_top_L), int(y_top_L)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
            else:
                cv2.putText(frame, "id:" + str(track_id) + " Km,h:    ", (int(x_top_L), int(y_top_L)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
            df_output = update_output_df(df_output,track_id,object_center_X,object_center_Y,speed_ds[track_id][2])
        draw_permanents()
        if ex == 0:
            cv2.imwrite(os.path.join('runs', 'aux.png'), frame)

    aux = 0

    cv2.imshow('display_video', frame)
    cv2.waitKey(30)
    # cap_out.write(frame)
    ret, frame = cap.read()
cv2.imwrite(os.path.join('runs' , 'permanent_traces_lines.png'), fixed_img_lines)
cv2.imwrite(os.path.join('runs' , 'permanent_traces_dots.png'), fixed_img_dots)
clean_df = df_output.copy()
df_output.to_csv(os.path.join('runs' , 'raw_data'), index = False)
clean_df = clean_df.drop_duplicates(subset=['ID'], keep='last')
clean_df.to_csv(os.path.join('runs', 'filtered_data'),index = False )
cap.release()
# cap_out.release()
cv2.destroyAllWindows()

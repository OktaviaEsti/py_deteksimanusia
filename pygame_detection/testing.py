from sre_constants import BRANCH
from symbol import parameters
from turtle import circle

from pyparsing import line
import env, sensors , feature
import random
import pygame
import math
import numpy as np

import matplotlib.pyplot as plt

import time
import pandas as pd


folder_path = 'D:\ESTI\Capstone_2021\Python\Testing\data_raw\collected_circle'
def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

FeatureMap = feature.featuresDetection()
environment = env.buildenvironment((800,1400))
environment.originalMap = environment.map.copy()
laser = sensors.LaserSensor(200,environment.originalMap,uncertenty=(0.01,0.0)) #tdnya 0.5 001
environment.map.fill((255,255,255))
environment.infomap = environment.map.copy()
originalMap = environment.map.copy()


position = np.array([90, 552])          # initial robot pos

running = True
FEATURE_DETECTION = True
BREAKE_POINT_INO = 0

#Tes Akurasi
circle_data = []
person_data =[]
circle_ok = []
error_jarak =[]

def cirlce_accuracy(data):
    data = np.array(data)
    circle_pos = data[:,1]
    real_circle = np.array([
        [343, 626, 5],[372.5, 627, 5.5],
        [560.5, 406.5, 7],[587.5, 408.5, 8.5],
        [748, 597, 5],[776, 597, 4.5],
        [835.5, 485, 7.5],[868.5, 482, 7.5],
        [1014.5, 538, 6],[1017.5, 565.5, 7],
        [1096, 496.5, 20]])
    r_pos = np.array(real_circle[:,2])
    x_pos = np.array(real_circle[:,0])
    y_pos = np.array(real_circle[:,1])
    error_jarak2 =[]
    error_jari = []
    for circle in circle_pos:
        for x in range(0, len(x_pos)):
            if circle[0]>(x_pos[x]-r_pos[x]) and circle[0]<(x_pos[x]+r_pos[x]) and circle[1]>(y_pos[x]-r_pos[x]) and circle[1]<(y_pos[x]+r_pos[x]):
                circle_ok.append(circle)

                delta_jarak = FeatureMap.distpoint2point([circle[0],circle[1]],[x_pos[x], y_pos[x]])
                error_jarak.append([(delta_jarak), data[x,0]])

                delta_jari = abs(circle[2]-r_pos[x])/r_pos[x]*100
                hai = abs(delta_jarak)/data[x,0]*100
                error_jarak2.append(hai)
                error_jari.append(delta_jari)
    akurasi_deteksi = len(circle_ok)*100/len(circle_data)
    print("\n\nAkurasi Deteksi Lingkaran = %.02f" % (akurasi_deteksi),"%")
    # rata_error_posisi = np.average(np.array(error_jarak))
    # print(error_jarak)
    error_jari = 100 - np.average(np.array(error_jari))
    error_jarak2 = 100 - np.average(np.array(error_jarak2))
    print("\n\nAkurasi Posisi Lingkaran = %.02f" % (error_jarak2),"%")
    print("\n\nAkurasi Jari-jari Lingkaran = %.02f" % (error_jari),"%\n\n\n")

def person_accuracy(data):
    data = np.array(data)
    person_in_map = np.array([
        [[343, 626, 5],[372.5, 627, 5.5]],
        [[560.5, 406.5, 7],[587.5, 408.5, 8.5]],
        [[748, 597, 5],[776, 597, 4.5]],
        [[835.5, 485, 7.5],[868.5, 482, 7.5]],
        [[1014.5, 538, 6],[1017.5, 565.5, 7]]
    ])
    person_detected = []
    # print(person_pos.shape, person_pos[1][1][1]) # out= (5, 2, 3) 209.5
    for i in range(0,len(data)):
        circle1, circle2, npeople = data[i]
        person_xpos = np.minimum(circle1[0],circle2[0])
        person_ypos = np.minimum(circle1[1],circle2[1])
        person_xpos2 = np.maximum(circle1[0],circle2[0])
        person_ypos2 = np.maximum(circle1[1],circle2[1])

        for k in range(0, len(person_in_map)):
            kaki1, kaki2 = person_in_map[k]
            if(
                (person_xpos>(np.minimum(kaki1[0],kaki2[0])-kaki1[2]))
                and
                (person_xpos2<(np.maximum(kaki1[0],kaki2[0])+kaki2[2]))
                and
                (person_ypos>(np.minimum(kaki1[1],kaki2[1])-kaki1[2]))
                and
                (person_ypos2<(np.maximum(kaki1[1],kaki2[1])+kaki2[2]))
                ):
                person_detected.append([[person_xpos,person_ypos],[person_xpos2, person_ypos2]])
    Akurasi_deteksi_manusia = len(person_detected)/len(data)*100
    if len(data) != 0:
        # print(np.array(person_detected))
        print("\n\n\nJumlah Manusia Terdeteksi= ",len(data),"orang\n\n\n")
        print("\n\nJumlah Deteksi yang Tepat = ",len(person_detected),"orang\n\n\n")
        print("\n\n\nAkurasi Deteksi Manusia = %.02f" % (Akurasi_deteksi_manusia),"%\n\n\n")
    else:
        print("\n\n\n Tidak terdeteksi manusia")

def plot_line(error_jarak):
    error_jarak.sort()
    error_jarak = np.array(error_jarak)
    plt.plot(error_jarak[:,0]*0.01, error_jarak[:,1]*0.01)
    plt.title('Line Fitting')
    plt.xlabel('Jarak Robot Lingkaran(m)', color='#1C2833')
    plt.ylabel('Error Posisi(m)', color='#1C2833')
    plt.grid()
    plt.tight_layout()
    plt.show()
    

while running:
    environment.infomap = originalMap.copy()
    FEATURE_DETECTION = True
    BREAKE_POINT_INO = 0
    END_POINT = [0,0]
    sensorON = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if pygame.mouse.get_focused():
        sensorON = True
    # elif not pygame.mouse.get_focused():
    #     sensorON = False

    if sensorON:

        if position[0]<1210: #posisi target 1150
            # robot_step = np.array([0,10])        #perpindahan y
            robot_step = np.array([10,0])        #perpindahan x
        else:
            # sensorON =  False
            # circle_data = np.array(circle_data)
            # cirlce_accuracy(circle_data)
            person_accuracy(person_data)
            running = False
            # robot_step = np.array([0,0]) 

        position = position + robot_step
        laser.position = position
        sensor_data = laser.sense_obstacle()
        FeatureMap.laser_points_set(sensor_data)
        print("epoch")

        #Variabel-variabel
        person = []
        counter_person = 0
        n_counter = []
        FeatureMap.CIRCLES_DETECTED = []

        #draw all of the detected points 
        # map_pts = np.array(FeatureMap.LASERPOINTS)
        # print("map= \n", FeatureMap.LASERPOINTS)
        # map_pts = map_pts[:,0]
        # COLOR = random_color()
        # for point in map_pts:
        #         environment.infomap.set_at((int(point[0]),int(point[1])), (0,255,0))
        #         pygame.draw.circle(environment.infomap,COLOR,(int(point[0]),int(point[1])),2,0)

        circless = []
        results = []
        while BREAKE_POINT_INO < abs(FeatureMap.NP - FeatureMap.PMIN):
            # print("BP LAST: ",BREAKE_POINT_INO,abs(FeatureMap.NP - FeatureMap.PMIN))
            line_seedSeg = FeatureMap.seed_segment_detection(laser.position, BREAKE_POINT_INO)
            circle_seedSeg = FeatureMap.circle_seed_segment_detection(laser.position, BREAKE_POINT_INO)
            line_state = False       
            circle_state = False
            circle_state2 = False

            if line_seedSeg == False and  circle_seedSeg == False:
                # BREAKE_POINT_INO =FeatureMap.break_point_backup
                break
            elif (line_seedSeg != False) and (circle_seedSeg != False):
                    line_state = True
            elif line_seedSeg == False and circle_seedSeg != False:     
                circle_state = True
            elif circle_seedSeg == False and line_seedSeg != False:
                line_state = True


            if line_state == True:
                line_seedSegment = line_seedSeg[0]
                PREDICTED_POINTS_TODRAW = line_seedSeg[1]
                INDICES = line_seedSeg[2]
                results = FeatureMap.seed_segment_growing(INDICES,BREAKE_POINT_INO)
                results_line = results
                if results != False:
                    OUTERMOST = results[2]
                    #Balik ke circle detection kalu garis pendek
                    if (FeatureMap.distpoint2point(OUTERMOST[0], OUTERMOST[1])<20) :
                        circle_results = FeatureMap.circle_seed_segment_growing(INDICES,BREAKE_POINT_INO)
                        x_circ, y_circ, r_circ =  FeatureMap.CIRCLE_PARAMS
                        if (FeatureMap.res_circle < FeatureMap.res_line) and (r_circ<10):
                            results = circle_results
                            circle_state2 = True
                            circle_state = True   #different bool var to escape circle segment growing func
                            line_state = False

                elif results == False and circle_seedSeg != False:
                    circle_state = True
                    line_state = False


            if circle_state == True:
                if circle_state2 == True:
                    line_seedSegment = circle_seedSeg[0]
                    PREDICTED_POINTS_TODRAW = circle_seedSeg[1]
                    INDICES = circle_seedSeg[2]
                else:
                    if circle_seedSeg != False:
                        line_seedSegment = circle_seedSeg[0]
                        PREDICTED_POINTS_TODRAW = circle_seedSeg[1]
                        INDICES = circle_seedSeg[2]
                        results = FeatureMap.circle_seed_segment_growing(INDICES,BREAKE_POINT_INO)
                        x_circ, y_circ, r_circ =  FeatureMap.CIRCLE_PARAMS
                        if (FeatureMap.res_circle > 10) or (r_circ>20):
                            if(line_seedSeg != False):
                                circle_state = False
                                line_seedSegment = line_seedSeg[0]
                                PREDICTED_POINTS_TODRAW = line_seedSeg[1]
                                INDICES = line_seedSeg[2]
                                results = results_line


                    if circle_seedSeg==False:
                        continue
                

            if results == False:
                    BREAKE_POINT_INO = INDICES[1]
                    circle_state = False
                    circle_state2 = False
                    line_state = False
                    continue
            line_eq = results[1]
            params = results[4]
            line_seg = results[0]
            OUTERMOST = results[2]
           
            BREAKE_POINT_INO = results[3]

            if line_state == True:
                END_POINT[0] = FeatureMap.projection_point2line(OUTERMOST[0], params)
                END_POINT[1] = FeatureMap.projection_point2line(OUTERMOST[1], params)
            if circle_state == True:
                jarak = FeatureMap.distpoint2circle(params, position)
                circle_data.append([jarak, params])
                if(params[2]<10):
                    FeatureMap.CIRCLES_DETECTED.append(params)

                END_POINT[0] = FeatureMap.projection_point2circle(OUTERMOST[0], params)
                END_POINT[1] = FeatureMap.projection_point2circle(OUTERMOST[1], params)

         #    OBJECT CLASSIFICATION 
                # kalau ada 2 lingkaran dengan jarak dalam rentang x, maka akan dianggapp manusia
                #coba ada while-nya
            
            circles_detected = FeatureMap.CIRCLES_DETECTED   

            if circles_detected :
                for n in range(0,len(circles_detected)):
                    if n == (len(circles_detected)-1):
                        break      
                    if n in n_counter:
                        continue
                    distance_circles = FeatureMap.dist_twocircles(circles_detected[n], circles_detected[n+1])
                    if distance_circles <= 40:
                        counter_person += 1
                        person.append([circles_detected[n],circles_detected[n+1], counter_person])
                        person_data.append(np.array([circles_detected[n],circles_detected[n+1], counter_person]))
                        n_counter.append(n)
                        n_counter.append(n+1)

            COLOR = random_color()
            CIRCLE_COLOR = random_color()
            RECT_COLOR = (205,92,92)
       
       
            for point in line_seg:
                environment.infomap.set_at((int(point[0][0]),int(point[0][1])), (0,255,0))
                pygame.draw.circle(environment.infomap,COLOR,(int(point[0][0]),int(point[0][1])),2,0)
            
            # x= FeatureMap.NP
            # print("prb=",FeatureMap.LASERPOINTS[24:x])
                
            # draw robot position:
            pygame.draw.circle(environment.infomap, (0,0,255) ,laser.position,5) # draw robot pos

            
            if line_state == True:
                pygame.draw.line(environment.infomap,  (0,0,255) ,END_POINT[0], END_POINT[1],2)
            
            if circle_state == True or circle_state2==True:
                x_c, y_c, r_c = params
                pygame.draw.circle(environment.infomap, CIRCLE_COLOR ,(x_c,y_c),r_c,2)

            # print(f"Alfa {m}  beta {c}")

            #draw square
            if person:
                for n in range(0, len(person)):
                    circle1, circle2, people = person[n]
                    x_c1, y_c1, r_c1 = circle1
                    x_c2, y_c2, r_c2 = circle2
                
                    if x_c1 < x_c2 :
                        left_rect = x_c1-r_c1-5
                    else:
                        left_rect = x_c2-r_c2-5
                    
                    if y_c1 < y_c2 :
                        top_rect = y_c1-r_c1-5
                    else:
                        top_rect = y_c2-r_c2-5
                    
                    width_rect = r_c1 + abs(x_c2 - x_c1) + r_c2 + 10
                    height_rect = r_c1 + abs(y_c2 - y_c1) + r_c2 + 10
                    pygame.draw.rect(environment.infomap, RECT_COLOR,(left_rect, top_rect, width_rect, height_rect), 2)
                    # not using text because it takes too long for program to run

           
            environment.dataStorage(sensor_data)
            environment.show_sensorData(environment.infomap)



    environment.map.blit(environment.infomap,(0,0))
    pygame.display.update()
# plot_line(error_jarak)
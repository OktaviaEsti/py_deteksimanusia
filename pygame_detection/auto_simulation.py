import env, sensors , feature
import random
import pygame
import math
import numpy as np

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

    if sensorON:

        if position[0]<1210: #posisi target
            # robot_step = np.array([0,0])        #perpindahan y
            # position = np.array([330, 552]) 
            robot_step = np.array([10,0])        #perpindahan x
        else:
            robot_step = np.array([0,0]) 

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
                if(params[2]<10):
                    FeatureMap.CIRCLES_DETECTED.append(params)
                END_POINT[0] = FeatureMap.projection_point2circle(OUTERMOST[0], params)
                END_POINT[1] = FeatureMap.projection_point2circle(OUTERMOST[1], params)

         #    OBJECT CLASSIFICATION
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
                        n_counter.append(n)
                        n_counter.append(n+1)

            COLOR = random_color()
            CIRCLE_COLOR = random_color()
            RECT_COLOR = (205,92,92)
       
       
            for point in line_seg:
                environment.infomap.set_at((int(point[0][0]),int(point[0][1])), (0,255,0))
                pygame.draw.circle(environment.infomap,COLOR,(int(point[0][0]),int(point[0][1])),2,0)
            
                
            # draw robot position:
            # externalMap= pygame.image.load('map/map_autotest.png')
            # environment.infomap.blit(externalMap,(0,0))
            pygame.draw.circle(environment.infomap, (0,0,255) ,laser.position,8) # draw robot pos
            
            if line_state == True:
                pygame.draw.line(environment.infomap,  (0,0,255) ,END_POINT[0], END_POINT[1],2)
            
            if circle_state == True or circle_state2==True:
                x_c, y_c, r_c = params
                pygame.draw.circle(environment.infomap, CIRCLE_COLOR ,(x_c,y_c),r_c,2)

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

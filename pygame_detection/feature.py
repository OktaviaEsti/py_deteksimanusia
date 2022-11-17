from ctypes import pointer
import math
from fractions import Fraction
from re import X
from turtle import distance

import numpy as np
from scipy.odr import *

import matplotlib.pyplot as plt


class featuresDetection:
    # Class variables
    res_line = 0


    def __init__(self):
        self.EPSILON = 10
        self.DELTA = 8
        self.DELTA_CIRCLE = 10
        self.EPSILON_CIRCLE = 20
        self.SNUM = 6
        self.SNUM_CIRCLE = 5
        self.PMIN = 15
        self.PMIN_CIRCLE = 6
        self.GMAX = 10 #30
        self.SEED_SEGMENTS = []
        self.LINE_SEGMENTS = []
        self.CIRCLE_SEGMENTS = []
        self.CIRCLES_DETECTED = []
        self.LASERPOINTS = []
        self.LINE_PARAMS = None
        self.CIRCLE_PARAMS = None
        self.NP = len(self.LASERPOINTS) - 1
        self.LMIN = 1
        self.LR = 0
        self.PR = 0
        self.res_line2 = 0
        self.res_circle2 = 0
        self.LEN_LINE_SEGMENTS = 0
        self.LEN_CIRCLE_SEGMENTS = 0
        self.dumbvar = 0

    def distpoint2point(self, point1, point2):
        Px = (point1[0] - point2[0]) ** 2
        Py = (point1[1] - point2[1]) ** 2
        return math.sqrt(Px + Py)

    def distpoint21ine(self, params, point):
        A, B, C = params

        distance = abs(A * point[0] + B * point[1] + C) / math.sqrt(A ** 2 + B ** 2)
        return distance
   
    def line2points(self, m, b):
        x = 5
        y = m * x + b

        x2 = 2000
        y2 = m * x2 + b
        return [(x, y), (x2, y2)]

    def lineForm_G2SI(self, A, B, C):
        m = -A / B
        B = - C / B
        return m, B

    # slope-intercept to general form
    def lineForm_si2G(self, m, B):
        A, B, C = - m, 1, -B
        if A < 0:
            A, B, C = -A, -B, -C
        den_a = Fraction(A).limit_denominator(1000).as_integer_ratio()[1]
        den_c = Fraction(C).limit_denominator(1000).as_integer_ratio()[1]

        gcd = np.gcd(den_a, den_c)
        lcm = den_a * den_c / gcd

        A = A * lcm
        B = B * lcm
        C = C * lcm
        return [A, B, C]

    def line_intersect_general(self, params1, params2):

        a1, b1, c1 = params1
        a2, b2, c2 = params2
        x = (c1 * b2 - b1 * c2) / (b1 * a2 - a1 * b2)
        y = (a1 * c2 - a2 * c1)/(b1 * a2 - a1 * b2)
        return x, y

    def points_2Line(self, point1, point2):
        m, b = 0, 0
        if point2[0] == point1[0]:
            pass
        else:
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
            b = point2[1] - m * point2[0]
        return m, b

    def projection_point2line(self, point, params):
        m, b = params
        x, y = point
        m2 = -1 / m
        c2 = y - m2 * x
        intersection_x = - (b - c2) / (m - m2)
        intersection_y = m2 * intersection_x + c2
        return intersection_x, intersection_y

    def AD2pos(self, distance, angle, robot_position):
        x = distance * math.cos(angle) + robot_position[0]
        y = -distance * math.sin(angle) + robot_position[1]
        return int(x), int(y)

    def laser_points_set(self, data):
        self.LASERPOINTS = []
        if not data:
            pass
        else:
            for point in data:
                coordinates = self.AD2pos(point[0], point[1], point[2]) # distance, angle, robot_position 
                self.LASERPOINTS.append([coordinates, point[1]])        # data disimpan
        self.NP = len(self.LASERPOINTS) - 1


    def liniar_func(self, p, x):
        m, b = p
        return m * x + b

    def odr_fit(self, laser_points):
        x = np.array([i[0][0] for i in laser_points])
        y = np.array([i[0][1] for i in laser_points])
        linar_model = Model(self.liniar_func)
        data = RealData(x, y)

        odr_model = ODR(data, linar_model, beta0=[0., 0.])

        out = odr_model.run()
        self.res_line = out.sum_square
        m, b = out.beta
        return m, b


    ## CIRCLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE!!!!!!!!!!

    def return_circle(self, c):
        x_c = c[1] / 2
        y_c = c[2] / 2
        r = c[0] + x_c ** 2 + y_c ** 2
        return x_c, y_c, np.sqrt(r)

    def circle_fit(self, laser_points):
        # y, x = pts[:, 0], pts[:, 1]
        x = np.array([i[0][0] for i in laser_points])
        y = np.array([i[0][1] for i in laser_points])
        N = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        u = x - x_mean
        v = y - y_mean
        # 圆心未知
        S_uuu = np.sum(u**3)
        S_vvv = np.sum(v**3)

        S_uu = np.sum(u ** 2)
        S_vv = np.sum(v ** 2)
        S_uv = np.sum(u * v)

        S_uvv = np.sum(u*(v**2))
        S_uuv = np.sum((u**2)*v)


        u_c = (S_uuu*S_vv + S_uvv*S_vv - S_vvv*S_uv - S_uuv*S_uv)/(2*(S_uu * S_vv - (S_uv)**2))
        v_c = (S_uu*S_vvv + S_uuv*S_uu - S_uuu*S_uv - S_uvv*S_uv)/(2*(S_uu * S_vv - (S_uv)**2))

        x_c = u_c + x_mean
        y_c = v_c + y_mean

        R = np.sum(np.sqrt((u - u_c)**2 + (v - v_c)**2)) / N

        # Calcul des distances au centre (xc_1, yc_1)
        Ri_1     = np.sqrt((x-x_c)**2 + (y-y_c)**2)
        # residu2_2  = sum((Ri_2**2-R_2**2)**2)
        self.res_circle = np.sum((Ri_1-R)**2)
        # self.res_circle =np.sum((Ri_1-R))
        return [x_c, y_c, R]



    def distpoint2circle(self, circle_params, point):
        x_c, y_c, r_c = circle_params
        x, y = point
        dist =np.sqrt(abs((x-x_c)**2+(y-y_c)**2))-r_c
        return dist
    
    def projection_point2circle(self, point, circle_params):
        x, y = point
        x_c, y_c, r_c = circle_params
        delta_x = x-x_c
        delta_y = y-y_c
        theta = math.atan(delta_y/delta_x)
        intersection_x = x_c + (r_c*math.cos(theta))
        intersection_y = y_c + (r_c*math.sin(theta))
        return intersection_x, intersection_y
    
    def distpoint2point_incircle(self, params, first_point, next_point):      #Belum dipakee
        x1, y1 = first_point
        x2, y2 = next_point
        x_c, y_c, r_c = params
        def slope(x1, y1, x2, y2): # Line slope given two points:
            return (y2-y1)/(x2-x1)

        def angle(s1, s2): 
            return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

        slope1 = slope(x1, y1, x_c, y_c)
        slope2 = slope(x2, y2, x_c, y_c)

        ang = angle(slope1, slope2)
        # print('Angle in degrees = ', ang)
        
        dist = (ang/360)*2*math.pi*r_c
        return dist
    
    def dist_twocircles(self, params1, params2):
        x_c1, y_c1, r_c1 = params1
        x_c2, y_c2, r_c2 = params2
        point1 = [x_c1, y_c1]
        point2 = [x_c2, y_c2]

        outer_point1 = self.projection_point2circle(point2, params1)
        outer_point2 = self.projection_point2circle(point1, params2)
        dist = self.distpoint2point(outer_point1, outer_point2)
        return dist
    

    # END CIRCLEEEEEEEEEEEEEEEEEEE

    def predictionPoint(self, line_params, sensed_point, robotpos):
        m, b = self.points_2Line(robotpos, sensed_point)
        params1 = self.lineForm_si2G(m, b)
        predx, predy = self.line_intersect_general(params1, line_params)
        return predx, predy

    def seed_segment_detection(self, robot_position, break_point_ind):
        flag = True
        self.NP = max(0, self.NP)
        self.SEED_SEGMENTS = []
        for i in range(break_point_ind, (self.NP - self.PMIN)):
            prediction_point_to_draw = []
            j = i + self.SNUM
            m, c = self.odr_fit(self.LASERPOINTS[i:j])
            params = self.lineForm_si2G(m, c)

            for k in range(i, j):
                flag = True
                prediction_point = self.predictionPoint(params, self.LASERPOINTS[k][0], robot_position)
                prediction_point_to_draw.append(prediction_point)
                dl = self.distpoint2point(prediction_point, self.LASERPOINTS[k][0])
              
                if dl > self.DELTA:
                    flag = False
                    break

                if k<(j-1): # ilangin titk jarak jauh
                    dl_2 = self.distpoint2point(self.LASERPOINTS[k+1][0], self.LASERPOINTS[k][0]) 
                    if dl_2 > self.DELTA:
                        flag = False
                        break

                d2 = self.distpoint21ine(params, prediction_point)

                if d2 > self.EPSILON:
                    flag = False
                    break

                # if (k<j-1) and (self.distpoint2point(self.LASERPOINTS[k][0], self.LASERPOINTS[k+1][0]) > self.GMAX):
                #     flag = False
                #     break
                
            if flag:
                self.LINE_PARAMS = params
                return [self.LASERPOINTS[i:j], prediction_point_to_draw, (i, j)]
        return False

    def circle_seed_segment_detection(self, robot_position, break_point_ind):
        flag = True
        self.NP = max(0, self.NP)
        self.SEED_SEGMENTS = []
        for i in range(break_point_ind, (self.NP - self.PMIN_CIRCLE)):
            prediction_point_to_draw = []
            j = i + self.SNUM_CIRCLE
            circle_params = self.circle_fit(self.LASERPOINTS[i:j])
            for k in range(i,j):
                flag = True
                self.break_point_backup = k
                prediction_point_to_draw = robot_position # dumb
                if k == 0:
                    continue
                dl = self.distpoint2point(self.LASERPOINTS[k][0], self.LASERPOINTS[k-1][0])
              
                if dl > self.DELTA_CIRCLE: 
                    # print(dl,(k-i), "hai",k)
                    flag = False
                    break

                # d2 = self.distpoint2circle(circle_params, self.LASERPOINTS[k][0])
                # if d2 > self.EPSILON_CIRCLE:
                #     break

                # d3 = self.distpoint2point_incircle(circle_params, self.LASERPOINTS[k][0], self.LASERPOINTS[k-1][0])
                # if d3 > (circle_params[2]*0.25):
                #     flag = False
                #     break
            if flag:
                self.CIRCLE_PARAMS = circle_params
                return [self.LASERPOINTS[i:j], (self.LASERPOINTS[i][0],self.LASERPOINTS[j][0]), (i, j)]
        return False


    def seed_segment_growing(self, indices, break_point):
        line_eq = self.LINE_PARAMS
        i, j = indices
        PB, PF = max(break_point, i - 1), min(j + 1, len(self.LASERPOINTS) - 1)
        while self.distpoint21ine(line_eq, self.LASERPOINTS[PF][0]) < self.EPSILON:
            if PF > self.NP - 1:
                break
            elif self.distpoint2point(self.LASERPOINTS[PF][0], self.LASERPOINTS[PF-1][0]) > self.GMAX:
                break
            else:
                m, b = self.odr_fit(self.LASERPOINTS[PB:PF])
                line_eq = self.lineForm_si2G(m, b)
                POINT = self.LASERPOINTS[PF][0]
            PF = PF + 1
            NEXTPOINT = self.LASERPOINTS[PF][0]
            self.test_p_np = [self.distpoint2point(POINT, NEXTPOINT), POINT, NEXTPOINT]
            
        PF = PF - 1

        while self.distpoint21ine(line_eq, self.LASERPOINTS[PB][0]) < self.EPSILON:
            if PB < break_point:
                break
            elif self.distpoint2point(self.LASERPOINTS[PB][0], self.LASERPOINTS[PB+1][0]) > self.GMAX:
                break
            else:
                m, b = self.odr_fit(self.LASERPOINTS[PB:PF])
                line_eq = self.lineForm_si2G(m, b)
                POINT = self.LASERPOINTS[PB][0]
            PB = PB - 1
            NEXTPOINT = self.LASERPOINTS[PB][0]
            self.test_p_np2 = [self.distpoint2point(POINT, NEXTPOINT), POINT, NEXTPOINT]
            
        PB = PB + 1
        LR = self.distpoint2point(self.LASERPOINTS[PB][0], self.LASERPOINTS[PF][0])
        PR = len(self.LASERPOINTS[PB:PF])

        if (LR >= self.LMIN) and (PR >= self.PMIN):
            self.LINE_PARAMS = line_eq
            m, b = self.lineForm_G2SI(line_eq[0], line_eq[1], line_eq[2])
            self.two_points = self.line2points(m, b)
            self.LINE_SEGMENTS.append((self.LASERPOINTS[PB + 1][0], self.LASERPOINTS[PF - 1][0]))
            self.LEN_LINE_SEGMENTS = self.distpoint2point(self.LASERPOINTS[PB + 1][0],self.LASERPOINTS[PF - 1][0])
            return [self.LASERPOINTS[PB:PF], self.two_points,
                    (self.LASERPOINTS[PB + 1][0], self.LASERPOINTS[PF - 1][0]), PF, (m, b), line_eq]
        else:
            return False
   
   
    def circle_seed_segment_growing(self, indices, break_point):
        circle_eq = self.CIRCLE_PARAMS
        i, j = indices
        PB, PF = max(break_point, i - 1), min(j + 1, len(self.LASERPOINTS) - 1)
        while PF < (self.NP - 1):
            # print("PF", PF, self.distpoint2point(self.LASERPOINTS[PF+1][0], self.LASERPOINTS[PF][0]))
            if self.distpoint2point(self.LASERPOINTS[PF-1][0], self.LASERPOINTS[PF][0]) < self.DELTA_CIRCLE:
                if self.distpoint2circle(circle_eq, self.LASERPOINTS[PF][0]) < self.EPSILON_CIRCLE:
                    circle_eq = self.circle_fit(self.LASERPOINTS[PB:PF])
                    PF = PF + 1
                    # if self.distpoint2point_incircle(circle_eq, self.LASERPOINTS[PF-1][0], self.LASERPOINTS[PF][0])>(circle_eq[2]*0.25):
                    #     PF = PF - 1
                    #     break
                else:
                  break
            else:
                # print("delta",self.distpoint2point(self.LASERPOINTS[PF+1][0], self.LASERPOINTS[PF][0]))
                break
        PF = PF - 1

        while PB > break_point:
            # print("PB", PB, self.distpoint2point(self.LASERPOINTS[PB-1][0], self.LASERPOINTS[PB][0]))
            if self.distpoint2point(self.LASERPOINTS[PB+1][0], self.LASERPOINTS[PB][0]) < self.DELTA_CIRCLE:
                if self.distpoint2circle(circle_eq, self.LASERPOINTS[PB][0]) < self.EPSILON_CIRCLE:
                    circle_eq = self.circle_fit(self.LASERPOINTS[PB:PF])
                    PB = PB - 1
                else:
                    break
            else:
                break
        PB = PB + 1

        LR = self.distpoint2point(self.LASERPOINTS[PB][0], self.LASERPOINTS[PF][0])
        PR = len(self.LASERPOINTS[PB:PF])

        if (LR >= self.LMIN) and (PR >= self.PMIN_CIRCLE):
            self.CIRCLE_PARAMS = circle_eq
            self.two_points = 1000 # dumb var
            self.CIRCLE_SEGMENTS.append((self.LASERPOINTS[PB + 1][0], self.LASERPOINTS[PF - 1][0]))
            return [self.LASERPOINTS[PB:PF], self.two_points,
                    (self.LASERPOINTS[PB + 1][0], self.LASERPOINTS[PF - 1][0]), PF, circle_eq]
        else:
            return False


        

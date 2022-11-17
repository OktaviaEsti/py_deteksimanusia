import numpy as np
import math

import feature

FeatureMap = feature.featuresDetection()


seed =np.array([[(133, 354), 3.4243359924128747], [(157, 349), 3.4400439556808236], [(159, 350), 3.455751918948773], [(162, 351), 3.4714598822167217], [(162, 352), 3.4871678454846706], [(163, 353), 3.5028758087526195], [(163, 355), 3.518583772020569], [(162, 357), 3.5342917352885177], [(161, 359), 3.5499996985564666], [(326, 359), 5.796238445873168], [(325, 357), 5.811946409141118], [(326, 355), 5.827654372409067], [(326, 354), 5.843362335677016], [(328, 354), 5.859070298944965]])
seed = seed[:,0]

def distpoint2point( point1, point2):
    Px = (point1[0] - point2[0]) ** 2
    Py = (point1[1] - point2[1]) ** 2
    return math.sqrt(Px + Py)
distt = []
for po in range(0, len(seed)):
    if po == (len(seed)-1):
        break
    # print(seed[po])
    jarak= distpoint2point(seed[po], seed[po+1])
    if(jarak == 165.0):
        print("masalah: ",seed[po], seed[po+1])
    distt.append(jarak)

print(np.max(distt))
    



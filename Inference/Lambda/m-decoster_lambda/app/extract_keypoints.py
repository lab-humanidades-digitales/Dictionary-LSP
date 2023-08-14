

def openPoseDict(xP, yP, xLH, yLH, xRH, yRH, xF, yF):
    
    zP = [0.0 for _ in range(len(xP))]
    zLH = [0.0 for _ in range(len(xLH))]
    zRH = [0.0 for _ in range(len(xRH))]
    zF = [0.0 for _ in range(len(xF))]

    opd = {"version":1.2,
     "people":[
         {"pose_keypoints_2d":[item for sublist in zip(xP, yP, zP) for item in sublist][:-1],
          "face_keypoints_2d":[item for sublist in zip(xF, yF, zF) for item in sublist][:-1],
          "hand_left_keypoints_2d":[item for sublist in zip(xLH, yLH, zLH) for item in sublist][:-1],
          "hand_right_keypoints_2d":[item for sublist in zip(xRH, yRH, zRH) for item in sublist][:-1],
          "pose_keypoints_3d":[],
          "face_keypoints_3d":[],
          "hand_left_keypoints_3d":[],
          "hand_right_keypoints_3d":[]
          }
         ]
     }
   
    return opd

def extratXYFromBodyPart(fileData, bodyName, exclusivePoints=[]):

    if exclusivePoints: #256.0
        x = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["x"]) if pos in exclusivePoints]
        y = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["y"]) if pos in exclusivePoints]
    else:

        x = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["x"])]
        y = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["y"])]

    return x, y

def keypointsFormat(fileData, bodyPart):
    xP, yP, xLH, yLH, xRH, yRH, xF, yF = [], [], [], [], [], [], [], []
    for bodyName in bodyPart:

        if(bodyName == "pose"):
            xP, yP = extratXYFromBodyPart(fileData,"pose")
        elif(bodyName == "hands"):
            xLH, yLH = extratXYFromBodyPart(fileData,"left_hand")
            xRH, yRH = extratXYFromBodyPart(fileData,"right_hand")

        elif(bodyName == "face"):

            nose_points = [1,5,6,218,438]
            mouth_points = [78,191,80,81,82,13,312,311,310,415,308,
                            95,88,178,87,14,317,402,318,324,
                            61,185,40,39,37,0,267,269,270,409,291,
                            146,91,181,84,17,314,405,321,375]
            #mouth_points = [0,37,39,40,61,185,267,269,270,291,409, 
            #                12,38,41,42,62,183,268,271,272,292,407,
            #                15,86,89,96,179,316,319,325,403,
            #                17,84,91,146,181,314,321,375,405]
            left_eyes_points = [33,133,157,158,159,160,161,173,246,
                                7,144,145,153,154,155,163]
            left_eyebrow_points = [63,66,70,105,107]
                                   #46,52,53,55,65]
            right_eyes_points = [263,362,384,385,386,387,388,398,466,
                                 249,373,374,380,381,382,390]
            right_eyebrow_points = [293,296,300,334,336]
                                    #276,282,283,285,295]
  
            #There are 97 points
            exclusivePoints = nose_points
            exclusivePoints = exclusivePoints + mouth_points
            exclusivePoints = exclusivePoints + left_eyes_points
            exclusivePoints = exclusivePoints + left_eyebrow_points
            exclusivePoints = exclusivePoints + right_eyes_points
            exclusivePoints = exclusivePoints + right_eyebrow_points
            
            xF, yF = extratXYFromBodyPart(fileData,"face",exclusivePoints)
    
    opd = openPoseDict(xP, yP, xLH, yLH, xRH, yRH, xF, yF)
    #print(opd)
    return opd

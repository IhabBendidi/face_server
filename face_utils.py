from pathlib import Path
import os
import pickle
import datetime
import cv2
import dlib




# Hog method face detector
global hog_face_detector
hog_face_detector = dlib.get_frontal_face_detector()

# Face landmarks predictor model
predictor_68_point_model =  "models/shape_predictor_68_face_landmarks.dat"
global pose_predictor_68_point
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

# face encoding model
face_recognition_model =  "models/dlib_face_recognition_resnet_model_v1.dat"
global face_encoder
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# encodings
global default_encodings
default_encodings = "encodings/encodings.pickle"
global encoding_data

def load_encodings(encodings=default_encodings):
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    try:
        data = pickle.loads(open(encodings, "rb").read())
    except EOFError :
        data = None
    return data





def write_encodings(encodings,file_path=default_encodings):
    print("[INFO] writing encodings...")
    f = open(file_path, "wb")
    data = {"time":datetime.datetime.now(),"data":encodings}
    f.write(pickle.dumps(data))
    f.close()




def update_encodings():
    encoding_data = load_encodings(default_encodings)






config = Path(default_encodings)
if config.is_file():
    encoding_data = load_encodings(default_encodings)
else :
    file = open(default_encodings, "x")
    encoding_data = load_encodings(default_encodings)

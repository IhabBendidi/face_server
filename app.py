import imutils
import dlib
import cv2
import os
import numpy as np
from imutils import paths
import classifier
from face_utils import *
import flask
from werkzeug.utils import secure_filename
from flask import request




ALLOWED_EXTENSIONS = {'jpg', 'png', 'mov', 'mp4'}

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "users"
app.config['TEMP_FOLDER'] = "temp"
app.config["DEBUG"] = True



def detection_method(method):
	if method == "hog":
		face_detector = hog_face_detector
	else :
		face_detector = None
	return face_detector




def process_raw_images(imagePaths,method="hog"):
	# loop over the image paths
	processedImages = []
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		# load the input image and convert it from BGR (OpenCV ordering)
		# to dlib ordering (RGB) or GRAY coloring for haar

		image = cv2.imread(imagePath)
		if method == "haar":
			processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		data = {"category" : name, "raw":image,"image" : processed_image}
		processedImages.append(data)
	return processedImages

# 1
def get_images(folder,method="hog"):
	imagePaths = list(paths.list_images(folder))
	processedImages = process_raw_images(imagePaths,method)
	return processedImages


def detect_face_boxes_training(img,method="hog"):
	face_detector = detection_method(method)
	boxes = []

	raw_face_locations = face_detector(img["image"], 1)

	for face in raw_face_locations :
		rect_to_css = face.top(), face.right(), face.bottom(), face.left() # this is just for HOG, do it for the other methods too
		boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], img["image"].shape[1]), min(rect_to_css[2], img["image"].shape[0]), max(rect_to_css[3], 0)))

	return boxes


def training_face_detection(images,method="hog"):
	detected_images = []
	for image in images :
		boxes = detect_face_boxes_training(image,method)
		for box in boxes:
			detected_image = {"category" : image["category"], "raw":image["raw"],"box" : box}
			detected_images.append(detected_image)
	return detected_images


# 3
def detect_landmarks_training(images):
	image_landmarks = []
	for i in range(0,len(images)):
		images[i]["box"] = dlib.rectangle(images[i]["box"][3], images[i]["box"][0], images[i]["box"][1], images[i]["box"][2])
		images[i]["raw"] = cv2.cvtColor(images[i]["raw"], cv2.COLOR_BGR2RGB)
		pose_predictor = pose_predictor_68_point

		raw_landmark = pose_predictor(images[i]["raw"], images[i]["box"])
		data = {"category":images[i]["category"],"raw":images[i]["raw"],"landmark":raw_landmark}
		image_landmarks.append(data)
	return image_landmarks

# 4
def encode_training_faces(images):

	encodings=[]
	for image in images :

		encoding = np.array(face_encoder.compute_face_descriptor(image["raw"], image["landmark"],1))
		data = {"category":image["category"],"encoding":encoding}
		encodings.append(data)
	return encodings










#1
def preprocess(image,method="hog"):
	# load the input image and convert it from BGR to RGB

	img = cv2.imread(image)
	processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return processed_image




#2
def detect_face_boxes_prediction(img,method="hog"):
	face_detector = detection_method(method)
	boxes = []

	raw_face_locations = face_detector(img, 1)

	for face in raw_face_locations :
		rect_to_css = face.top(), face.right(), face.bottom(), face.left() # this is just for HOG, do it for the other methods too
		boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], img.shape[1]), min(rect_to_css[2], img.shape[0]), max(rect_to_css[3], 0)))

	return boxes






#3
def detect_landmarks_prediction(processed_image,boxes):
	boxes = [dlib.rectangle(box[3], box[0], box[1], box[2]) for box in boxes]
	pose_predictor = pose_predictor_68_point
	raw_landmarks = [pose_predictor(processed_image, box) for box in boxes]
	return raw_landmarks



#4
def encode_prediction(processed_image,raw_landmarks):
	encodings = [np.array(face_encoder.compute_face_descriptor(processed_image, raw_landmark_set,1)) for raw_landmark_set in raw_landmarks]
	return encodings


def _recognize_simple(encoding,datas):
	matches = []
	for data in datas["data"] :
		match = (classifier.face_distance(data["encoding"], encoding) <= 0.6)
		matches.append(match)
	name = "Unknown"
	precision = 1

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}
		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = datas["data"][i]["category"]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
		precision = counts.get(name,0)/len(matchedIdxs)
	response = {"category" : name,"precision":precision}
	return response


#5
def recognize(encodings, boxes,data):
	response = []
# loop over the facial embeddings
	for (box,encoding) in zip(boxes,encodings):
		category = _recognize_simple(encoding,data)
		prediction = {"category":category["category"],"precision":category["precision"],"box":box}
		response.append(prediction)
	return response



def train_face_model(folder,encoding_path=default_encodings,method ="hog"):
	processed_images = get_images(folder,method)
	detected_images = training_face_detection(processed_images,method)
	image_landmarks = detect_landmarks_training(detected_images)
	encodings = encode_training_faces(image_landmarks)
	write_encodings(encodings)
	update_encodings()
	print("End of training")



def predict_faces(image,method="hog",encoding_path=default_encodings):
	if encoding_path == default_encodings :
		data = encoding_data
	else :
		data = load_encodings(encoding_path)
	processed_image = preprocess(image,method)
	boxes = detect_face_boxes_prediction(processed_image,method)
	raw_landmarks = detect_landmarks_prediction(processed_image,boxes)
	encodings = encode_prediction(processed_image,raw_landmarks)
	response = recognize(encodings, boxes,data)
	return response















@app.route('/api/enroll', methods=['POST'])
def enroll():
	status = 200
	output = {}
	if 'cin_file' not in request.files:
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="CIN_FILE_NOT_SENT"
		return output,status
	cin_file = request.files['cin_file']
	if 'username' not  in request.args:
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="MISSING_ARGUMENTS"
		return output,status
	if cin_file.filename == '':
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="INEXISTANT_FILENAME"
		return output,status
	username = request.args.get('username')
	if username == '':
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="INEXISTANT_USERNAME"
		return output,status
	filename = secure_filename(cin_file.filename)
	filename = os.path.join(username, filename)
	try :
		os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], username))
	except :
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="USER_ALREADY_EXISTS"
		return output,status
	cin_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	try :
		#train_face_model(os.path.join(app.config['UPLOAD_FOLDER'], username))
		train_face_model(app.config['UPLOAD_FOLDER'])
		output['trained'] = "success"
		output['EnrollStatus'] = "APPROVED"
		output['DecisionReason']="ENROLLED_AS_USER"
	except :
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="ERROR_DETECTED"
	return output,status


@app.route('/api/authentificate', methods=['POST'])
def authentificate():
	status = 200
	output = {}
	if 'auth_video' not in request.files:
		output['score'] = None
		output['Decision'] = "REJECTED"
		output['DecisionReason']="AUTHENTIFICATION_VIDEO_NOT_SENT"
		return output,status
	auth_video = request.files['auth_video']
	if 'username' not  in request.args:
		output['score'] = None
		output['Decision'] = "REJECTED"
		output['DecisionReason']="MISSING_ARGUMENTS"
		return output,status
	if auth_video.filename == '':
		output['score'] = None
		output['Decision'] = "REJECTED"
		output['DecisionReason']="INEXISTANT_FILENAME"
		return output,status
	username = request.args.get('username')
	if username == '':
		output['score'] = None
		output['Decision'] = "REJECTED"
		output['DecisionReason']="INEXISTANT_USERNAME"
		return output,status
	filename = secure_filename(auth_video.filename)
	auth_video.save(os.path.join(app.config['TEMP_FOLDER'], filename))
	try :
		response = predict_faces(os.path.join(app.config['TEMP_FOLDER'], filename))
		print(response)
		output['score'] = response[0]["precision"]
		print(response[0]["category"])
		if username == response[0]["category"]:
			output['Decision'] = "APPROVED"
			output['DecisionReason']="AUTHENTIFICATION_SUCCESS"
		else :
			output['score'] = - output['score']
			output['Decision'] = "REJECTED"
			output['DecisionReason']="BIOMETRIC_MISMATCH"
	except :
		output['score'] = None
		output['Decision'] = "REJECTED"
		output['DecisionReason']="ERROR_DETECTED"
	os.remove(os.path.join(app.config['TEMP_FOLDER'], filename))
	return output,status


app.run()

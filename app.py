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
from keras.preprocessing.image import img_to_array
import time
from keras import backend as K
import tensorflow as tf
import shutil

quota_bar = 0.95
print("[INFO] loading liveness detector...")

#model = load_model("models/liveness.model")
#model._make_predict_function()
le_live = pickle.loads(open("encodings/le.pickle", "rb").read())

ALLOWED_EXTENSIONS = {'jpg', 'png', 'mov', 'mp4'}

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "users"
app.config['TEMP_FOLDER'] = "temp"
app.config["DEBUG"] = False



def recognize_liveness(frame,quota_bar):
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	with tf.Graph().as_default():
		with tf.Session() as sess:
			K.set_session(sess)
			model = load_model("models/liveness.model")
			detections = net.forward()
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections
				if confidence > 0.3:
					# compute the (x, y)-coordinates of the bounding box for
					# the face and extract the face ROI
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# ensure the detected bounding box does fall outside the
					# dimensions of the frame
					startX = max(0, startX)
					startY = max(0, startY)
					endX = min(w, endX)
					endY = min(h, endY)

					# extract the face ROI and then preproces it in the exact
					# same manner as our training data
					face = frame[startY:endY, startX:endX]
					face = cv2.resize(face, (32, 32))
					face = face.astype("float") / 255.0
					face = img_to_array(face)
					face = np.expand_dims(face, axis=0)

					# pass the face ROI through the trained liveness detector
					# model to determine if the face is "real" or "fake"
					preds = model.predict(face)[0]
					j = np.argmax(preds)
					label = le_live.classes_[j]
					return label


def get_video_liveness(video_path,quota_bar):
	vidcap = cv2.VideoCapture(video_path)
	success,image = vidcap.read()
	count = 0
	tester = 0
	total_labels = []
	real_labels = []
	returned_frame = image
	while success :
		count += 1
		success,frame = vidcap.read()
		if count >= 10 and success:
			label = recognize_liveness(frame,quota_bar)
			print(str(label))
			total_labels.append(label)
			if label is not None :
				tester += 1
				if label.decode("utf-8") == 'real':
					real_labels.append(label)
					returned_frame = frame
			count=0


	quota = len(real_labels) / len(total_labels)
	print(quota)
	if quota > quota_bar:
		print(True)
		return True,returned_frame
	else :
		print(False)
		return False,returned_frame




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
		#print(counts)
		name_count = counts.get(name,0)


		precision = counts.get(name,0)/len(matchedIdxs)
	names = []
	for temp_name, counter in counts.items():
		if counter == name_count:
			names.append(temp_name)
	response = {"category" : names,"precision":precision}
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
	#if encoding_path == default_encodings :
		#data = encoding_data
	#else :
	data = load_encodings(encoding_path)
	processed_image = preprocess(image,method)
	boxes = detect_face_boxes_prediction(processed_image,method)

	raw_landmarks = detect_landmarks_prediction(processed_image,boxes)

	encodings = encode_prediction(processed_image,raw_landmarks)

	response = recognize(encodings, boxes,data)
	#print(response)


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
		update_encodings()
		output['trained'] = "success"
		output['EnrollStatus'] = "APPROVED"
		output['DecisionReason']="ENROLLED_AS_USER"
	except Exception as e:
		print(str(e))
		output['trained'] = "failed"
		output['EnrollStatus'] = "REJECTED"
		output['DecisionReason']="ERROR_DETECTED : " + str(e)
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

		# Work on video here
		(liveliness,frame) = get_video_liveness(os.path.join(app.config['TEMP_FOLDER'], filename),quota_bar)
		frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
		cv2.imwrite(os.path.join(app.config['TEMP_FOLDER'], "temp_frame.jpg"),frame)
		if liveliness :
			response = predict_faces(os.path.join(app.config['TEMP_FOLDER'], "temp_frame.jpg"))
			output['score'] = response[0]["precision"]
			#print(response[0]["category"])
			if username in response[0]["category"]:
				output['Decision'] = "APPROVED"
				output['DecisionReason']="AUTHENTIFICATION_SUCCESS"
			else :
				output['score'] = - output['score']
				output['Decision'] = "REJECTED"
				output['DecisionReason']="BIOMETRIC_MISMATCH"
		else :
			output['score'] = None
			output['Decision'] = "REJECTED"
			output['DecisionReason']="LIVELESS_FALSE"
	except Exception as e:
		print(str(e))
		output['score'] = None
		output['Decision'] = "REJECTED"
		output['DecisionReason']="ERROR_DETECTED : " + str(e)
	os.remove(os.path.join(app.config['TEMP_FOLDER'], filename))
	os.remove(os.path.join(app.config['TEMP_FOLDER'], "temp_frame.jpg"))
	return output,status

@app.route('/api/reset', methods=['POST'])
def reset_server():
	output = {}
	try :
		shutil.rmtree('users')
		os.mkdir('users')
		open("users/.placeholder", 'a').close()
		os.remove("encodings/encodings.pickle")
		open("encodings/encodings.pickle", 'a').close()
		output['code'] = 200
		output['msg'] = "DONE"
	except Exception as e :
		output['code'] = 500
		output['msg'] = "ERROR_DETECTED : " + str(e)
	return output

app.run(host= '0.0.0.0')

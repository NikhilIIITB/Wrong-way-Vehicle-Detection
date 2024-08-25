import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from PIL import Image
from math import sqrt
from collections import OrderedDict
from centroid import CentroidTracker


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", default = "Yolo/" ,required=False,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.6,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
LABELS = open(labelsPath).read().strip().split("\n")
counter = [0,0,0,0,0,0,0,0,0,0,0]
wrong_counter = 0;


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3_custom_final.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3_custom_test.cfg"])
ct = CentroidTracker()

print("loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("{} total frames in video".format(total))
except:
	print("could not determine # of frames in video")
	print("no approx. completion time can be provided")
	total = -1

previousCentroids = OrderedDict()

frame_count = 0
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	boxes = []
	confidences = []
	classIDs = []
	rects = []
	currentframe = 0

	for output in layerOutputs:
		for detection in output:


			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["confidence"]:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	if len(idxs) > 0:
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if classIDs[i] != 2:
				rects.append(boxes[i])
				#objects,ids,centroids= ct.update(rects)
				objects= ct.update(rects)

				#print("current-keys",objects.keys())
				#print("current-values",objects.values())
				#print("ids",ids)
				#print("previousCentroids",centroids)

				# loop over the tracked objects

				for (objectID, centroid) in objects.items():
					# draw both the ID of the object and the centroid of the
					# object on the output frame
					#text = "ID {}".format(objectID)
					text = " "

					'''if(len(previousCentroids) == len(objects.values())):
						if(len(previousCentroids > 1)):
							if(	class'''

					if len(previousCentroids) >= 1:
						if objectID in previousCentroids.keys():
							#print("previousCentroids[objectID]", previousCentroids[objectID])
							if ((sqrt((centroid[0] * centroid [0]) + (centroid[1] * centroid[1])) > 5 + sqrt((previousCentroids[objectID][0] * previousCentroids[objectID][0]) + (previousCentroids[objectID][1] * previousCentroids[objectID][1]))) & (centroid[0] > previousCentroids[objectID][0] )):
								counter[objectID] += 1;
								if(counter[objectID] == 40):
									#wrong_counter++
									frame_count = frame_count + 1
									Viname = 'Frame/frame'+str(frame_count)+'.jpg'
									print('Creating....', Viname)
									cv2.imwrite(Viname, frame)

									im = Image.open(Viname)
									print("Frame Captured")
									print("Croping the image")
									crop_img = 'Crop/crop'+str(frame_count)+'.jpg'
									im_res = im.crop((x,y,x+w,y+w))
									im_res.save(crop_img)
								#print(objectID , (sqrt((centroid[0] * centroid [0]) + (centroid[1] * centroid[1])) - sqrt((previousCentroids[objectID][0] * previousCentroids[objectID][0]) + (previousCentroids[objectID][1] * previousCentroids[objectID][1]))))
									text = "Wrong Way id {}".format(objectID)


				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			previousCentroids = objects.copy()
			#print("previousCentroids\t",previousCentroids)

			#text = "{}: {:.4f}".format(LABELS[classIDs[i]],
			#	confidences[i])
			#cv2.putText(frame, text, (x, y - 5),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("single frame took {:.4f} seconds".format(elap))
			print("estimated total time to finish: {:.4f}".format(
				elap * total))

	writer.write(frame)

writer.release()
vs.release()




print("Running Licence Plate Detector")
# Load Yolo
net = cv2.dnn.readNet("yolov3-licence_final.weights", "yolov3-licence.cfg")
classes = []
with open("licence.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[16]:

# import required module
import os
# assign directory
directory = 'Crop'

# iterate over files in
# that directory
licence_count = 0
for filename in os.listdir(directory):

    f = os.path.join(directory, filename)

    licence_count = licence_count +1
    # Loading image
    img = cv2.imread(f)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape


    # In[17]:


    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    # In[18]:


    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # In[19]:


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    # In[27]:

    count = 0
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)



            cv2.imwrite("img.jpg", img)
            im_res = Image.open("img.jpg")
            im = im_res.crop((x, y, x + w, y + h))
            name = 'Licence/licence' + str(licence_count) + '.jpg'
            im.save(name)
            count=count+1



            #cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


    cv2.imshow("Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

pipeline = keras_ocr.pipeline.Pipeline()


images = [
    keras_ocr.tools.read(img) for img in ['Crop/crop2.jpg','Crop/crop1.jpg'

    ]
]
print(np.shape(images))

len(images)

prediction_groups = pipeline.recognize(images)

plt.figure(figsize = (10,20))
plt.imshow(images[0])

plt.figure(figsize = (10,20))
plt.imshow(images[1])
'''



predicted_image_1 = prediction_groups[0]
for text, box in predicted_image_1:
    print(text)

predicted_image_2 = prediction_groups[1]
for text, box in predicted_image_2:
    print(text)


'''
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image,
                                    predictions=predictions,
                                    ax=ax)
plt.show()

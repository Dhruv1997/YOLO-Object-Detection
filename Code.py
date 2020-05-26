# Import the necessary packages & libraries
import numpy as np
import argparse
import imutils
import time
import cv2
import os

#------------------------------------------------

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#------------------------------------------------

# Load the COCO class labels (trained on the YOLO model)
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible Class Label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# Derive the paths to the YOLO weights and model Configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load our YOLO object detector trained on COCO dataset (80 classes)
# Determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#------------------------------------------------

# Initialize the video stream, pointer to output video file, and frame dimensions.
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Try to determine the total no. of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# If, an error occurred while trying to determine the total
# Then, number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

#------------------------------------------------

# Loop over frames from the video file stream
while True:
	# Read the next frame from the file
	(grabbed, frame) = vs.read()

	# If the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# If the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

#------------------------------------------------

	# Construct a blob from the input frame and then perform a forward pass of the
	# YOLO object detector, giving us our bounding boxes and associated probabilities.
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# Initialize our lists of detected bounding boxes, confidences, and class IDs.
	boxes = []
	confidences = []
	classIDs = []

#------------------------------------------------

	# Loop over each of the layer outputs
	for output in layerOutputs:
		# Loop over each of the detections
		for detection in output:
			# Extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# Filter out weak predictions by ensuring the detected
			# Probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# Scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# Use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# Update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

#------------------------------------------------

	# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# Ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# Extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# Draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#------------------------------------------------

	# Check if the video writer is None
	if writer is None:
		# Initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# Some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# Write the output frame to disk
	writer.write(frame)

# Release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

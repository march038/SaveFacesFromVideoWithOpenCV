import cv2 # for image an video processing
import os # for interacting with local files & directories via the operating system
import numpy as np # for numerical operations
import face_recognition # for face ddetection and recognition

# load annd open sample video in which we want to detect the 7 faces that show up in the video
video_path = "sample2.mp4"
cap = cv2.VideoCapture(video_path)

# specify a directory in which we want to save a picture for each detected face or create the directory if it doesn't exist yet
save_dir = "faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# let's define a list to save detected face's encodings into it
known_face_encodings = []

# let's also define a variable to count how many faces are detected in the image, this should result in 7 after executing the code
number_faces = 0

# loop through each frame in the video
while cap.isOpened():
    # read a single frame from the video to see if the code works, break the loop if no frame is found
    success, frame = cap.read()
    if not success:
        break
    
    # convert the image from BGR format for OpenCV to RGB format for face_recognition format
    rgb_frame =  np.ascontiguousarray(frame[:, :, ::-1])
    
    #  find all face locations and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # loop through each face found in the current frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # compare the face's encodings with known faces's encodings 
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.625)
        # calculate the distance to all known faces
        distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # if the face is new (no matches found or distance & greater than the tolerance threshold) add it to the set of faces
        if not any(matches) or min(distance, default=1) > 0.625:
            # add the new face's encoding to our known_face_encodings list
            known_face_encodings.append(face_encoding)
            # extract the coordinates of the face location
            top, right, bottom, left = face_location
            
            # crop the face from the frame using the coordinates from above and resize the image to 200x200
            face_image = frame[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, (200,200), interpolation=cv2.INTER_CUBIC)
            # sharpen the resized image using the cv2.filter2D function
            sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened_image = cv2.filter2D(face_image_resized, -1, sharpening_kernel)
            # save that cropped, resized and sharpened face image to the specified directory
            cv2.imwrite(os.path.join(save_dir, f"face_{number_faces}.jpg"), sharpened_image)
            # increase the number_faces variable by 1 for each new face detected
            number_faces += 1

# release the video capture object to free the computer's ressources
cap.release()

# print something when the script is done and return the number of faces detecte with the number_faces variable
print(f"Detection process completed. {number_faces} faces were identified and saved to the folder.")
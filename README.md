Hi!


This project leverages OpenCV and face_recognition to detect all faces in a video file (mp4) and save each etected face as an image in a specified local directory.
I recommend using conda for package management and suggest Visual Studio Code or PyCharm as the IDE for running OpenCV scripts. 
In my experience, and from what I've gathered from others, running OpenCV in a Jupyter Notebook via conda often leads to Python crashing because conda attempts to open Spyder for image/video visualization.

This code can be easily adapted for permanent applications such as webcam surveillance. In such scenarios, it's crucial to be aware of local data protection laws concerning the storage and handling of (face) image files.
It should be very easy to delete the image files after a certain time, GPT could be helpful for this ;)

As always, the code and necessary libraries are explained using comments inside the py.file :)

Have fun using this script!

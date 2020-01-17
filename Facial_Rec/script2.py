import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
jorge_image = face_recognition.load_image_file("Images/jorge.jpg")
jorge_face_encoding = face_recognition.face_encodings(jorge_image)[0]

lynn_image = face_recognition.load_image_file("Images/Lynn.png")
lynn_face_encoding = face_recognition.face_encodings(lynn_image)[0]

kellye_image = face_recognition.load_image_file("Images/Kellye2.png")
kellye_face_encoding = face_recognition.face_encodings(kellye_image)[0]

cynthia_image = face_recognition.load_image_file("Images/Cynthia.png")
cynthia_face_encoding = face_recognition.face_encodings(cynthia_image)[0]

manny_image = face_recognition.load_image_file("Images/Manny.png")
manny_face_encoding = face_recognition.face_encodings(manny_image)[0]

adam_image = face_recognition.load_image_file("Images/Adamk.jpg")
adam_face_encoding = face_recognition.face_encodings(adam_image)[0]

alex_image = face_recognition.load_image_file("Images/Alex.png")
alex_face_encoding = face_recognition.face_encodings(alex_image)[0]

alicokadar_image = face_recognition.load_image_file("Images/Alicokadar.png")
alicokadar_face_encoding = face_recognition.face_encodings(alicokadar_image)[0]

alli_image = face_recognition.load_image_file("Images/Alli.png")
alli_face_encoding = face_recognition.face_encodings(alli_image)[0]

amy_image = face_recognition.load_image_file("Images/Amy.jpg")
amy_face_encoding = face_recognition.face_encodings(amy_image)[0]

amyd_image = face_recognition.load_image_file("Images/AmyD.jpg")
amyd_face_encoding = face_recognition.face_encodings(amyd_image)[0]

anselmo_image = face_recognition.load_image_file("Images/Anselmo.jpg")
anselmo_face_encoding = face_recognition.face_encodings(anselmo_image)[0]

austin_image = face_recognition.load_image_file("Images/Austin.jpg")
austin_face_encoding = face_recognition.face_encodings(austin_image)[0]

bhavini_image = face_recognition.load_image_file("Images/Bhavini.png")
bhavini_face_encoding = face_recognition.face_encodings(bhavini_image)[0]

bobby_image = face_recognition.load_image_file("Images/Bobby.jpg")
bobby_face_encoding = face_recognition.face_encodings(bobby_image)[0]

brian_image = face_recognition.load_image_file("Images/Brian.jpg")
brian_face_encoding = face_recognition.face_encodings(brian_image)[0]

chris_image = face_recognition.load_image_file("Images/Chris.jpg")
chris_face_encoding = face_recognition.face_encodings(chris_image)[0]

danielle_image = face_recognition.load_image_file("Images/Danielle.jpg")
danielle_face_encoding = face_recognition.face_encodings(danielle_image)[0]

debra_image = face_recognition.load_image_file("Images/Debra.png")
debra_face_encoding = face_recognition.face_encodings(debra_image)[0]

dundar_image = face_recognition.load_image_file("Images/Dundar.png")
dundar_face_encoding = face_recognition.face_encodings(dundar_image)[0]

hari_image = face_recognition.load_image_file("Images/Hari.jpg")
hari_face_encoding = face_recognition.face_encodings(hari_image)[0]

hayley_image = face_recognition.load_image_file("Images/Hayley.png")
hayley_face_encoding = face_recognition.face_encodings(hayley_image)[0]

hazel_image = face_recognition.load_image_file("Images/Hazel.jpg")
hazel_face_encoding = face_recognition.face_encodings(hazel_image)[0]

jadd_image = face_recognition.load_image_file("Images/Jadd.jpg")
jadd_face_encoding = face_recognition.face_encodings(jadd_image)[0]

jainder_image = face_recognition.load_image_file("Images/Jainder.jpg")
jainder_face_encoding = face_recognition.face_encodings(jainder_image)[0]

janie_image = face_recognition.load_image_file("Images/Janie.jpg")
janie_face_encoding = face_recognition.face_encodings(janie_image)[0]

jennifer_image = face_recognition.load_image_file("Images/Jennifer.jpg")
jennifer_face_encoding = face_recognition.face_encodings(jennifer_image)[0]

jie_image = face_recognition.load_image_file("Images/Jie.png")
jie_face_encoding = face_recognition.face_encodings(jie_image)[0]

josefina_image = face_recognition.load_image_file("Images/Josefina.png")
josefina_face_encoding = face_recognition.face_encodings(josefina_image)[0]

joseph_image = face_recognition.load_image_file("Images/Joseph.jpg")
joseph_face_encoding = face_recognition.face_encodings(joseph_image)[0]

kevin_image = face_recognition.load_image_file("Images/Kevin.jpg")
kevin_face_encoding = face_recognition.face_encodings(kevin_image)[0]

manmita_image = face_recognition.load_image_file("Images/Manmita.jpg")
manmita_face_encoding = face_recognition.face_encodings(manmita_image)[0]

matthew_image = face_recognition.load_image_file("Images/Matthew.png")
matthew_face_encoding = face_recognition.face_encodings(matthew_image)[0]

michael_image = face_recognition.load_image_file("Images/Michael.jpg")
michael_face_encoding = face_recognition.face_encodings(michael_image)[0]

mingming_image = face_recognition.load_image_file("Images/Mingming.jpg")
mingming_face_encoding = face_recognition.face_encodings(mingming_image)[0]

ningze_image = face_recognition.load_image_file("Images/Ningze.png")
ningze_face_encoding = face_recognition.face_encodings(ningze_image)[0]

poonam_image = face_recognition.load_image_file("Images/Poonam.jpg")
poonam_face_encoding = face_recognition.face_encodings(poonam_image)[0]

rebekah_image = face_recognition.load_image_file("Images/Rebekah.jpg")
rebekah_face_encoding = face_recognition.face_encodings(rebekah_image)[0]

rutabah_image = face_recognition.load_image_file("Images/Rutabah.png")
rutabah_face_encoding = face_recognition.face_encodings(rutabah_image)[0]

sam_image = face_recognition.load_image_file("Images/Sam.jpg")
sam_face_encoding = face_recognition.face_encodings(sam_image)[0]

sarah_image = face_recognition.load_image_file("Images/Sarah.png")
sarah_face_encoding = face_recognition.face_encodings(sarah_image)[0]

steven_image = face_recognition.load_image_file("Images/Steven.jpg")
steven_face_encoding = face_recognition.face_encodings(steven_image)[0]

stevenH_image = face_recognition.load_image_file("Images/StevenH.jpg")
stevenH_face_encoding = face_recognition.face_encodings(stevenH_image)[0]

teresa_image = face_recognition.load_image_file("Images/Theresa.jpg")
teresa_face_encoding = face_recognition.face_encodings(teresa_image)[0]

thuria_image = face_recognition.load_image_file("Images/Thuria.jpg")
thuria_face_encoding = face_recognition.face_encodings(thuria_image)[0]

tristan_image = face_recognition.load_image_file("Images/Tristan.jpg")
tristan_face_encoding = face_recognition.face_encodings(tristan_image)[0]

upasna_image = face_recognition.load_image_file("Images/Upasna.jpg")
upasna_face_encoding = face_recognition.face_encodings(upasna_image)[0]

billy_image = face_recognition.load_image_file("Images/Billy.jpg")
billy_face_encoding = face_recognition.face_encodings(billy_image)[0]

# Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    jorge_face_encoding,
    lynn_face_encoding, 
    kellye_face_encoding,
    cynthia_face_encoding,
    manny_face_encoding,
    adam_face_encoding,
    alex_face_encoding,
    alicokadar_face_encoding,
    alli_face_encoding,
    amy_face_encoding,
    amyd_face_encoding,
    anselmo_face_encoding,
    austin_face_encoding,
    bhavini_face_encoding,
    bobby_face_encoding,
    brian_face_encoding,
    chris_face_encoding,
    danielle_face_encoding,
    debra_face_encoding,
    dundar_face_encoding,
    hari_face_encoding,
    hayley_face_encoding,
    hazel_face_encoding,
    jadd_face_encoding,
    jainder_face_encoding,
    janie_face_encoding,
    jennifer_face_encoding,
    jie_face_encoding,
    josefina_face_encoding,
    joseph_face_encoding,
    kevin_face_encoding,
    manmita_face_encoding,
    matthew_face_encoding,
    michael_face_encoding,
    mingming_face_encoding,
    ningze_face_encoding,
    poonam_face_encoding,
    rebekah_face_encoding,
    rutabah_face_encoding,
    sam_face_encoding,
    sarah_face_encoding,
    steven_face_encoding,
    stevenH_face_encoding,
    teresa_face_encoding,
    thuria_face_encoding,
    tristan_face_encoding,
    upasna_face_encoding,
    billy_face_encoding
]
known_face_names = [
    "Jorge A. Cavazos",
    "Lynn Leifker",
    "Kellye Rennell",
    "Cynthia Juarez", 
    "Manuel Lara",
    "Adam Keeling",
    "Alex Frame",
    "Alicokadar",
    "Alli Vaughn",
    "Amy Koldeway",
    "Amy Dach",
    "Anselmo Jr. Garza",
    "Austin Schein",
    "Bhavini Vyas",
    "Bobby Robles",
    "Brian McHugh",
    "Christopher Nguyen",
    "Danielle Bustillos SoRelle",
    "Debra Steinman",
    "Dundar Karabay",
    "Hari hara Chidambaram Muthiah",
    "Hayley Jellison",
    "Hazel Despain",
    "Jadd Cheng",
    "Jainder Soundararajan",
    "Janie Lua",
    "Jennifer Lawless",
    "Jie Bai",
    "Josefina Blanchard",
    "Joseph Grantham",
    "Kevin Clark",
    "Manmita",
    "Matthew Duncan",
    "Michael Ramirez",
    "Mingming Chen",
    "Ningze Sun",
    "Poonam Goel",
    "Rebekah Rowland",
    "Rutabah Khan",
    "Sam Bender",
    "Sarah Cross",
    "Steven Kroman",
    "Steven Holloway",
    "Theresa Carrino",
    "Thuria Abdelaziz",
    "Tristan Patrick Serigny",
    "Upasna Gautam",
    "Wenbin Billy Zhao"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
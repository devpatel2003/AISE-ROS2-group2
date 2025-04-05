import cv2

# load haar cascade files
car_cascade = cv2.CascadeClassifier('cars.xml')
pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
bike_cascade = cv2.CascadeClassifier('two_wheeler.xml')

# start video capture on default camera
cap = cv2.VideoCapture(0)

# set frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale (required for haar algorithm)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect obejcts
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    bikes = bike_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # draw bounding boxes and labels
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Pedestrian', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for (x, y, w, h) in bikes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Bike', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show frame
    cv2.imshow('Real-time Detection', frame)

    # allow user to break with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up
cap.release()
cv2.destroyAllWindows()

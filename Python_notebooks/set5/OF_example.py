import cv2
import numpy as np

# The video feed is read in as
# a VideoCapture object
cap = cv2.VideoCapture("/Users/tianze/cs4243_lab/CS4243_2023_images_small/10236.mp4")

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw optical flow vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Display the frame with optical flow vectorsqq
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
    cv2.imshow("Optical Flow", frame)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()

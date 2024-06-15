{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2ff9d0a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "cap = cv2.VideoCapture(0) \n",
    "ret, frame1 = cap.read()\n",
    "prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "bbox = (0, 0, 0, 0)\n",
    "\n",
    "while True:\n",
    "    ret, frame2 = cap.read()\n",
    "    if not ret:\n",
    "        break\n",
    "    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "    frame_diff = cv2.absdiff(prev_gray, gray)\n",
    "\n",
    "    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)\n",
    "\n",
    "    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "\n",
    "    for contour in contours:\n",
    "        x, y, w, h = cv2.boundingRect(contour)\n",
    "        area = cv2.contourArea(contour)\n",
    "\n",
    "        if area > 2000:\n",
    "            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "            bbox = (x, y, w, h)\n",
    "\n",
    "    cv2.imshow('Motion Detection', frame2)\n",
    "    prev_gray = gray\n",
    "    if cv2.waitKey(20) & 0xFF == ord('q'):\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5be4d9ea",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

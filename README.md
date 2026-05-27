## End Goal
Have an automated scoring system that can be used to assist referees in real matches

### Before
![demo](demo_gif.gif)

### Current
https://github.com/user-attachments/assets/85147811-d809-4bde-9c9b-d0e8675236f3

**Current progress:** Able to track and differentiate athletes in real-time along with their respective gloves and feet, as well as their heads and bodies
**Next Goal:** Collision detection and scoring logic

## Tools
*   **Model:** YOLOV8 (Ultralytics)
*   **Language:** Python 3.9.13
*   **Specs: 30FPS on Intel Integrated Graphics, i7-13700H**

## Setup
1. Create a virtual environment and activate it
2. pip install -r requirements.txt
3. Download modified dataset from https://drive.google.com/drive/folders/15uquvfbVMLcIIaBK5-x_I5-eAHs1quFz?usp=drive_link
*  To download trained model directly follow the link /Training Set 2/weights/best v2.pt
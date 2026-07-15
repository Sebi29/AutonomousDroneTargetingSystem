# Autonomous Drone Targeting System

## Descriere
Acest proiect implementează un sistem autonom de control și vizare pentru drone, dezvoltat în Python. Proiectul integrează simulatorul **Webots** pentru fizica și controlul zborului cu modelul de machine learning **YOLO** (Ultralytics) pentru detectarea, identificarea și urmărirea vehiculelor în timp real.

## Tehnologii utilizate
* **Limbaj:** Python
* **Simulare:** Webots (Robot Controller API)
* **Computer Vision:** YOLOv11 (Ultralytics), OpenCV
* **Procesare date:** NumPy

## Structura Proiectului
* `controllers/` - Conține scripturile de control pentru drona autonomă și pentru vehiculele din simulare.
* `worlds/` - Conține mediul 3D de simulare pentru Webots.

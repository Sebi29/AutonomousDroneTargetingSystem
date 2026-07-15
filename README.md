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

## Instalare și Configurare

**1. Clonarea repository-ului:**
Deschideți un terminal și rulați comanda:

```bash
git clone https://github.com/Sebi29/AutonomousDroneTargetingSystem.git(https://github.com/Sebi29/AutonomousDroneTargetingSystem.git)
```

**2. Instalarea dependențelor:**
Acest proiect folosește Python. Pentru a instala modulele necesare detectării și procesării imaginilor, rulați:

```bash
pip install ultralytics opencv-python numpy
```

**3. Mediul de Simulare:**
Descărcați și instalați simulatorul open-source [Webots](https://cyberbotics.com/).

## Mod de Rulare

1. Deschideți aplicația **Webots**.
2. Din meniul principal, accesați `File` -> `Open World...` și navigați către folderul `worlds/` al acestui proiect pentru a deschide fișierul `my_simulation.wbt`.
3. Apăsați butonul **Play** din interfața simulatorului Webots.
4. Controller-ul (`drone_brain.py`) va porni automat, va executa secvența de decolare pentru stabilizare, iar apoi va activa modelul YOLO pentru a detecta și urmări mașinile din simulare.

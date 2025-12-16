@ -1,2 +1,92 @@
# Gesture_controller
A project, used to play games without the physical touch with keyborad or mouse, everything is in air.
# 🖐️ Gesture Controller for PC Games  
A Python-based **hand-gesture controller** that lets you play PC games like **Subway Surfers** and **Hill Climb Racing** using your webcam.  
It uses **MediaPipe**, **OpenCV**, and **PyAutoGUI** to detect hand gestures and convert them into keyboard inputs.

---

## 🚀 Features
- Control games without a keyboard  
- Real-time gesture tracking using your webcam  
- Supports common game actions like:  
  - Jump  
  - Duck  
  - Move Left / Right  
  - Accelerate / Brake  
- Works with any game that uses keyboard controls  
- High accuracy hand-landmark detection (MediaPipe)

---

## 🎮 Game Controls (Gesture → Action)

### **Subway Surfers**
| Gesture | Action |
|--------|--------|
| Swipe Hand Up | Jump |
| Swipe Hand Down | Roll / Duck |
| Move Hand Left | Left |
| Move Hand Right | Right |

---

### **Hill Climb Racing**
| Gesture | Action |
|--------|--------|
| Fist | Brake |
| Open Palm | Accelerate |
| Tilt Hand Left | Lean Back |
| Tilt Hand Right | Lean Forward |

> ⚠️ You can edit these controls inside the code (PyAutoGUI key mappings).

---

## 🛠️ Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- PyAutoGUI  
- NumPy

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mohan-G-hub/Gesture_controller.git
cd Gesture_controller

2️⃣ Install Required Libraries
pip install opencv-python mediapipe numpy pyautogui

▶️ How to Run

1.Connect your webcam
2.Open the project folder
3.Run the script:
python main.py

4.Keep your hand in front of the webcam
5.Open the game (Subway Surfers / Hill Climb Racing)
6.Use gestures to play!

🔧 Configuration

You can change gesture actions inside gesture_controller.py:
pyautogui.press('up')     # Jump
pyautogui.press('down')   # Duck
pyautogui.press('left')   # Move Left
pyautogui.press('right')  # Move Right

🧠 How It Works

MediaPipe detects 21 hand landmarks
Distances + angles between fingers are calculated
Gestures are classified (open palm, fist, swipe, tilt, etc.)
Actions are mapped to keyboard keys using PyAutoGUI

📝 License

This project is open-source. Feel free to modify and improve it!
# Gesture_game_controller

"""Quick test to see if MediaPipe works"""
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    print("\n1. Importing cv2...")
    import cv2
    print("✓ OpenCV imported successfully")
    print("  OpenCV version:", cv2.__version__)
except Exception as e:
    print(f"❌ Error importing cv2: {e}")
    sys.exit(1)

try:
    print("\n2. Importing mediapipe...")
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    print("  MediaPipe version:", mp.__version__)
except Exception as e:
    print(f"❌ Error importing mediapipe: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    print("✓ MediaPipe Hands initialized successfully")
    hands.close()
except Exception as e:
    print(f"❌ Error initializing MediaPipe Hands: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! MediaPipe is working correctly.")

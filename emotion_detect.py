import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)

confidence_points = 0
stress_points = 0
total_frames = 0

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        total_frames += 1

        # Scoring logic
        if dominant_emotion == "happy":
            confidence_points += 2
        elif dominant_emotion == "neutral":
            confidence_points += 1
        elif dominant_emotion in ["angry", "sad", "disgust"]:
            stress_points += 2
        elif dominant_emotion == "fear":
            stress_points += 3

        # Calculate percentages
        confidence_percent = int((confidence_points / (total_frames * 2)) * 100)
        stress_percent = int((stress_points / (total_frames * 3)) * 100)

        # Display
        cv2.putText(frame, f'Emotion: {dominant_emotion}',
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Confidence: {confidence_percent}%',
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f'Stress: {stress_percent}%',
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    except:
        pass

    cv2.imshow("Interview Analysis System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- After Interview Ends ----

end_time = time.time()
duration = int(end_time - start_time)

if total_frames > 0:
    final_confidence = int((confidence_points / (total_frames * 2)) * 100)
    final_stress = int((stress_points / (total_frames * 3)) * 100)
else:
    final_confidence = 0
    final_stress = 0

# Save report
with open("session_report.txt", "w") as file:
    file.write("Interview Analysis Report\n")
    file.write("--------------------------\n")
    file.write(f"Session Duration: {duration} seconds\n")
    file.write(f"Final Confidence: {final_confidence}%\n")
    file.write(f"Final Stress: {final_stress}%\n")
    file.write(f"Total Frames Analyzed: {total_frames}\n")

print("\nSession report saved as session_report.txt")

cap.release()
cv2.destroyAllWindows()
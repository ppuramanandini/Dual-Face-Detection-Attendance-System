# Dual Face Detection Attendance System

A smart AI-powered attendance system that uses ID scanning + face detection to automatically mark attendance.  
This removes the need for manual attendance, ensuring accuracy, speed, and security.

## Features
- Two-step verification: Scans ID (QR code / barcode / typed ID) and verifies face.
- Prevents proxy attendance: Face must match the ID for attendance to be marked.
- Real-time logging: Attendance is recorded instantly into a CSV file.
- Offline operation: Works without internet.
- Easy deployment: Runs on any laptop or PC with a webcam.

## Technologies Used
- Language: Python
- Libraries: OpenCV, face_recognition, NumPy, Pandas, datetime
- Hardware: Standard webcam, ID card with QR/Barcode (or typed ID input)
- Version Control: Git and GitHub

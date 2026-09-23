import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        corners, ids, _ = detector.detectMarkers(frame)
        
        h, w = frame.shape[:2]
        focal = w
        mtx = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0,0,1]], dtype=float)
        dist = np.zeros((5,1))

        if ids is not None:
            s = 0.5
            h = 1
        
        obj_pts = np.float32([
            [-s, -s, 0],
            [s, -s, 0],
            [s, s, 0],
            [-s, s, 0],

            [-s, -s, h],
            [s, -s, h],
            [s, s, h],
            [-s, s, h],
        ])

        ref_3d = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)

        for i in range(len(ids)):

            _, rvec, tvec = cv2.slovePnP(ref_3d, corners[i], mtx, dist)

            imgpts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
            for j in range(4): cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255, 0, 0), 2)
            cv2.drawContours(frame, [imgpts[4]], -1, (0, 0, 255), 2)

        cv2.imshow("Realidade Aumentada", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

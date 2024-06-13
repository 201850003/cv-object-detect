import cv2
import os


class Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.MultiTracker_create()

    def initialize(self, frame, bboxes):
        for bbox in bboxes:
            self.tracker.add(cv2.legacy.TrackerKCF_create(), frame, tuple(bbox))

    def update(self, frame):
        success, boxes = self.tracker.update(frame)
        return boxes


def draw_boxes(frame, boxes):
    for box in boxes:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    return frame


def select_rois(frame):
    bboxes = cv2.selectROIs("Frame", frame, fromCenter=False)
    cv2.destroyWindow("Frame")
    return bboxes


def main():
    video_path = 'data/hw-4.mp4'
    output_folder = './q3'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    bboxes = select_rois(frame)
    if len(bboxes) == 0:
        print("No bounding boxes selected. Exiting...")
        return

    tracker = Tracker()
    tracker.initialize(frame, bboxes)

    specific_frames = [0, 14, 63, 99, 124, 183, 223, 255, 283, 351]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = tracker.update(frame)
        frame_with_boxes = draw_boxes(frame, boxes)

        if frame_count in specific_frames:
            output_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_path, frame_with_boxes)
            print(f"Saved frame {frame_count} to {output_path}")

        cv2.imshow('Multi-Object Tracking', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import os
from q4d import detect  # 确保已经正确导入detect函数


class Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.MultiTracker_create()
        self.tracked_boxes = []  # 存储跟踪框

    def add_new_trackers(self, frame, new_bboxes):
        for bbox in new_bboxes:
            # print(next_id)
            self.tracker.add(cv2.legacy.TrackerKCF_create(), frame, tuple(bbox))
            self.tracked_boxes.append(bbox)

    def update(self, frame):
        """更新所有跟踪器的状态，并返回当前跟踪的框"""
        success, boxes = self.tracker.update(frame)
        if success:
            self.tracked_boxes = [tuple(box) for box in boxes]
        return boxes

    # def reinitialize(self, frame, bboxes, ids):
    #     """使用新的跟踪框和ID重新初始化跟踪器"""
    #     self.tracker = cv2.legacy.MultiTracker_create()
    #     self.tracked_boxes = []
    #     self.ids = []
    #     for bbox, id in zip(bboxes, ids):
    #         self.tracker.add(cv2.legacy.TrackerKCF_create(), frame, tuple(bbox))
    #         self.tracked_boxes.append(bbox)
    #         self.ids.append(id)  # 重新分配原有的ID，保持跟踪连续性

    def reinitialize_trackers(self, frame):
        """使用当前跟踪框重新初始化所有跟踪器"""
        self.tracker = cv2.legacy.MultiTracker_create()  # 重新创建跟踪器实例
        for bbox in self.tracked_boxes:
            self.tracker.add(cv2.legacy.TrackerKCF_create(), frame, bbox)


import cv2


def draw_boxes(frame, boxes, face_num, cup_num, razor_num):
    # 初始化计数器
    face_count = 1
    cup_count = 1
    razor_count = 1

    # 总框的索引，用于确定何时切换到下一个物品类型
    index = 0

    for box in boxes:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # 判断当前框属于哪一类型，并相应地增加计数器
        if index < face_num:
            item_type = 'face'
            count = face_count
            face_count += 1
        elif index < face_num + cup_num:
            item_type = 'cup'
            count = cup_count
            cup_count += 1
        else:
            item_type = 'razor'
            count = razor_count
            razor_count += 1

        # 生成标签文本
        label = f"{item_type}{count}"

        # 在框上方显示标签
        cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 更新索引
        index += 1

    return frame


# def iou(box1, box2):
#     """计算两个边界框的交并比"""
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2
#     inter_x1 = max(x1, x2)
#     inter_y1 = max(y1, y2)
#     inter_x2 = min(x1+w1, x2+w2)
#     inter_y2 = min(y1+h1, y2+h2)
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     union_area = w1*h1 + w2*h2 - inter_area
#     return inter_area / union_area if union_area > 0 else 0


# def filter_new_boxes(tracked_boxes, detected_boxes):
#     """筛选出真正的新目标边界框，没有与现有跟踪框重大重叠"""
#     new_boxes = []
#     for dbox in detected_boxes:
#         if all(iou(dbox, tbox) < 0.5 for tbox in tracked_boxes):  # 交并比阈值可调整
#             new_boxes.append(dbox)
#     return new_boxes

def flatten_boxes(box_list):
    """将嵌套列表中的所有元组展平到一个单一列表中，忽略空的子列表"""
    flattened_list = []
    for sublist in box_list:
        if sublist:  # 检查子列表是否非空
            flattened_list.extend(sublist)  # 添加所有元组到新列表中
    return flattened_list


def main():
    video_path = 'data/hw-4.mp4'
    output_folder = './q4t-output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("读取视频失败")
        return

    tracker = Tracker()
    # box_dict = detect(frame)  # res dict
    # face_box = box_dict.get('face')
    # cup_box = box_dict.get('cup')
    # razor_box = box_dict.get('razor')
    #
    # tracker.add_new_trackers(frame, face_box)
    # tracker.add_new_trackers(frame, cup_box)
    # tracker.add_new_trackers(frame, razor_box)

    frame_count = 0
    frame_gap = 20
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # print(tracker.tracked_boxes[:])

        if frame_count % frame_gap == 0 :
            tracker.tracked_boxes = []
            box_dict = detect(frame)  # res dict
            face_box = box_dict.get('face')
            cup_box = box_dict.get('cup')
            razor_box = box_dict.get('razor')

            tracker.add_new_trackers(frame, face_box)
            tracker.add_new_trackers(frame, cup_box)
            tracker.add_new_trackers(frame, razor_box)

            tracker.reinitialize_trackers(frame)  # 确保跟踪器内部状态更新
            if len(cup_box)<1:
                frame_gap=3
            elif frame_gap==3:
                frame_gap = 20

        # print(tracker.tracked_boxes[:])
        boxes = tracker.update(frame)
        # print(tracker.tracked_boxes[:])
        frame_with_boxes = draw_boxes(frame, boxes, len(face_box), len(cup_box), len(razor_box))  # 绘制跟踪框

        output_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(output_path, frame_with_boxes)
        print(f"保存帧 {frame_count} 到 {output_path}")

        cv2.imshow('多目标跟踪', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

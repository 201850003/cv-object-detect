import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 模板图像路径
cup_front_template_path = 'data/templates/cup-front.png'
cup_up_template_path = 'data/templates/cup-up.png'
cup_move_template_path = 'data/templates/cup-move.png'
cup_down_template_path = 'data/templates/cup-down.png'
cup_incline_template_path = 'data/templates/cup-incline.png'
cup_incline_move_template_path = 'data/templates/cup-incline-move.png'
cup_incline_up_template_path = 'data/templates/cup-incline-up.png'
cup_incline_down_template_path = 'data/templates/cup-incline-down.png'
cup_move_quick_template_path = 'data/templates/cup-move-quick.png'

razor_front_template_path = 'data/templates/razor-front.png'
razor_incline_template_path = 'data/templates/razor-up.png'
razor_up_template_path = 'data/templates/razor-incline.png'
razor_incline_up_template_path = 'data/templates/razor-incline-up.png'

# 加载模板图像
cup_front_template = cv2.imread(cup_front_template_path, cv2.IMREAD_GRAYSCALE)
cup_up_template = cv2.imread(cup_up_template_path, cv2.IMREAD_GRAYSCALE)
cup_move_template = cv2.imread(cup_move_template_path, cv2.IMREAD_GRAYSCALE)
cup_down_template = cv2.imread(cup_down_template_path, cv2.IMREAD_GRAYSCALE)
cup_incline_template = cv2.imread(cup_incline_template_path, cv2.IMREAD_GRAYSCALE)
cup_incline_move_template = cv2.imread(cup_incline_move_template_path, cv2.IMREAD_GRAYSCALE)
cup_incline_up_template = cv2.imread(cup_incline_up_template_path, cv2.IMREAD_GRAYSCALE)
cup_incline_down_template = cv2.imread(cup_incline_down_template_path, cv2.IMREAD_GRAYSCALE)
cup_move_quick_template = cv2.imread(cup_move_quick_template_path, cv2.IMREAD_GRAYSCALE)

razor_front_template = cv2.imread(razor_front_template_path, cv2.IMREAD_GRAYSCALE)
razor_incline_template = cv2.imread(razor_incline_template_path, cv2.IMREAD_GRAYSCALE)
razor_up_template = cv2.imread(razor_up_template_path, cv2.IMREAD_GRAYSCALE)
razor_incline_up_template = cv2.imread(razor_incline_up_template_path, cv2.IMREAD_GRAYSCALE)


# 定义目标检测函数（Haar级联分类器）
def detect_faces(image, face_cascade, scale_factor=1.1, min_neighbors=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return faces


# 非极大值抑制函数
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]

    while len(idxs) > 0:
        last = idxs[0]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[1:]])
        yy1 = np.maximum(y1[last], y1[idxs[1:]])
        xx2 = np.minimum(x2[last], x2[idxs[1:]])
        yy2 = np.minimum(y2[last], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[1:]]

        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    return boxes[pick].astype("int")


# 定义多尺度模板匹配函数
def multi_scale_template_matching(image, template, threshold=0.7, scale_range=(0.7, 1.3), scale_steps=6):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    h, w = template.shape[:2]
    matches = []

    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        resized_template = cv2.resize(template, (int(w * scale), int(h * scale)))
        if resized_template.shape[0] > gray.shape[0] or resized_template.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            matches.append(
                (pt[0], pt[1], pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0], res[pt[1], pt[0]]))

    # 应用非极大值抑制来过滤匹配结果
    matches = np.array(matches)
    if len(matches) > 0:
        matches = non_max_suppression(matches, overlap_thresh=0.3)

    return matches


# 待检测图像的目录
test_image_dir = 'data/q1-frame'
output_dir = 'data/outframe'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)


def detect(test_image):
    # 检测人脸
    faces = detect_faces(test_image, face_cascade)

    # 多尺度模板匹配水杯
    threshold_second = 0.8
    cup_front_matches = multi_scale_template_matching(test_image, cup_front_template)
    cup_up_matches = multi_scale_template_matching(test_image, cup_up_template)
    cup_move_matches = multi_scale_template_matching(test_image, cup_move_template)
    cup_down_matches = multi_scale_template_matching(test_image, cup_down_template, threshold_second)
    cup_incline_matches = multi_scale_template_matching(test_image, cup_incline_template, threshold_second)
    cup_incline_move_matches = multi_scale_template_matching(test_image, cup_incline_move_template, threshold_second)
    cup_incline_up_matches = multi_scale_template_matching(test_image, cup_incline_up_template, threshold_second)
    cup_incline_down_matches = multi_scale_template_matching(test_image, cup_incline_down_template, threshold_second)
    cup_move_quick_matches = multi_scale_template_matching(test_image, cup_move_quick_template, threshold_second)

    # 多尺度模板匹配书本
    razor_front_matches = multi_scale_template_matching(test_image, razor_front_template)
    razor_incline_matches = multi_scale_template_matching(test_image, razor_incline_template)
    razor_up_matches = multi_scale_template_matching(test_image, razor_up_template)
    razor_incline_up_matches = multi_scale_template_matching(test_image, razor_incline_up_template)

    # # 绘制人脸边界框
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(test_image, 'face', (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    #
    # # 绘制水杯边界框
    # for match in [cup_front_matches, cup_up_matches, cup_move_matches, cup_down_matches, cup_incline_matches,
    #               cup_incline_move_matches, cup_incline_up_matches, cup_incline_down_matches, cup_move_quick_matches]:
    #     if match.size > 0:
    #         for (x1, y1, x2, y2, _) in match:
    #             cv2.rectangle(test_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #             cv2.putText(test_image, 'cup', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    #
    # # 绘制刮胡刀边界框
    # for match in [razor_front_matches, razor_up_matches, razor_incline_matches, razor_incline_up_matches]:
    #     if match.size > 0:
    #         for (x1, y1, x2, y2, _) in match:
    #             cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #             cv2.putText(test_image, 'razor', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)


    # # 使用Matplotlib显示结果（可选）
    # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    # plt.title(f'Detected Objects in {test_image}')
    # plt.axis('off')
    # plt.show()

    # 整合所有检测结果
    # all_matches = []
    # all_matches.extend([(x, y, w, h) for (x, y, w, h) in faces])  # 人脸检测结果
    # for match_set in [cup_front_matches, cup_up_matches, cup_move_matches, cup_down_matches, cup_incline_matches,
    #               cup_incline_move_matches, cup_incline_up_matches, cup_incline_down_matches, cup_move_quick_matches,
    #                   razor_front_matches, razor_up_matches, razor_incline_matches, razor_incline_up_matches]:
    #     all_matches.extend([(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, _) in match_set])

    face_match_res = []
    cup_match_res = []
    cup_match_res_final = []
    razor_match_res = []
    razor_match_res_final = []

    face_match_res.extend([(x, y, w, h) for (x, y, w, h) in faces])

    for match_set in [cup_front_matches, cup_up_matches, cup_move_matches, cup_down_matches, cup_incline_matches,
                      cup_incline_move_matches, cup_incline_up_matches, cup_incline_down_matches,
                      cup_move_quick_matches]:
        cup_match_res.extend([(x1, y1, x2, y2, _) for (x1, y1, x2, y2, _) in match_set])
    for match_set in [razor_front_matches, razor_up_matches, razor_incline_matches, razor_incline_up_matches]:
        razor_match_res.extend([(x1, y1, x2, y2, _) for (x1, y1, x2, y2, _) in match_set])

    if len(cup_match_res)>0:
        cup_match_res = np.array(cup_match_res)
        cup_match_res = non_max_suppression(cup_match_res)

    if len(razor_match_res)>0:
        razor_match_res = np.array(razor_match_res)
        razor_match_res = non_max_suppression(razor_match_res)

    cup_match_res_final.extend([(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, _) in cup_match_res])
    razor_match_res_final.extend([(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, _) in razor_match_res])

    res_dict = {
        'face':face_match_res,
        'cup':cup_match_res_final,
        'razor':razor_match_res_final
    }

    return res_dict

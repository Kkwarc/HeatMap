"""
Michał Kwarciński
"""
import numpy as np
import cv2
import cvzone
import math
import time
from sort import Sort
from ultralytics import YOLO


cap = cv2.VideoCapture("../Films/testVid1.mp4")  # for video


model = YOLO('runs/detect/yolov8n_custom_test_full_video_data/weights/best.pt')

# tracking
chair_tracker = Sort(max_age=math.inf, min_hits=3, iou_threshold=0.3)
table_tracker = Sort(max_age=math.inf, min_hits=3, iou_threshold=0.3)

classNames = ["person", "chair", "table"]


class Object:
    def __init__(self, id, x1, y1, x2, y2, is_table=True):
        self.id = id
        self.x1 = [x1]
        self.y1 = [y1]
        self.x2 = [x2]
        self.y2 = [y2]
        self.is_table = is_table
        self.objects = []
        self.people = 0

    def posUpdate(self, x1, y1, x2, y2):
        if len(self.x1) > 10:
            self.x1.pop(0)
            self.x2.pop(0)
            self.y1.pop(0)
            self.y2.pop(0)
        self.x1.append(x1)
        self.y1.append(y1)
        self.x2.append(x2)
        self.y2.append(y2)

    def getPos(self):
        x1 = int(np.average(self.x1))
        x2 = int(np.average(self.x2))
        y1 = int(np.average(self.y1))
        y2 = int(np.average(self.y2))
        x1 = 1 if x1 == 0 else x1
        x2 = 1 if x2 == 0 else x2
        y1 = 1 if y1 == 0 else y1
        y2 = 1 if y2 == 0 else y2
        return x1, y1, x2, y2

    def getSnapArea(self):
        pos = self.getPos()
        center = (int(abs((pos[2] + pos[0]) / 2)), int(abs((pos[3] + pos[1]) / 2)))
        r = int(np.sqrt((center[0] - pos[0]) ** 2 + (center[1] - pos[1]) ** 2))
        if self.is_table:
            return center, int(2.5 * r)
        return center, int(1.5 * r)


def changeFrameSize(img):
    height, width = img.shape[:2]
    scale_percent = 50  # zmniejsz skalę obrazu o 50 procent

    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    new_size = (new_width, new_height)

    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def addTimeImpactToHeatMask(array):
    array[array > 1] -= 1  # zapominaj stare wartości
    return array


def addPersonToHeatMask(array, box, speed, step_factor):
    x1, y1, width, height = box
    x2 = x1 + width
    y2 = y1 + height

    rect_width = abs(width)
    rect_height = abs(height)

    step_x = rect_width / step_factor
    step_y = rect_height / step_factor

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    j_indices, i_indices = np.indices(array.shape)

    distances = ((i_indices - x_center) / (rect_width / 2))**2 + ((j_indices - y_center) / (rect_height / 2))**2
    inside_mask = distances <= 1

    dist_x = np.abs(i_indices - x_center)
    dist_y = np.abs(j_indices - y_center)

    values = np.maximum(speed - np.minimum(step_x * dist_x / rect_width, step_y * dist_y / rect_height), 0)
    values[~inside_mask] = 0

    array += values.astype(int)
    return array


def convertToColor(image):
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    color_image[:, :, 0] = np.where(image > 0, np.minimum(image * 2, 255), 0)  # Kanał niebieski
    color_image[:, :, 2] = np.where(image > 255 / 2, np.minimum((image - 255 / 2) * 2, 255), 0)  # Kanał czerwony
    return color_image


def genHeatMap(img):
    HEAT_MAP = img
    image_float = HEAT_MAP.astype(float)
    min_val = np.min(image_float)
    max_val = np.max(image_float)
    HEAT_MAP = 255 * (image_float - min_val) / (max_val - min_val)

    HEAT_MAP = convertToColor(HEAT_MAP)
    return HEAT_MAP


def addImages(imgMain, img):
    overlay_rgb = img[:, :, 0:3]
    result = cv2.add(imgMain, overlay_rgb)
    return result


def detectObjects(frame, model):
    results = model(frame, stream=True, show=False)
    cv2.waitKey(1)

    people, chairs, tables = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = 1 if x1 == 0 else x1
            y1 = 1 if y1 == 0 else y1
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil(box.conf[0] * 100) / 100
            if conf < 0.75 or (0.8 < x2/x1 < 1.2) or (0.8 < y2/y1 < 1.2):
                continue

            currentArray = np.array([x1, y1, x2, y2, conf])

            class_id = int(box.cls[0])
            class_name = classNames[class_id]
            if class_name == "person":
                people = np.vstack((people, currentArray))
            elif class_name == "table":
                tables = np.vstack((tables, currentArray))
            else:
                chairs = np.vstack((chairs, currentArray))

    return people, tables, chairs


def updateObjectsTracker(objects, tracker):
    result = []
    if len(objects) > 0:
        detections = np.array(objects)
        result = tracker.update(detections)
    return result


def similarTables(table_list, result):
    for table in table_list:
        x1, y1, x2, y2 = table.getPos()
        if result[0] > 0.8 * x1 and result[2] < 1.2 * x2 and result[1] > 0.8 * y1 and result[3] < 1.2 * y2:
            return True
    return False


def resultToObj(result, old_list, is_tables):
    for r in result:
        if r[4] not in [o.id for o in old_list]:
            if not similarTables(old_list, r):
                old_list.append(Object(r[4], int(r[0]), int(r[1]), int(r[2]), int(r[3]), is_table=is_tables))
    return old_list


def getPersonToChairs(frame, object, objs_chairs, heatMask):
    best_obj = None
    bbox = int(object[0]), int(object[1]), int(object[2] - object[0]), int(object[3]) - int(object[1])
    object_center = (int((bbox[1] + bbox[3] / 2)), int((bbox[0] + bbox[2] / 2)))
    cvzone.cornerRect(frame, bbox=bbox, l=9, rt=2, colorR=(255, 0, 255))
    best_r = math.inf
    for obj in objs_chairs:
        objCenter, radius = obj.getSnapArea()
        r = np.sqrt((objCenter[1] - object_center[0]) ** 2 + (objCenter[0] - object_center[1]) ** 2)
        if r > radius:
            continue

        if r < best_r:
            best_r = r
            best_obj = obj.id
            heatMask = addPersonToHeatMask(heatMask, bbox, speed=255, step_factor=50)
    for i, obj in enumerate(objs_chairs):
        if obj.id == best_obj:
            objs_chairs[i].people += 1
    return objs_chairs, heatMask


def getChairsToTables(object, objs_tables):
    best_obj = None
    x1, y1, x2, y2 = object.getPos()
    bbox = (x1, y1, x2-x1, y2-y1)
    object_center = (int((bbox[1] + bbox[3] / 2)), int((bbox[0] + bbox[2] / 2)))
    best_r = math.inf
    for obj in objs_tables:
        objCenter, radius = obj.getSnapArea()
        r = np.sqrt((objCenter[1] - object_center[0]) ** 2 + (objCenter[0] - object_center[1]) ** 2)
        if r > radius:
            continue

        if r < best_r:
            best_r = r
            best_obj = obj.id
    for i, obj in enumerate(objs_tables):
        if obj.id == best_obj:
            objs_tables[i].people += object.people
            if object.id not in [o.id for o in objs_tables[i].objects]:
                objs_tables[i].objects.append(object)
    return objs_tables


def drawSnapArea(frame, obj, color=(0, 255, 0)):
    center, radius = obj.getSnapArea()
    cv2.circle(frame, center, radius, color, 2)
    return frame


def drawObjInfo(frame, obj, color=(0, 255, 0)):
    center, r = obj.getSnapArea()
    if obj.is_table:
        cv2.putText(frame, f"Id: {obj.id}, P: {obj.people}/{len(obj.objects)}",
                    (max(int(center[0]-75), 0), center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def resetTables(tables):
    for i, table in enumerate(tables):
        tables[i].people = 0
    return tables


MAIN_TABLES = []
MAIN_CHAIRS = []
HEAT_MAP = None
heatMask = None


i = 0
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame = changeFrameSize(frame)
    if heatMask is None:
        heatMask = np.zeros(frame.shape[:2])
    if HEAT_MAP is None:
        HEAT_MAP = heatMask

    people, tables, chairs = detectObjects(frame, model)

    resultChair = updateObjectsTracker(chairs, chair_tracker)
    resultTables = updateObjectsTracker(tables, table_tracker)

    MAIN_TABLES = resultToObj(resultTables, MAIN_TABLES, is_tables=True)
    MAIN_CHAIRS = resultToObj(resultChair, MAIN_CHAIRS, is_tables=False)

    for person in people:
        MAIN_CHAIRS, heatMask = getPersonToChairs(frame, person, MAIN_CHAIRS, heatMask)

    for chair in MAIN_CHAIRS:
        MAIN_TABLES = getChairsToTables(chair, MAIN_TABLES)

    for obj in MAIN_TABLES:
        frame = drawSnapArea(frame, obj)
        frame = drawObjInfo(frame, obj)

    for obj in MAIN_CHAIRS:
        frame = drawSnapArea(frame, obj, color=(255, 0, 0))
        frame = drawObjInfo(frame, obj, color=(255, 0, 0))

    HEAT_MAP = genHeatMap(heatMask)
    heatMask = addTimeImpactToHeatMask(heatMask)
    MAIN_TABLES = resetTables(MAIN_TABLES)
    MAIN_CHAIRS = resetTables(MAIN_CHAIRS)

    # cv2.imshow("Frame", frame)
    # cv2.imshow("heatMask", heatMask)
    # cv2.imshow("heatMap", HEAT_MAP)
    finalHeatMap = addImages(HEAT_MAP, frame)
    cv2.imshow("IMG", finalHeatMap)

    print(f'Frame time: {time.time()-start_time}')
    cv2.imwrite(f"gifData/{i}.png", finalHeatMap)
    i += 1
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


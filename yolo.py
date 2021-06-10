import cv2
import time
import telegram


class Model:
    def __init__(self):
        self.image = None
        self.cap = cv2.VideoCapture(0)
        self.time_start = 0
        self.time_end = 0
        self.second = self.time_end - self.time_start
        self.timer = 60
        self.chat_id = '-575650338'
        self.bot = telegram.Bot(token='1734977699:AAHp_s_iPQctGHWbxg1-H0lhYREWwVxYRFQ')
        self.yolo_cfg = "yolov4-tiny-obj.cfg"
        self.yolo_weights = "yolov4-tiny-obj_final.weights"
        self.video_path = ""
        self.net = cv2.dnn_DetectionModel(self.yolo_cfg, self.yolo_weights)
        self.net.setInputSize((512, 512))
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open("obj.names", "rt") as f:
            self.names = f.read().rstrip('\n').split('\n')

    def predict_mask_on_image(self):
        self.time_start = time.time()
        classes, confidences, boxes = self.net.detect(self.image, confThreshold=0.7, nmsThreshold=0.4)
        class_list = []
        if len(classes) != 0:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                class_list.append(classId)
                label = '%.2f' % confidence
                label = '%s: %s' % (self.names[classId], label)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                top = max(top, label_size[1])
                if classId == 0:  # Maskeli
                    cv2.rectangle(self.image, box, color=(0, 0, 255), thickness=2)
                if classId == 1:  # Maskesiz
                    cv2.rectangle(self.image, box, color=(0, 255, 0), thickness=2)
                if classId == 2:  # Hatalı
                    cv2.rectangle(self.image, box, color=(0, 255, 255), thickness=2)
                cv2.putText(self.image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("Bitirme Projesi", self.image)
        cv2.waitKey(1)
        self.time_end = time.time()
        self.second = self.time_end - self.time_start
        self.timer += self.second
        print(self.timer)
        if ((0 in class_list) or (2 in class_list)) and self.timer >= 60:
            self.timer = 0

            cv2.imwrite("send_photo.jpg", self.image)
            try:
                self.bot.send_photo(self.chat_id, photo=open("send_photo.jpg", "rb"))
            except Exception as e:
                print(e)
                self.chat_id = '-575650338'
                self.bot = telegram.Bot(token='1734977699:AAHp_s_iPQctGHWbxg1-H0lhYREWwVxYRFQ')
                self.bot.send_photo(self.chat_id, photo=open("send_photo.jpg", "rb"))

    def predict_video(self):
        while True:
            _, frame = self.cap.read()
            self.image = frame
            self.predict_mask_on_image()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def predict_image(self, image_path):
        self.image = cv2.imread(0)
        self.predict_mask_on_image()


if __name__ == '__main__':
    m = Model()
    # video_path = "C:/Users/Fatih/Desktop/Bitirme Projesi/Kodlar/YOLO/video/demo.mp4"
    m.predict_video()
    # m.predict_image("resim.png")

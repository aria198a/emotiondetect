from fer import FER      # 匯入 FER 模組
import cv2    # 匯入OpenCV模組


detector = FER()   # 省略mtcnn參數設定，使用預設值 mtcnn=False
image = cv2.imread("C:/Users/fclin/Desktop/emotionTextbook/picturesdata/happy1.jpg")
emotions = detector.detect_emotions(image)    # 偵測image的臉部表情

print(emotions)       # 顯示偵測結果
print(emotions[0]['box'])    # 臉部的邊界框的位置 x 和 y 座標，以及寬度和高度
print(emotions[0]['emotions']['happy'])
























































# from hsemotion import facial_emotions
# import cv2


# a = facial_emotions.HSEmotionRecognizer()
# # define object 

# img_path = "picturesdata/happy1.jpg"

# face_img = cv2.imread(img_path)

# # 使用cv2 load image

# classs, score = a.predict_emotions(face_img)
# # use function in object

# print("Predicted Emotion:", classs)
# print("Confidence Score:", score)

'''
def predict_emotions(self,face_img, logits=True):
        features=self.extract_features(face_img)
        scores=self.get_probab(features)[0]
        if self.is_mtl:
            x=scores[:-2]
        else:
            x=scores
        pred=np.argmax(x)
        
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2]=e_x
            else:
                scores=e_x
        return self.idx_to_class[pred],scores    
'''
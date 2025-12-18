import cv2, numpy as np, tensorflow as tf

def crack_severity(img_path, model):
    img=cv2.imread(img_path)
    img=cv2.resize(img,(224,224))/255
    img=np.expand_dims(img,0)
    prob=float(model.predict(img)[0])
    gray=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
    texture=min(np.var(cv2.Laplacian(gray,cv2.CV_64F))/1000,1)
    return prob, round(prob*texture*100,2)

import tensorflow as tf, cv2, numpy as np, matplotlib.pyplot as plt

def grad_cam(image_path, model):
    img=cv2.imread(image_path)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_res=cv2.resize(img_rgb,(224,224))/255
    x=np.expand_dims(img_res,0)

    grad_model=tf.keras.models.Model(
        [model.inputs],
        [model.layers[-3].output, model.output])

    with tf.GradientTape() as tape:
        conv_out, pred=grad_model(x)
        loss=pred[:,0]

    grads=tape.gradient(loss, conv_out)[0]
    weights=tf.reduce_mean(grads,axis=(0,1))
    cam=np.zeros(conv_out[0].shape[:2])

    for i,w in enumerate(weights):
        cam+=w*conv_out[0][:,:,i]

    cam=np.maximum(cam,0)
    cam=cv2.resize(cam,(img.shape[1],img.shape[0]))
    cam/=cam.max()

    heatmap=cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    out=cv2.addWeighted(img_rgb,0.6,heatmap,0.4,0)

    plt.imshow(out);plt.axis('off')

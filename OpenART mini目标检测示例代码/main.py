import seekfree, pyb
import sensor, image, time, tf, gc

sensor.reset()                      # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
# sensor.set_windowing((224,224))
clock = time.clock()                # Create a clock object to track the FPS.


#设置模型路径
face_detect = '/sd/yolo3_iou_smartcar_final_with_post_processing.tflite'
#载入模型
net1 = tf.load(face_detect)


net_path = "num_model_20250422_1239.tflite"                                  # 定义模型的路径
labels = [line.rstrip() for line in open("/sd/labels_num.txt")]   # 加载标签
net2 = tf.load(net_path, load_to_fb=True)


arr = [0] * 10

while(True):
    clock.tick()
    img = sensor.snapshot()

    num = 0
    #使用模型进行识别
    for obj in tf.detect(net1,img):
        x1,y1,x2,y2,label,scores = obj
        if(scores>0.70):
            # print(obj)
            w = x2- x1
            h = y2 - y1
            x1 = int((x1-0.1)*img.width())
            y1 = int(y1*img.height())
            w = int(w*img.width())
            h = int(h*img.height())
            img.draw_rectangle((x1-20,y1,180,180),thickness=2,color=(255,0,0))
            img1 = img.copy(1, 1, roi=(x1-20,y1,180,180))
            for obj in tf.classify(net2 , img1, min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
                # print("**********\nTop 1 Detections at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
                sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
                # 打印准确率最高的结果
                for i in range(1):
                    # print("%s = %f" % (sorted_list[i][0], sorted_list[i][1]))
                    if sorted_list[i][1] > 0.7:
                        # print("%s = %f" % (sorted_list[i][0], sorted_list[i][1]))
                            arr[num] = int(sorted_list[i][0])
                            # print(num,arr[num])
            num = num + 1

    if num == 2:
        print(num,arr[0],arr[1])
        arr = [0] * 10
        num=0
    # print(clock.fps())

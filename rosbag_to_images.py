from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
from rosbags.image import message_to_cvimage # pip install rosbags-image
import os
from tqdm import tqdm


def main():
    rosbag_paths = [
                    '/home/mfi/repos/ros1_ws/src/philip/data/lego_fish_high_2025-06-09-17-30-24.bag',
                    # '/mnt/hdd2/yizhouhu/bags/lego_R_2025-05-10-17-07-08.bag',
                    # '/mnt/hdd2/yizhouhu/bags/lego_S_2025-05-08-17-50-45.bag',
                    # '/mnt/hdd2/yizhouhu/bags/lego_fish_high_2025-04-29-16-06-19.bag'
                    ]
    image_output_dir = '/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/061225/'
    N = 200 # 1 image per N frames
    width = 1000

    # extract images
    for rosbag_path in rosbag_paths:
        with AnyReader([Path(rosbag_path)]) as reader:
            frame = 0
            for connection in reader.connections:
                print(connection.topic, connection.msgtype)
            for connection, timestamp, rawdata in tqdm(reader.messages()):
                if 'gen3_image' in connection.topic: # topic Name of images
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    if frame%N == 0:
                        img = message_to_cvimage(msg, 'bgr8') # change encoding type if needed
                        img = cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1])), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(image_output_dir + f'/{os.path.basename(rosbag_path).split(".")[0]}_%06i.png' % frame, img)
                    frame += 1

if __name__ == '__main__':
    main()
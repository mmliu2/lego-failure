from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
from rosbags.image import message_to_cvimage # pip install rosbags-image
import os
from tqdm import tqdm


def main():
    rosbag_paths = ['/mnt/hdd2/yizhouhu/bags/lego_cliff_2025-05-10-17-19-46.bag',
                    '/mnt/hdd2/yizhouhu/bags/lego_R_2025-05-10-17-07-08.bag',
                    '/mnt/hdd2/yizhouhu/bags/lego_S_2025-05-08-17-50-45.bag',
                    '/mnt/hdd2/yizhouhu/bags/lego_fish_high_2025-04-29-16-06-19.bag']
    image_output_dir = './data/test_images/'
    N = 20 # 1 image per N frames

    # extract images
    for rosbag_path in rosbag_paths:
        with AnyReader([Path(rosbag_path)]) as reader:
            frame = 0
            for connection in reader.connections:
                print(connection.topic, connection.msgtype)
            for connection, timestamp, rawdata in tqdm(reader.messages()):
                if '/gen3_image/compressed' in connection.topic: # topic Name of images
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, 'bgr8') # change encoding type if needed
                    if frame%N == 0:
                        cv2.imwrite(image_output_dir + f'/{os.path.basename(rosbag_path).split(".")[0]}_%06i.png' % frame, img)
                    frame += 1

if __name__ == '__main__':
    main()
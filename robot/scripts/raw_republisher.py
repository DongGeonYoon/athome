#!/usr/bin/env python3
"""Replay helper: compressed colour images -> raw bgr8 /camera/camera/color/image_raw (what the V4L2 camera node
publishes live), same header. Used to test live CG-D timing with raw-image transport from a bag."""
import cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

rclpy.init(); n = Node('raw_republisher'); pub = n.create_publisher(Image, '/camera/camera/color/image_raw', 5)
def cb(m):
    img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
    o = Image(); o.header = m.header; o.height, o.width = img.shape[:2]; o.encoding = 'bgr8'; o.step = img.shape[1] * 3
    o.data = img.tobytes(); pub.publish(o)
n.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', cb, 10)
rclpy.spin(n)

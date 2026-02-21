'''
Lidar Data:
topics: /scan
message type: sensor_msgs/LaserScan or sensor_msgs/msg/LaserScan
'''
class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.front_dist = 1.0
        self.left_dist = 1.0
        self.right_dist = 1.0
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

    def scan_callback(self, msg: LaserScan):
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return
        center = n // 2
        side_offset = max(3, n // 12)  # roughly 30 degrees window depending on lidar resolution
        # use small windows and safe_min_distance to ignore inf/nan
        self.front_dist = safe_min_distance(ranges, max(0, center - 2), min(n, center + 3))
        self.left_dist = safe_min_distance(ranges, max(0, center + side_offset - 2), min(n, center + side_offset + 3))
        self.right_dist = safe_min_distance(ranges, max(0, center - side_offset - 2), min(n, center - side_offset + 3))

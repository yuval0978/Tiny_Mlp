'''
Mpu6050 publishers:
topics: /mpu6050 or /imu
message type: sensor_msgs/Imu or sensor_msgs/msg/Imu

need to install:
sudo apt install ros-humble-tf-transformations

90° = 1.57 radians
180° = 3.14 radians
'''
class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')
        self.current_yaw = 0.0
        self.subscription = self.create_subscription(
            Imu, '/imu', self.imu_callback, 50)

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        self.current_yaw = float(yaw)


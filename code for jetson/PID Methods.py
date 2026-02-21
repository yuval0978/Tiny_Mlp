
# ------------------ Control primitives ------------------

def turn_left_90(imu_node: ImuNode, pid: PIDController, mlp: TinyMLP, dt=0.01, pwm_speed=DEFAULT_TURN_SPEED):
    pid.reset()
    start_yaw = imu_node.current_yaw
    target_yaw = normalize_angle(start_yaw + math.pi / 2.0)
    e_hist = [0.0, 0.0, 0.0]; u_hist = [0.0, 0.0]

    while rclpy.ok():
        current_yaw = imu_node.current_yaw
        error = normalize_angle(target_yaw - current_yaw)
        if abs(error) < 0.03:
            stop_motors()
            return
        feat = build_feature_vector(e_hist, pid.integral, u_hist, sp=0.0)
        out = mlp.forward(feat)
        scale = 2.0 if abs(error) > 0.5 else max(0.2, 1.0 - abs(error))
        out = out * scale
        new_gains = np.array([pid.Kp + out[0], pid.Ki + out[1], pid.Kd + out[2]])
        pid.Kp, pid.Ki, pid.Kd = [float(np.clip(v, a, b)) for v, a, b in zip(new_gains, [0, 0, 0], [80, 15, 8])]
        u = pid.compute(error, dt)
        rotate_left(speed=pwm_speed)
        # online learn (optional)
        dL_dout = -error * out
        mlp.backward_from_dout(dL_dout)
        mlp.step_sgd(lr=0.1, clip=0.03)
        e_hist = [error] + e_hist[:2]
        u_hist = [u] + u_hist[:1]
        time.sleep(dt)


def turn_right_90(imu_node: ImuNode, pid: PIDController, mlp: TinyMLP, dt=0.01, pwm_speed=DEFAULT_TURN_SPEED):
    pid.reset()
    start_yaw = imu_node.current_yaw
    target_yaw = normalize_angle(start_yaw - math.pi / 2.0)
    e_hist = [0.0, 0.0, 0.0]; u_hist = [0.0, 0.0]

    while rclpy.ok():
        current_yaw = imu_node.current_yaw
        error = normalize_angle(target_yaw - current_yaw)
        if abs(error) < 0.03:
            stop_motors()
            return
        feat = build_feature_vector(e_hist, pid.integral, u_hist, sp=0.0)
        out = mlp.forward(feat)
        scale = 2.0 if abs(error) > 0.5 else max(0.2, 1.0 - abs(error))
        out = out * scale
        new_gains = np.array([pid.Kp + out[0], pid.Ki + out[1], pid.Kd + out[2]])
        pid.Kp, pid.Ki, pid.Kd = [float(np.clip(v, a, b)) for v, a, b in zip(new_gains, [0, 0, 0], [80, 15, 8])]
        u = pid.compute(error, dt)
        rotate_right(speed=pwm_speed)
        # online learn (optional)
        dL_dout = -error * out
        mlp.backward_from_dout(dL_dout)
        mlp.step_sgd(lr=0.1, clip=0.03)
        e_hist = [error] + e_hist[:2]
        u_hist = [u] + u_hist[:1]
        time.sleep(dt)


def drive_straight(imu_node: ImuNode, pid: PIDController, mlp: TinyMLP, pwm_speed=DEFAULT_DRIVE_SPEED, dt=0.01, desired_heading=None, duration=None):
    """
    Drive forward while keeping heading stable.
    If desired_heading is None -> use current yaw at start (maintain that heading).
    duration: optional seconds to drive, otherwise returns only when stopped externally.
    """
    pid.reset()
    start_heading = imu_node.current_yaw if desired_heading is None else desired_heading
    target_yaw = normalize_angle(start_heading)
    e_hist = [0.0, 0.0, 0.0]; u_hist = [0.0, 0.0]
    t0 = time.time()
    forward_base(pwm_speed)

    while rclpy.ok():
        if duration is not None and (time.time() - t0) >= duration:
            stop_motors()
            return
        yaw = imu_node.current_yaw
        error = normalize_angle(target_yaw - yaw)
        # small-deadband: if tiny error, keep base forward
        if abs(error) < 0.005:
            forward_base(pwm_speed)
        else:
            feat = build_feature_vector(e_hist, pid.integral, u_hist, sp=0.0)
            out = mlp.forward(feat)
            scale = 2.0 if abs(error) > 0.5 else max(0.2, 1.0 - abs(error))
            out = out * scale
            new_gains = np.array([pid.Kp + out[0], pid.Ki + out[1], pid.Kd + out[2]])
            pid.Kp, pid.Ki, pid.Kd = [float(np.clip(v, a, b)) for v, a, b in zip(new_gains, [0, 0, 0], [80, 15, 8])]
            u = pid.compute(error, dt)
            # convert u to differential PWM correction (scale and clamp sensibly)
            # choose a scale factor so u values produce small PWM deltas
            CORR_SCALE = 0.8  # adjust this in testing
            delta = float(np.clip(CORR_SCALE * u, -1.0, 1.0)) * pwm_speed * 100.0
            base = pwm_speed * 100.0
            left_dc = float(np.clip(base - delta, 0.0, 100.0))
            right_dc = float(np.clip(base + delta, 0.0, 100.0))
            pwm_left_fwd.ChangeDutyCycle(left_dc)
            pwm_right_fwd.ChangeDutyCycle(right_dc)
            dL_dout = -error * out
            mlp.backward_from_dout(dL_dout)
            mlp.step_sgd(lr=0.1, clip=0.03)
            e_hist = [error] + e_hist[:2]
            u_hist = [u] + u_hist[:1]
        time.sleep(dt)
# ------------------ Maze solver main loop ------------------
def maze_solver(imu_node: ImuNode, lidar_node: LidarNode, pid: PIDController, mlp: TinyMLP): #uses try and except to catch unkwon errors and stop the motors if they occur
    """Main maze behavior loop (blocking)."""
    try:
        while rclpy.ok():
            front = lidar_node.front_dist
            left = lidar_node.left_dist
            right = lidar_node.right_dist

            # debug logging (minimal)
            # print(f"front={front:.2f}, left={left:.2f}, right={right:.2f}")

            if front < 0.10:  # wall at 10 cm or closer
                stop_motors()
                time.sleep(0.05)
                if left > right:
                    turn_left_90(imu_node, pid, mlp, pwm_speed=DEFAULT_TURN_SPEED)
                else:
                    turn_right_90(imu_node, pid, mlp, pwm_speed=DEFAULT_TURN_SPEED)
                # small pause after turn
                time.sleep(0.05)
            else:
                # drive forward while keeping heading stable (maintain current heading)
                drive_straight(imu_node, pid, mlp, pwm_speed=DEFAULT_DRIVE_SPEED, dt=0.01, desired_heading=None, duration=0.20)
                # drive_straight uses duration so we loop back to re-check lidar frequently
    except Exception as ex:
        print("Maze solver exception:", ex)
        stop_motors()
        

'''
motor stuff:
install this: pip install Jetson.GPIO
'''

# ------------------ GPIO / PWM Configuration ------------------
LEFT_FWD = 32
LEFT_BWD = 33
RIGHT_FWD = 35
RIGHT_BWD = 37

PWM_FREQ = 1000  # Hz

# safe default PWM speed values (0..1)
DEFAULT_TURN_SPEED = 0.45
DEFAULT_DRIVE_SPEED = 0.40

# initialize GPIO and PWMs
GPIO.setmode(GPIO.BOARD)
for pin in [LEFT_FWD, LEFT_BWD, RIGHT_FWD, RIGHT_BWD]:
    GPIO.setup(pin, GPIO.OUT)

pwm_left_fwd = GPIO.PWM(LEFT_FWD, PWM_FREQ)
pwm_left_bwd = GPIO.PWM(LEFT_BWD, PWM_FREQ)
pwm_right_fwd = GPIO.PWM(RIGHT_FWD, PWM_FREQ)
pwm_right_bwd = GPIO.PWM(RIGHT_BWD, PWM_FREQ)

for pwm in (pwm_left_fwd, pwm_left_bwd, pwm_right_fwd, pwm_right_bwd):
    pwm.start(0.0)


def stop_motors():
    """Stop all motor PWMs immediately."""
    for pwm in (pwm_left_fwd, pwm_left_bwd, pwm_right_fwd, pwm_right_bwd):
        try:
            pwm.ChangeDutyCycle(0.0)
        except Exception:
            pass


def rotate_left(speed=DEFAULT_TURN_SPEED):
    """Spin in place left: right wheel forward, left wheel backward."""
    s = float(np.clip(speed, 0.0, 1.0)) * 100.0
    pwm_left_fwd.ChangeDutyCycle(0.0)
    pwm_left_bwd.ChangeDutyCycle(s)
    pwm_right_fwd.ChangeDutyCycle(s)
    pwm_right_bwd.ChangeDutyCycle(0.0)


def rotate_right(speed=DEFAULT_TURN_SPEED):
    """Spin in place right: left wheel forward, right wheel backward."""
    s = float(np.clip(speed, 0.0, 1.0)) * 100.0
    pwm_left_fwd.ChangeDutyCycle(s)
    pwm_left_bwd.ChangeDutyCycle(0.0)
    pwm_right_fwd.ChangeDutyCycle(0.0)
    pwm_right_bwd.ChangeDutyCycle(s)


def forward_base(speed=DEFAULT_DRIVE_SPEED):
    """Both wheels forward at same base speed (0..1)."""
    s = float(np.clip(speed, 0.0, 1.0)) * 100.0
    pwm_left_bwd.ChangeDutyCycle(0.0)
    pwm_left_fwd.ChangeDutyCycle(s)
    pwm_right_bwd.ChangeDutyCycle(0.0)
    pwm_right_fwd.ChangeDutyCycle(s)


# ------------------ Helpers ------------------
def normalize_angle(angle):
    """Return angle normalized to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def safe_min_distance(ranges, start, end):
    """
    Return min finite distance in ranges[start:end].
    If no finite values, return a large number.
    """
    try:
        sub = np.array(ranges[start:end], dtype=np.float32)
    except Exception:
        return 1e3
    finite = np.isfinite(sub)
    if not np.any(finite):
        return 1e3
    return float(np.nanmin(np.where(finite, sub, np.nan)))




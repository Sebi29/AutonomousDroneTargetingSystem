from vehicle import Driver
import math

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def local_to_world_xz(lx, lz, tx, tz, yaw):
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    wx = tx + (cy * lx - sy * lz)
    wz = tz + (sy * lx + cy * lz)
    return wx, wz

def build_world_waypoints_from_road_local_xy(road_waypoints_local_xyz, road_tx, road_tz, road_yaw):
    out = []
    for (x, y, _z) in road_waypoints_local_xyz:
        wx, wz = local_to_world_xz(float(x), float(y), road_tx, road_tz, road_yaw)
        out.append((wx, wz))
    return out

def interpolate_waypoints(points, resolution=1.0):
    out = []
    for i in range(len(points) - 1):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[i+1]
        d = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(d / resolution))
        for j in range(steps):
            t = j / steps
            out.append((x1 + t*(x2-x1), y1 + t*(y2-y1), 0.0))
    out.append(points[-1])
    return out

def closest_index(points, x, z, last_i=0, window=120):
    n = len(points)
    if n == 0:
        return 0
    lo = max(0, last_i - window)
    hi = min(n - 1, last_i + window)
    best_i = lo
    best_d2 = 1e18
    for i in range(lo, hi + 1):
        px, pz = points[i]
        d2 = (px - x) ** 2 + (pz - z) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i

def lookahead_point(points, start_i, Ld):
    n = len(points)
    if n == 0:
        return (0.0, 0.0), 0
    i = start_i
    acc = 0.0
    px, pz = points[i]
    while i < n - 1 and acc < Ld:
        nx, nz = points[i + 1]
        acc += math.hypot(nx - px, nz - pz)
        px, pz = nx, nz
        i += 1
    return (px, pz), i

def run():
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    gps = driver.getDevice("gps")
    compass = driver.getDevice("compass")

    if gps is None or compass is None:
        print("❌ Missing sensors on the car.")
        return

    gps.enable(timestep)
    compass.enable(timestep)

    # Road alignment constants
    ROAD_TRANSLATION_XZ = (-43.8, 9.2)
    ROAD_YAW = -0.393

    ROAD_WAYPOINTS_LOCAL = [
        (0.0,   0.0,   0.0),
        (25.0,  0.0,   0.0),
        (50.0,  0.0,   0.0),
        (75.0,  0.0,   0.0),
        (100.0, 0.0,   0.0),
        (120.0, 10.0,  0.0),
        (135.0, 25.0,  0.0),
        (150.0, 45.0,  0.0),
        (165.0, 70.0,  0.0),
        (178.0, 98.0,  0.0),
        (188.0, 128.0, 0.0),
        (195.0, 160.0, 0.0),
        (200.0, 200.0, 0.0),
    ]

    ROAD_WAYPOINTS_LOCAL = interpolate_waypoints(ROAD_WAYPOINTS_LOCAL, 1.0)
    waypoints = build_world_waypoints_from_road_local_xy(
        ROAD_WAYPOINTS_LOCAL, ROAD_TRANSLATION_XZ[0], ROAD_TRANSLATION_XZ[1], ROAD_YAW
    )

    # Controller Params
    lookahead_min = 5.0     # Keeps corners tight
    lookahead_max = 15.0    
    lookahead_kv = 1.0      
    WHEELBASE = 2.7
    max_steer = 0.60        
    STEER_SIGN = -1.0  
    
    # OFFSET CONFIGURATION
    # 0.0 targets the exact center of the waypoints
    LATERAL_OFFSET = 0.0 

    v_max = 16.0             
    v_min = 16.0
    k_speed = 3.0           

    # Steering response tuning
    steer_smooth = 0.20      
    speed_smooth = 0.12

    steer_cmd = 0.0
    speed_cmd = 0.0
    last_i = 0

    print("🚗 Full Controller Loaded. Dead Center driving applied.")

    while driver.step() != -1:
        # Get Sensor Data
        pos = gps.getValues()
        x, z = float(pos[0]), float(pos[1])  

        c = compass.getValues()
        yaw = math.atan2(c[0], c[1])  

        v = float(driver.getCurrentSpeed())
        last_i = closest_index(waypoints, x, z, last_i=last_i, window=140)

        # 1. Get standard lookahead point on the centerline
        Ld = clamp(lookahead_min + lookahead_kv * v, lookahead_min, lookahead_max)
        (txw, tzw), target_idx = lookahead_point(waypoints, last_i, Ld)

        # 2. Perpendicular Offset Logic (kept active in case you want to easily change it later)
        (txw_prev, tzw_prev) = waypoints[max(0, target_idx - 2)]
        road_heading = math.atan2((tzw - tzw_prev), (txw - txw_prev))

        txw_final = txw + LATERAL_OFFSET * math.cos(road_heading + math.pi/2)
        tzw_final = tzw + LATERAL_OFFSET * math.sin(road_heading + math.pi/2)

        # 3. Pure Pursuit Calculation
        angle_to_target = math.atan2((tzw_final - z), (txw_final - x))
        alpha = wrap_pi(angle_to_target - yaw)

        kappa = 2.0 * math.sin(alpha) / max(Ld, 1e-3)
        raw_steer = STEER_SIGN * math.atan(WHEELBASE * kappa)
        raw_steer = clamp(raw_steer, -max_steer, max_steer)

        # 4. Velocity and Smoothing
        v_target = v_max * (1.0 / (1.0 + k_speed * abs(kappa)))
        v_target = clamp(v_target, v_min, v_max)

        steer_cmd = (1.0 - steer_smooth) * steer_cmd + steer_smooth * raw_steer
        speed_cmd = (1.0 - speed_smooth) * speed_cmd + speed_smooth * v_target

        # Actuate
        driver.setSteeringAngle(steer_cmd)
        driver.setCruisingSpeed(speed_cmd)

if __name__ == "__main__":
    run()
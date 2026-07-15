from controller import Supervisor, Keyboard
import numpy as np
import os
import math
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("⚠️ WARNING: Ultralytics not installed.")


def clamp(value, low, high):
    return max(low, min(value, high))


def is_valid_car_box(box, width, height, min_conf=0.60):
    conf = float(box.conf[0])
    if conf < min_conf:
        return False

    x1, y1, x2, y2 = map(float, box.xyxy[0])

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 2 or box_h <= 2:
        return False

    box_area_ratio = (box_w * box_h) / float(width * height)
    aspect_ratio = box_w / max(box_h, 1.0)
    cy = (y1 + y2) / 2.0

    if box_area_ratio < 0.0012:
        return False

    if box_area_ratio > 0.55:
        return False

    if aspect_ratio < 0.35 or aspect_ratio > 5.0:
        return False

    if cy > height * 0.94:
        return False

    return True


def spawn_payload(robot, gps_pos, step_count, name_suffix=""):
    px, py, pz = gps_pos
    bomb_def = f"BOMB_{name_suffix}_{step_count}"

    payload_vrml = f"""
    DEF {bomb_def} Solid {{
      translation {px} {py} {pz - 1.2}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor 1 0 0
            roughness 0.5
            metalness 0.5
          }}
          geometry Sphere {{
            radius 0.2
          }}
        }}
      ]
      name "kinetic_payload_{name_suffix}_{step_count}"
      boundingObject Sphere {{
        radius 0.2
      }}
      physics Physics {{
        mass 0.1
      }}
    }}
    """

    robot.getRoot().getField("children").importMFNodeFromString(-1, payload_vrml)

    bomb_node = robot.getFromDef(bomb_def)
    if bomb_node:
        bomb_node.setVelocity(robot.getSelf().getVelocity())


def select_target_box(results, width, height, reticle_x, reticle_y, attack_requested, attack_mode):
    best_box = None
    best_score = -1e9

    if len(results) == 0 or results[0].boxes is None:
        return None

    for box in results[0].boxes:
        if not is_valid_car_box(box, width, height, min_conf=0.60):
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)

        dist_to_reticle = math.hypot(cx - reticle_x, cy - reticle_y)
        dist_norm = dist_to_reticle / max(math.hypot(width, height), 1.0)

        if attack_requested or attack_mode:
            score = conf * 3.0 + area_ratio * 80.0 - dist_norm * 0.8
        else:
            score = conf * 2.0 + area_ratio * 40.0 - dist_norm * 3.0

        if score > best_score:
            best_score = score
            best_box = box

    return best_box


def run_robot():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    # ============================================================
    # HUD RETICLE
    # ============================================================
    reticle_x, reticle_y = None, None
    reticle_speed = 15

    # ============================================================
    # MOTORS
    # ============================================================
    motor_names = [
        "front left propeller",
        "front right propeller",
        "rear left propeller",
        "rear right propeller",
    ]

    motors = [robot.getDevice(n) for n in motor_names]
    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    # ============================================================
    # SENSORS
    # ============================================================
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    gps = robot.getDevice("gps")
    gps.enable(timestep)

    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)

    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    cam_pitch_motor = robot.getDevice("camera pitch")

    CAM_PITCH_MIN = -0.45
    CAM_PITCH_MAX = 1.20

    smoothed_cam_pitch = 0.10
    cam_pitch_base = 0.10

    # ============================================================
    # YOLO MODEL
    # ============================================================
    model = None
    model_name = "super_model.pt"

    if YOLO is not None and os.path.exists(model_name):
        try:
            model = YOLO(model_name)
            print("✅ SYSTEM READY: AIM_Y 0.50 TARGETING + SMOOTH EGRESS")
        except Exception as e:
            print(f"❌ MODEL ERROR: {e}")
            model = None
    else:
        print(f"⚠️ WARNING: '{model_name}' not found or YOLO missing.")

    # ============================================================
    # CAMERA / FOV COMPENSATION
    # ============================================================
    fov_ref = 0.785398
    target_size_ref = 0.60

    try:
        fov_now = float(camera.getFov())
    except Exception:
        fov_now = fov_ref

    target_size = target_size_ref * (
        math.tan(fov_ref / 2.0) / math.tan(fov_now / 2.0)
    )

    print(
        f"📷 FOV now={fov_now:.6f} rad | target_size adjusted to {target_size:.3f} "
        f"(ref {target_size_ref:.2f} @ {fov_ref:.6f})"
    )

    # ============================================================
    # BASE FLIGHT STABILITY
    # ============================================================
    k_vertical_thrust = 70.0
    k_vertical_p = 5.0
    k_roll_p = 50.0
    k_pitch_p = 20.0
    k_pitch_d = 2.0

    normal_altitude = 4.5
    current_base_altitude = normal_altitude

    # ============================================================
    # YAW / TARGET CENTERING
    # ============================================================
    k_yaw_p_manual_track = 0.35
    k_yaw_p_lock = 1.5
    k_yaw_damp = 0.40
    lead_time = 0.20

    # ============================================================
    # ATTACK / RELEASE SETTINGS
    # ============================================================
    drop_line_dist = 5.7

    lock_center_threshold = 0.16
    lock_required_seconds = 0.35
    required_lock_frames = int(lock_required_seconds / dt)

    attack_center_threshold = 0.06

    drop_ready_frames = 0
    required_drop_frames = 1  # Tragere instantă când e la distanța corectă

    overfly_y_threshold = 0.76

    lateral_roll_limit = 0.05
    lateral_roll_vel_limit = 0.25
    lateral_manual_limit = 0.04

    lateral_stable_frames = 0
    required_lateral_stable_frames = max(2, int(0.18 / dt))

    # ============================================================
    # ATTACK SAFETY / SPEED
    # ============================================================
    settle_time = 3.0

    max_pitch_abs = 17.0
    max_backward_pitch = 10.0
    max_pitch_rate = 20.0
    max_speed_limit = 18.0
    accel_ramp = 5.0

    tilt_slow_div = 3.0
    tilt_slow_min = 0.35

    smoothing_factor = 0.25
    current_pitch_force = 0.0

    # ============================================================
    # MANUAL CONTROL
    # ============================================================
    manual_pitch = 0.0
    target_manual_pitch = 0.0

    manual_roll = 0.0
    target_manual_roll = 0.0

    manual_pitch_strength = 16.0
    manual_roll_strength = 4.0

    manual_pitch_smoothing = 8.0
    manual_roll_smoothing = 8.0

    manual_forward_cap = 15.0
    manual_backward_cap = 7.0

    manual_max_pitch_abs = 22.0
    manual_max_pitch_rate = 22.0

    manual_deadband = 0.03

    roll_trim = 0.0

    # ============================================================
    # ATTACK STATE
    # ============================================================
    attack_mode = False
    attack_requested = False
    stable_lock_frames = 0

    attack_entry_frames = 0
    attack_entry_duration = int(0.35 / dt)
    pre_attack_pitch_cmd = 0.0

    # ============================================================
    # DETECTION MEMORY
    # ============================================================
    hold_frames = int(1.2 / dt)
    hold = 0

    last_yaw_cmd = 0.0
    last_pitch_cmd = 0.0

    last_err_x = 0.0
    last_err_x_rate = 0.0

    last_box_ratio_f = target_size
    last_distance_to_car_m = 999.0

    egress_frames = 0
    release_message_frames = 0

    step_count = 0

    print("🚀 DRONE INITIALIZED. FAST MANUAL SEARCH MODE ACTIVE.")
    print("Controls:")
    print("  W/A/S/D = move HUD reticle")
    print("  I/K     = FAST manual forward/backward")
    print("  J/L     = old-style lateral alignment")
    print("  U/O     = altitude up/down")
    print("  SPACE   = stable lock + aim_y 0.50 attack")

    # ============================================================
    # MAIN LOOP
    # ============================================================
    while robot.step(timestep) != -1:
        step_count += 1
        t = step_count * dt

        # ========================================================
        # SENSOR READINGS
        # ========================================================
        roll, pitch, yaw = imu.getRollPitchYaw()

        gps_pos = gps.getValues()
        altitude = max(gps_pos[2], 0.1) if gps_pos else 0.1

        roll_vel, pitch_vel, yaw_vel = gyro.getValues()
        tilt = abs(pitch) + abs(roll)

        width = camera.getWidth()
        height = camera.getHeight()
        img_data = camera.getImage()

        if reticle_x is None and width > 0:
            reticle_x = width // 2
            reticle_y = height // 2

        # ========================================================
        # RESET PER-FRAME INPUTS
        # ========================================================
        target_manual_pitch = 0.0
        target_manual_roll = 0.0

        hud_manual_pitch = 0.0
        hud_manual_roll = 0.0

        # ========================================================
        # WEBOTS KEYBOARD INPUT
        # ========================================================
        key = keyboard.getKey()

        while key != -1:
            if key in (Keyboard.UP, ord("I"), ord("i"), 73, 105):
                target_manual_pitch += -1.0

            elif key in (Keyboard.DOWN, ord("K"), ord("k"), 75, 107):
                target_manual_pitch += 1.0

            elif key in (Keyboard.LEFT, ord("J"), ord("j"), 74, 106):
                target_manual_roll += 1.0

            elif key in (Keyboard.RIGHT, ord("L"), ord("l"), 76, 108):
                target_manual_roll += -1.0

            elif key in (ord("U"), ord("u"), 85, 117):
                current_base_altitude = clamp(current_base_altitude + 2.0 * dt, 2.0, 15.0)

            elif key in (ord("O"), ord("o"), 79, 111):
                current_base_altitude = clamp(current_base_altitude - 2.0 * dt, 2.0, 15.0)

            elif key == ord(" ") and not attack_mode and t > settle_time and egress_frames == 0:
                attack_requested = True
                stable_lock_frames = 0
                drop_ready_frames = 0
                lateral_stable_frames = 0
                print("🎯 SPACE PRESSED: STABLE LOCK REQUESTED")

            key = keyboard.getKey()

        # ========================================================
        # VISION / HUD
        # ========================================================
        best_box = None
        hud_frame = None
        best_conf = 0.0

        if img_data:
            frame = np.frombuffer(img_data, np.uint8).reshape((height, width, 4))
            hud_frame = frame[:, :, :3].copy()

            if reticle_x is not None:
                cv2.drawMarker(
                    hud_frame,
                    (int(reticle_x), int(reticle_y)),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )
                cv2.circle(
                    hud_frame,
                    (int(reticle_x), int(reticle_y)),
                    10,
                    (0, 0, 255),
                    1,
                )

            if model is not None:
                results = model.predict(hud_frame, verbose=False, imgsz=320)

                best_box = select_target_box(
                    results,
                    width,
                    height,
                    reticle_x,
                    reticle_y,
                    attack_requested,
                    attack_mode,
                )

                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if is_valid_car_box(box, width, height, min_conf=0.60):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(hud_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        if hud_frame is None:
            continue

        # ========================================================
        # DEFAULT COMMANDS
        # ========================================================
        yaw_cmd = 0.0
        pitch_cmd = 0.0
        raw_pitch = 0.0
        mode_text = "MANUAL SEARCH"

        if t < settle_time:
            hold = 0
            stable_lock_frames = 0
            drop_ready_frames = 0
            lateral_stable_frames = 0
            attack_entry_frames = 0

            if cam_pitch_motor is not None:
                cam_pitch_motor.setPosition(0.10)

            cv2.putText(
                hud_frame,
                "SYSTEM WARMING UP...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        else:
            # ====================================================
            # TARGET DETECTED
            # ====================================================
            if best_box is not None:
                hold = hold_frames
                best_conf = float(best_box.conf[0])

                x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box_h = float(y2 - y1)

                # =================================================
                # AIM POINT - FIRST VARIANT
                # =================================================
                aim_x = cx
                aim_y = y1 + 0.42 * (y2 - y1)
                aim_y = clamp(aim_y, y1, y2)

                cv2.rectangle(hud_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(hud_frame, (int(aim_x), int(aim_y)), 5, (255, 0, 255), -1)
                cv2.line(
                    hud_frame,
                    (int(reticle_x), int(reticle_y)),
                    (int(cx), int(cy)),
                    (0, 255, 0),
                    2,
                )

                # =================================================
                # CAMERA STABILIZATION
                # =================================================
                err_y = (cy - height * 0.5) / (height / 2.0)

                cam_pitch_base += err_y * 2.5 * dt
                cam_pitch_base = clamp(cam_pitch_base, -0.2, 1.0)

                raw_cam_target = clamp(cam_pitch_base - pitch, -0.45, 1.2)

                smoothed_cam_pitch = (smoothed_cam_pitch * 0.5) + (raw_cam_target * 0.5)
                smoothed_cam_pitch = clamp(smoothed_cam_pitch, -0.45, 1.2)

                if cam_pitch_motor is not None and egress_frames == 0:
                    cam_pitch_motor.setPosition(smoothed_cam_pitch)

                # =================================================
                # FOV + DISTANCE
                # =================================================
                try:
                    fov_now = float(camera.getFov())
                except Exception:
                    fov_now = fov_ref

                target_size = target_size_ref * (
                    math.tan(fov_ref / 2.0) / math.tan(fov_now / 2.0)
                )

                fov_v = 2.0 * math.atan(math.tan(fov_now / 2.0) * (height / width))

                delta = ((height / 2.0) - cy) * (fov_v / height)
                alpha_angle = smoothed_cam_pitch + abs(pitch)
                beta = alpha_angle - delta

                if beta > 0.05:
                    distance_to_car_m = altitude / math.tan(beta)
                else:
                    distance_to_car_m = 999.0

                last_distance_to_car_m = distance_to_car_m

                # =================================================
                # IMAGE ERRORS
                # =================================================
                box_ratio = float(box_h) / float(height)

                box_ratio_f = (1.0 - 0.40) * last_box_ratio_f + 0.40 * box_ratio
                last_box_ratio_f = box_ratio_f

                err_x = (cx - width * 0.5) / (width / 2.0)
                aim_err_x = (aim_x - width * 0.5) / (width / 2.0)

                err_rate = (err_x - last_err_x) / max(dt, 1e-6)
                last_err_x_rate = 0.7 * last_err_x_rate + 0.3 * err_rate

                err_x_lead = clamp(err_x + last_err_x_rate * lead_time, -1.2, 1.2)
                last_err_x = err_x

                cy_norm = cy / float(height)
                aim_y_norm = aim_y / float(height)

                # =================================================
                # LOCK LOGIC
                # =================================================
                target_centered_for_lock = abs(err_x) < lock_center_threshold
                target_visible_for_lock = box_ratio_f > target_size * 0.30

                if target_centered_for_lock and target_visible_for_lock:
                    stable_lock_frames += 1
                else:
                    stable_lock_frames = 0

                if attack_requested and not attack_mode:
                    mode_text = "LOCKING - CENTERING TARGET"

                    yaw_cmd = clamp(-k_yaw_p_lock * err_x_lead, -1.0, 1.0)

                    if abs(err_x) < 0.25 and box_ratio_f < target_size * 0.75:
                        raw_pitch = -3.0
                    else:
                        raw_pitch = 0.0

                    if stable_lock_frames >= required_lock_frames:
                        attack_mode = True
                        attack_requested = False
                        stable_lock_frames = 0
                        drop_ready_frames = 0
                        lateral_stable_frames = 0

                        pre_attack_pitch_cmd = last_pitch_cmd
                        attack_entry_frames = attack_entry_duration

                        print("⚔️ STABLE LOCK CONFIRMED: SMOOTH ATTACK ACTIVE")

                elif attack_mode:
                    # =================================================
                    # ATTACK MODE WITH LATERAL STABILITY CHECK
                    # =================================================
                    mode_text = "ATTACK MODE - STABLE TARGETING"

                    if box_ratio_f > 0.70 or cy_norm > overfly_y_threshold:
                        yaw_cmd = last_yaw_cmd
                    else:
                        yaw_cmd = clamp(-k_yaw_p_lock * err_x_lead, -1.0, 1.0)

                    attack_target_pitch = -15.0

                    center_slow = clamp(1.0 - abs(err_x) * 0.6, 0.65, 1.0)
                    attack_target_pitch *= center_slow

                    if cy_norm > overfly_y_threshold:
                        mode_text = "ATTACK BRAKE - RELEASE ZONE"
                        attack_target_pitch = -4.0

                    lateral_currently_stable = (
                        abs(roll) < lateral_roll_limit
                        and abs(roll_vel) < lateral_roll_vel_limit
                        and abs(manual_roll) < lateral_manual_limit
                    )

                    if lateral_currently_stable:
                        lateral_stable_frames += 1
                    else:
                        lateral_stable_frames = 0

                    if lateral_stable_frames < required_lateral_stable_frames:
                        mode_text = "ATTACK STABILIZING LATERAL"
                        attack_target_pitch = min(attack_target_pitch, -8.0)

                    if attack_entry_frames > 0:
                        blend = 1.0 - (attack_entry_frames / max(1, attack_entry_duration))
                        raw_pitch = (1.0 - blend) * pre_attack_pitch_cmd + blend * attack_target_pitch
                        attack_entry_frames -= 1
                    else:
                        raw_pitch = attack_target_pitch

                else:
                    # =================================================
                    # FAST MANUAL MODE
                    # =================================================
                    mode_text = "FAST MANUAL SEARCH - TARGET VISIBLE"

                    yaw_cmd = clamp(-k_yaw_p_manual_track * err_x_lead, -0.35, 0.35)

                    raw_pitch = manual_pitch * manual_pitch_strength

                    if raw_pitch < 0.0:
                        raw_pitch = clamp(raw_pitch, -manual_forward_cap, 0.0)
                    else:
                        raw_pitch = clamp(raw_pitch, 0.0, manual_backward_cap)

                # =================================================
                # DROP LINE VISUAL
                # =================================================
                trigger_beta = math.atan(altitude / max(0.1, drop_line_dist))
                trigger_delta = alpha_angle - trigger_beta

                trigger_y = int((height / 2.0) - (trigger_delta * height / fov_v))
                trigger_y = int(clamp(trigger_y, 0, height - 1))

                line_color = (0, 0, 255) if attack_mode else (0, 165, 255)
                cv2.line(hud_frame, (0, trigger_y), (width, trigger_y), line_color, 2)

                # =================================================
                # RELEASE EVENT WITH AIM_Y 0.50
                # =================================================
                target_centered_for_drop = abs(aim_err_x) < attack_center_threshold

                # Fereastra ideală găsită anterior
                distance_good = 3.0 <= distance_to_car_m <= 4.2

                box_good = box_ratio_f > target_size * 0.75
                vertical_good = aim_y > height * 0.15 and aim_y < height * 0.85
                lateral_stable_enough = lateral_stable_frames >= required_lateral_stable_frames

                if (
                    attack_mode
                    and target_centered_for_drop
                    and distance_good
                    and box_good
                    and vertical_good
                    and lateral_stable_enough
                ):
                    drop_ready_frames += 1
                else:
                    drop_ready_frames = 0

                if attack_mode and drop_ready_frames >= required_drop_frames:
                    print(f"✅ RELEASE EVENT AT {distance_to_car_m:.1f}m!")
                    
                    spawn_payload(robot, gps_pos, step_count, "STRIKE")
                    release_message_frames = int(1.5 / dt)

                    attack_mode = False
                    attack_requested = False
                    stable_lock_frames = 0
                    drop_ready_frames = 0
                    lateral_stable_frames = 0
                    attack_entry_frames = 0

                    # Păstrăm valorile de pitch curente neatinse, pentru ca rampa din Egress să aibă de unde să plece!
                    # AM ȘTERS hard-reset-urile care aruncau drona pe spate.

                    manual_pitch = 0.0
                    manual_roll = 0.0
                    target_manual_pitch = 0.0
                    target_manual_roll = 0.0

                    hold = 0
                    last_err_x = 0.0
                    last_err_x_rate = 0.0

                    egress_frames = int(4.5 / dt)
                    current_base_altitude = clamp(current_base_altitude + 2.0, 5.5, 9.0)

                    cam_pitch_base = 0.10
                    smoothed_cam_pitch = 0.10

                    if cam_pitch_motor is not None:
                        cam_pitch_motor.setPosition(0.10)

                # =================================================
                # SAFETY SHAPING
                # =================================================
                speed_limit = min(max_speed_limit, (t - settle_time) * accel_ramp)

                if not attack_mode and egress_frames == 0:
                    tilt_factor = clamp(1.0 - tilt / 1.6, 0.60, 1.0)
                    raw_pitch *= tilt_factor

                    tilt_slow = clamp(1.0 - (tilt / tilt_slow_div), tilt_slow_min, 1.0)
                    raw_pitch *= tilt_slow

                if attack_mode:
                    active_pitch_abs = max_pitch_abs
                    active_pitch_rate = max_pitch_rate
                else:
                    active_pitch_abs = manual_max_pitch_abs
                    active_pitch_rate = manual_max_pitch_rate

                # Aplicăm limitele pe pitch normal doar dacă NU suntem în Egress
                if egress_frames == 0:
                    raw_pitch = clamp(raw_pitch, -speed_limit, speed_limit)
                    raw_pitch = clamp(raw_pitch, -active_pitch_abs, max_backward_pitch)
                    max_step = active_pitch_rate * dt
                    pitch_cmd = clamp(raw_pitch, last_pitch_cmd - max_step, last_pitch_cmd + max_step)
                    last_pitch_cmd = pitch_cmd
                    last_yaw_cmd = yaw_cmd

                # =================================================
                # HUD TEXT
                # =================================================
                cv2.putText(hud_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
                cv2.putText(hud_frame, f"conf={best_conf:.2f} err_x={err_x:+.2f} aimX={aim_err_x:+.2f} dist={distance_to_car_m:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(hud_frame, f"dropReady={drop_ready_frames}/{required_drop_frames} lat={lateral_stable_frames}/{required_lateral_stable_frames}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(hud_frame, f"cy={cy_norm:.2f} aimY={aim_y_norm:.2f} roll={roll:+.2f} pitchCmd={pitch_cmd:+.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # ====================================================
            # TARGET NOT DETECTED
            # ====================================================
            else:
                stable_lock_frames = 0
                drop_ready_frames = 0
                lateral_stable_frames = 0

                if attack_requested:
                    mode_text = "ATTACK REQUESTED - NO VALID TARGET"
                else:
                    mode_text = "SEARCHING / FAST MANUAL MODE"

                if attack_mode and last_distance_to_car_m <= 4.0:
                    print("⚠️ TARGET LOST CLOSE - ABORT / RECOVERY")

                    spawn_payload(robot, gps_pos, step_count, "BLIND_CLOSE")

                    attack_mode = False
                    attack_requested = False
                    attack_entry_frames = 0

                    # La fel, eliminăm hard-reseturile de pitch pentru o revenire lină
                    manual_pitch = 0.0
                    manual_roll = 0.0
                    target_manual_pitch = 0.0
                    target_manual_roll = 0.0

                    hold = 0
                    last_err_x = 0.0
                    last_err_x_rate = 0.0

                    egress_frames = int(4.5 / dt)
                    current_base_altitude = clamp(current_base_altitude + 2.0, 5.5, 9.0)

                    cam_pitch_base = 0.10
                    smoothed_cam_pitch = 0.10

                    if cam_pitch_motor is not None:
                        cam_pitch_motor.setPosition(0.10)

                elif hold > 0:
                    hold -= 1

                    last_yaw_cmd *= 0.75
                    last_pitch_cmd *= 0.90

                    yaw_cmd = last_yaw_cmd
                    pitch_cmd = last_pitch_cmd

                    if cam_pitch_motor is not None and egress_frames == 0:
                        safe_search_pitch = 0.10
                        smoothed_cam_pitch = (smoothed_cam_pitch * 0.95) + (safe_search_pitch * 0.05)
                        smoothed_cam_pitch = clamp(smoothed_cam_pitch, CAM_PITCH_MIN, CAM_PITCH_MAX)
                        cam_pitch_motor.setPosition(smoothed_cam_pitch)

                    mode_text = "COASTING - TARGET FLICKER"

                else:
                    if attack_mode:
                        print("⚠️ TARGET LOST - ABORT / RECOVERY")

                        spawn_payload(robot, gps_pos, step_count, "BLIND_LOST")

                        attack_mode = False
                        attack_requested = False
                        attack_entry_frames = 0

                        manual_pitch = 0.0
                        manual_roll = 0.0
                        target_manual_pitch = 0.0
                        target_manual_roll = 0.0

                        hold = 0
                        last_err_x = 0.0
                        last_err_x_rate = 0.0

                        egress_frames = int(4.5 / dt)
                        current_base_altitude = clamp(current_base_altitude + 2.0, 5.5, 9.0)

                        cam_pitch_base = 0.10
                        smoothed_cam_pitch = 0.10

                        if cam_pitch_motor is not None:
                            cam_pitch_motor.setPosition(0.10)

                    raw_pitch = manual_pitch * manual_pitch_strength

                    if raw_pitch < 0:
                        raw_pitch = clamp(raw_pitch, -manual_forward_cap, 0.0)
                    else:
                        raw_pitch = clamp(raw_pitch, 0.0, manual_backward_cap)

                    # Limităm și aici doar dacă NU suntem în Egress
                    if egress_frames == 0:
                        speed_limit = min(max_speed_limit, (t - settle_time) * accel_ramp)
                        raw_pitch = clamp(raw_pitch, -speed_limit, speed_limit)
                        raw_pitch = clamp(raw_pitch, -manual_max_pitch_abs, max_backward_pitch)

                        max_step = manual_max_pitch_rate * dt
                        pitch_cmd = clamp(raw_pitch, last_pitch_cmd - max_step, last_pitch_cmd + max_step)
                        last_pitch_cmd = pitch_cmd
                        last_yaw_cmd *= 0.85
                        yaw_cmd = last_yaw_cmd

                    if cam_pitch_motor is not None and egress_frames == 0:
                        safe_search_pitch = 0.10
                        smoothed_cam_pitch = (smoothed_cam_pitch * 0.95) + (safe_search_pitch * 0.05)
                        smoothed_cam_pitch = clamp(smoothed_cam_pitch, CAM_PITCH_MIN, CAM_PITCH_MAX)
                        cam_pitch_motor.setPosition(smoothed_cam_pitch)

                cv2.putText(hud_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(hud_frame, f"manual I/K={manual_pitch:+.2f} J/L={manual_roll:+.2f} pitchCmd={pitch_cmd:+.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ========================================================
        # RELEASE MESSAGE
        # ========================================================
        if release_message_frames > 0:
            release_message_frames -= 1
            cv2.putText(hud_frame, "RELEASE EVENT - AIM_Y 0.50", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # ========================================================
        # OPENCV HUD INPUT
        # ========================================================
        cv2.imshow("Targeting HUD", hud_frame)
        cv_key = cv2.waitKey(1) & 0xFF

        if cv_key in (ord("w"), ord("W")): reticle_y -= reticle_speed
        elif cv_key in (ord("s"), ord("S")): reticle_y += reticle_speed
        elif cv_key in (ord("a"), ord("A")): reticle_x -= reticle_speed
        elif cv_key in (ord("d"), ord("D")): reticle_x += reticle_speed
        elif cv_key in (ord("i"), ord("I")): hud_manual_pitch += -1.0
        elif cv_key in (ord("k"), ord("K")): hud_manual_pitch += 1.0
        elif cv_key in (ord("j"), ord("J")): hud_manual_roll += 1.0
        elif cv_key in (ord("l"), ord("L")): hud_manual_roll += -1.0
        elif cv_key in (ord("u"), ord("U")): current_base_altitude = clamp(current_base_altitude + 2.0 * dt, 2.0, 15.0)
        elif cv_key in (ord("o"), ord("O")): current_base_altitude = clamp(current_base_altitude - 2.0 * dt, 2.0, 15.0)
        elif cv_key == ord(" ") and not attack_mode and t > settle_time and egress_frames == 0:
            attack_requested = True
            stable_lock_frames = 0
            drop_ready_frames = 0
            lateral_stable_frames = 0
            print("🎯 SPACE PRESSED: STABLE LOCK REQUESTED")

        if reticle_x is not None:
            reticle_x = int(clamp(reticle_x, 0, width - 1))
            reticle_y = int(clamp(reticle_y, 0, height - 1))

        # ========================================================
        # COMBINE INPUTS + DIAGONAL NORMALIZATION
        # ========================================================
        target_manual_pitch += hud_manual_pitch
        target_manual_roll += hud_manual_roll

        target_manual_pitch = clamp(target_manual_pitch, -1.0, 1.0)
        target_manual_roll = clamp(target_manual_roll, -1.0, 1.0)

        mag = math.sqrt(target_manual_pitch * target_manual_pitch + target_manual_roll * target_manual_roll)
        if mag > 1.0:
            target_manual_pitch /= mag
            target_manual_roll /= mag

        manual_pitch += (target_manual_pitch - manual_pitch) * manual_pitch_smoothing * dt
        manual_roll += (target_manual_roll - manual_roll) * manual_roll_smoothing * dt

        if abs(manual_pitch) < manual_deadband: manual_pitch = 0.0
        if abs(manual_roll) < manual_deadband: manual_roll = 0.0

        manual_pitch = clamp(manual_pitch, -1.0, 1.0)
        manual_roll = clamp(manual_roll, -1.0, 1.0)

        # ========================================================
        # ATTACK / EGRESS: FADE MANUAL RESIDUE
        # ========================================================
        if attack_mode or egress_frames > 0:
            target_manual_pitch = 0.0
            target_manual_roll = 0.0
            manual_pitch *= 0.85
            manual_roll *= 0.70

        # ========================================================
        # POST-EVENT EGRESS (Smoothed Recovery)
        # ========================================================
        if egress_frames > 0:
            egress_frames -= 1
            
            # Resetăm input-urile manuale, Egress le ignoră
            manual_pitch = 0.0; manual_roll = 0.0; target_manual_pitch = 0.0; target_manual_roll = 0.0
            
            # Controlul direcției (Yaw)
            yaw_cmd = 0.0; last_yaw_cmd = 0.0

            # Urcăm controlat
            current_base_altitude = clamp(current_base_altitude + 0.7 * dt, 5.5, 9.0)

            # --- TRANZIȚIA LINĂ A PITCH-ULUI ---
            target_egress_pitch = 0.0
            pitch_recovery_smoothness = 4.0 
            
            pitch_cmd = last_pitch_cmd + clamp(target_egress_pitch - last_pitch_cmd, -pitch_recovery_smoothness * dt, pitch_recovery_smoothness * dt)
            last_pitch_cmd = pitch_cmd
            
            # Reducem forța de inerție pe pitch ca drona să nu se dea peste cap de la impuls
            current_pitch_force *= 0.70

            # Camera Auto-Center
            cam_pitch_base = 0.10
            smoothed_cam_pitch = 0.10
            if cam_pitch_motor is not None:
                cam_pitch_motor.setPosition(0.10)

            cv2.putText(hud_frame, "POST-EVENT RECOVERY - SMOOTH HOVER", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ========================================================
        # FINAL MIXING
        # ========================================================
        current_pitch_force = (
            current_pitch_force * (1.0 - smoothing_factor)
        ) + (
            pitch_cmd * smoothing_factor
        )

        # ========================================================
        # OLD-STYLE ROLL CONTROL / MANUAL LEFT-RIGHT
        # ========================================================
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_vel

        lateral_force = manual_roll * manual_roll_strength

        if not attack_mode and egress_frames == 0:
            roll_input += lateral_force

        roll_input += roll_trim

        pitch_input = (
            k_pitch_p * clamp(pitch, -1, 1)
            + k_pitch_d * pitch_vel
            + current_pitch_force
        )

        yaw_final = clamp(yaw_cmd - yaw_vel * k_yaw_damp, -1.0, 1.0)

        # ========================================================
        # ALTITUDE HOLD
        # ========================================================
        desired_alt = current_base_altitude

        clamped_diff_alt = clamp(desired_alt - altitude + 0.6, -1.0, 1.0)
        vertical_input = k_vertical_p * pow(clamped_diff_alt, 3.0)

        safe_pitch = clamp(pitch, -1.0, 1.0)
        safe_roll = clamp(roll, -1.0, 1.0)

        tilt_factor_lift = 1.0 / max(0.5, math.cos(safe_pitch) * math.cos(safe_roll))
        lift_boost = k_vertical_thrust * (tilt_factor_lift - 1.0)
        lift_boost = clamp(lift_boost, 0.0, 15.0)

        total_thrust = k_vertical_thrust + vertical_input + lift_boost

        # ========================================================
        # MOTOR MIXING
        # ========================================================
        m1 = total_thrust - roll_input + pitch_input - yaw_final
        m2 = total_thrust + roll_input + pitch_input + yaw_final
        m3 = total_thrust - roll_input - pitch_input + yaw_final
        m4 = total_thrust + roll_input - pitch_input - yaw_final

        motors[0].setVelocity(clamp(m1, -600, 600))
        motors[1].setVelocity(clamp(-m2, -600, 600))
        motors[2].setVelocity(clamp(-m3, -600, 600))
        motors[3].setVelocity(clamp(m4, -600, 600))


if __name__ == "__main__":
    run_robot()
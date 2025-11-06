# Authors: Nopphakorn Subs. Niwatchai Wang. Supasate Wor. Narith Tha. Tawan Thaep. Napatharak Muan.
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan, BatteryState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import math

def clamp(v, lo, hi):
    """Clamp value v between lo and hi."""
    return max(lo, min(hi, v))

def normalize_angle_rad(a):
    """Map angle to (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a

def deg2rad(d): 
    return d * math.pi / 180.0

def rad2deg(r): 
    return r * 180.0 / math.pi

def quat2yaw(q):
    """Return yaw (rad) from geometry_msgs/Quaternion."""
    w, x, y, z = q.w, q.x, q.y, q.z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# --- Tunables ---
Kp_ang = 2.0          # angular P gain (rad/s per rad of error)
Kp_lin = 0.6          # linear P gain  (m/s per m of error)
LIN_MAX = 0.25        # max |linear x| m/s
ANG_MAX = 1.2         # max |angular z| rad/s
YAW_TOL_DEG = 0.5     # stop turning within this tolerance
POS_TOL_M = 0.05      # stop at goal if closer than this
DT = 0.1              # control period (s)

# Turn and GoTo specific parameters
Kp_ang_turn = 2.0     # rad/s per rad (TurnTo)
Kp_ang_track = 2.5    # rad/s per rad (GoTo heading correction)
ANG_MAX_TRACK = 1.2   # max angular velocity for heading correction
DEAD_DEG = 0.005        # no turning when nearly aligned
OFF_DEG_SLOW = 20.0   # slow forward if off-heading more than this

class Turtlebot3Controller(Node):
    def __init__(self):
        super().__init__('turtlebot3_controller')
        
        # Publishers and Subscribers
        self.cmdVelPublisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scanSubscriber = self.create_subscription(
            LaserScan, 'scan', self.scanCallback, qos_profile=qos_profile_sensor_data)
        self.batteryStateSubscriber = self.create_subscription(
            BatteryState, 'battery_state', self.batteryStateCallback, 10)
        self.odomSubscriber = self.create_subscription(
            Odometry, 'odom', self.odomCallback, 10)

        # Sensor data storage
        self.valueLaserRaw = {
            'range_min': 0.0,
            'range_max': 0.0,
            'ranges': [0.0] * 360,
        }
        self.valueBatteryState = None

        # Odometry-derived state
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw = 0.0
        self.have_odom = False
        self.init_yaw = None
        self.init_done = False

        # State Machine
        self.state = 0

        # Shared state for relative motions
        self.mid = {
            "active": False,
            "type": None,
            "ref_x": 0.0,
            "ref_y": 0.0,
            "ref_yaw": 0.0,
            "target": 0.0,
            "acc_dist": 0.0
        }

        self.timer = self.create_timer(DT, self.timerCallback)
        
        self.actions = [
            lambda: GoUntil(self, {"north": True}),
            lambda: TurnTo(self, -90)
        ]
        
        self.get_logger().info('TurtleBot3 Controller initialized')

    def publishVelocityCommand(self, linearVelocity, angularVelocity):
        """Publish velocity commands to cmd_vel topic."""
        msg = Twist()
        msg.linear.x = float(linearVelocity)
        msg.angular.z = float(angularVelocity)
        self.cmdVelPublisher.publish(msg)

    # --- Subscribers ---
    def scanCallback(self, msg: LaserScan):
        """Handle laser scan data."""
        self.valueLaserRaw = {
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'ranges': list(msg.ranges),
        }

    def batteryStateCallback(self, msg: BatteryState):
        """Handle battery state data."""
        self.valueBatteryState = msg

    def odomCallback(self, msg: Odometry):
        """Handle odometry data."""
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.yaw = quat2yaw(msg.pose.pose.orientation)
        
        if not self.have_odom:
            self.have_odom = True
            self.init_yaw = self.yaw
            self.get_logger().info(f'Initial yaw set to {rad2deg(self.init_yaw):.1f} degrees')

    def timerCallback(self):
        """Main control loop."""            
        if not self.init_done:
            self.init_done = True
            self.get_logger().info(f'Controller initialized at position ({self.pos_x:.2f}, {self.pos_y:.2f})')
            
        # if self.actions[self.state]():
        #     self.state = (self.state + 1) % len(self.actions)
        # self.get_logger().info(f"{DetectSurroundings(self)}")
        if self.state == 0:
            # self.get_logger().info("State 0: Going until North wall detected...")
            if GoToNextNode(self, prefer_turn='left'):
                self.state = 0
                self.get_logger().info("Continue")
        elif self.state == 1:
            robotStop(self)
            # self.get_logger().info("State 1: Turning right 90 degrees...")
            # if TurnTo(self, -90):
            #     self.state = 0
                # self.get_logger().info("Turn complete. Transitioning to State 0.")

def robotStop(node: Turtlebot3Controller):
    """Stop the robot."""
    node.publishVelocityCommand(0.0, 0.0)
    
def DetectSurroundings(self, wall_threshold=0.18):
    """
    Detects if there is a wall in the North, South, East, and West directions
    relative to the robot's current orientation.
    Returns a dictionary with boolean values for each direction.
    """
    if not self.valueLaserRaw['ranges']:
        return {'north': False, 'east': False, 'south': False, 'west': False}

    ranges = list(self.valueLaserRaw['ranges'])
    rmax = self.valueLaserRaw['range_max']
    
    # Replace 0.0 (invalid scan) with max range
    ranges = [rmax if r == 0.0 else r for r in ranges]
    
    # North: front sector (e.g., -30 to +30 degrees)
    north_ranges = ranges[0:2] + ranges[358:360]
    min_north_dist = min(north_ranges) if north_ranges else rmax

    # East: right sector (e.g., 60 to 120 degrees)
    east_ranges = ranges[268:272]
    min_east_dist = min(east_ranges) if east_ranges else rmax

    # South: back sector (e.g., 150 to 210 degrees)
    south_ranges = ranges[178:182]
    min_south_dist = min(south_ranges) if south_ranges else rmax

    # West: left sector (e.g., 240 to 300 degrees)
    west_ranges = ranges[88:92]
    min_west_dist = min(west_ranges) if west_ranges else rmax

    surroundings = {
        'north': min_north_dist < wall_threshold,
        'east': min_east_dist < wall_threshold,
        'south': min_south_dist < wall_threshold,
        'west': min_west_dist < wall_threshold,
    }
    return surroundings

def CorridorFollow(turtle,rel_distance_cm):
    """
    Move forward by 'rel_distance_cm' centimeters from the starting position.
    Returns True when movement is complete.
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    # Initialize on first call or when switching from another behavior
    if not turtle.mid["active"] or turtle.mid["type"] != "goto":
        turtle.mid.update({
            "active": True,
            "type": "goto",
            "ref_x": turtle.pos_x,
            "ref_y": turtle.pos_y,
            "ref_yaw": turtle.yaw,
            "target": rel_distance_cm / 100.0,  # Convert cm to meters
            "acc_dist": 0,
            "startup_counter": 0,  # Add counter for startup nudge
        })
        turtle.get_logger().info(f'Starting forward movement of {rel_distance_cm} cm')

    # Calculate forward progress along the initial heading
    dx = turtle.pos_x - turtle.mid["ref_x"]
    dy = turtle.pos_y - turtle.mid["ref_y"]
    
    # Project displacement onto initial heading direction
    cos_yaw = math.cos(turtle.mid["ref_yaw"])
    sin_yaw = math.sin(turtle.mid["ref_yaw"])
    forward_distance = pow(pow(dx * cos_yaw,2) + pow(dy * sin_yaw,2), 0.5)
    turtle.mid.update({
        "acc_dist": turtle.mid["acc_dist"] + abs(forward_distance),
        "ref_x": turtle.pos_x,
        "ref_y": turtle.pos_y
        })
    
    remaining = turtle.mid["target"] - turtle.mid["acc_dist"]
    print(remaining)

    # Check if we've reached the target
    if remaining <= POS_TOL_M:
        turtle.publishVelocityCommand(0.0, 0.0)
        turtle.mid["active"] = False
        turtle.get_logger().info(f'GoTo completed (remaining: {remaining*100:.1f} cm)')
        return True

    # Startup nudge to overcome static friction
    turtle.mid["startup_counter"] += 1
    if turtle.mid["startup_counter"] <= 3:  # First 3 cycles (0.3 seconds)
        # Give a strong initial push to get moving
        turtle.publishVelocityCommand(0.15, 0.0)
        return False

    # Normal proportional control after startup
    # Ensure minimum velocity to keep moving
    min_velocity = 0.08  # Minimum velocity to maintain motion
    vx = clamp(Kp_lin * remaining, min_velocity, LIN_MAX)
    
    # For very short distances, use a higher minimum to ensure completion
    if remaining < 0.3:  # Less than 30cm remaining
        vx = max(vx, 0.1)

    # Heading correction to maintain straight line
    yaw_err = normalize_angle_rad(turtle.mid["ref_yaw"] - turtle.yaw)
    err_deg = abs(rad2deg(yaw_err))

    ranges = turtle.valueLaserRaw['ranges']
    rmax = turtle.valueLaserRaw['range_max']
    ranges = [rmax if r == 0.0 else r for r in ranges]
    left_wall = min(ranges[20:60])
    right_wall = min(ranges[360-60:360-20])
    front_wall = min(ranges[:30] + ranges[330:])

    kp_ang = 1.8
    ang_err = left_wall - right_wall
    print("Wall Error: ", ang_err)
    ang_deadzone = 0.06
    #เเก้ตรงนี้
    if(abs(ang_err) > ang_deadzone):
        turtle.publishVelocityCommand(0.10, ang_err * kp_ang)
    else:
        print("Going Forward")

    if(front_wall < 0.10):
        turtle.publishVelocityCommand(0, 0)
        return remaining

    if err_deg <= DEAD_DEG:
        wz = 0.0
    else:
        wz = clamp(Kp_ang_track * yaw_err, -ANG_MAX_TRACK, ANG_MAX_TRACK)
        # Slow down if significantly off-heading, but maintain minimum
        if err_deg > OFF_DEG_SLOW:
            vx = max(vx * 0.5, min_velocity)

    
    return False

def TurnTo(turtle, rel_angle_deg) -> bool:
    """
    Relative turn: rotate by 'rel_angle_deg' CCW from the pose at first call.
    Returns True when turn is complete.
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    # Initialize on first call or when switching from another behavior
    # Use .get() to safely check for keys
    if not turtle.mid.get("active") or turtle.mid.get("type") != "turn":
        turtle.mid.update({
            "active": True,
            "type": "turn",
            "ref_yaw": turtle.yaw,
            "target": float(rel_angle_deg)
        })
        turtle.get_logger().info(f'Starting turn of {rel_angle_deg} degrees')

    # Calculate target yaw and current error
    target_yaw = normalize_angle_rad(turtle.mid["ref_yaw"] + deg2rad(turtle.mid["target"]))
    err = normalize_angle_rad(target_yaw - turtle.yaw)
    err_deg = abs(rad2deg(err))

    # Check if we've reached the target
    if err_deg <= YAW_TOL_DEG:
        turtle.publishVelocityCommand(0.0, 0.0)
        turtle.mid["active"] = False
        turtle.get_logger().info(f'Turn completed (error: {err_deg:.1f} deg)')
        return True

    # Apply proportional control
    wz = clamp(Kp_ang_turn * err, -ANG_MAX, ANG_MAX)
    turtle.publishVelocityCommand(0.0, wz)
    return False

def GoTo(turtle: Turtlebot3Controller, rel_distance_cm: float) -> bool:
    """
    Move forward by 'rel_distance_cm' centimeters from the starting position.
    Returns True when movement is complete.
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    # Initialize on first call or when switching from another behavior
    # Use .get() to safely check for keys
    if not turtle.mid.get("active") or turtle.mid.get("type") != "goto":
        turtle.mid.update({
            "active": True,
            "type": "goto",
            "ref_x": turtle.pos_x,
            "ref_y": turtle.pos_y,
            "ref_yaw": turtle.yaw,
            "target": rel_distance_cm / 100.0,  # Convert cm to meters
            "startup_counter": 0  # Add counter for startup nudge
        })
        turtle.get_logger().info(f'Starting forward movement of {rel_distance_cm} cm')

    # Calculate forward progress along the initial heading
    dx = turtle.pos_x - turtle.mid["ref_x"]
    dy = turtle.pos_y - turtle.mid["ref_y"]
    
    # Project displacement onto initial heading direction
    cos_yaw = math.cos(turtle.mid["ref_yaw"])
    sin_yaw = math.sin(turtle.mid["ref_yaw"])
    forward_distance = dx * cos_yaw + dy * sin_yaw
    
    remaining = turtle.mid["target"] - forward_distance

    # Check if we've reached the target
    if remaining <= POS_TOL_M:
        turtle.publishVelocityCommand(0.0, 0.0)
        turtle.mid["active"] = False
        turtle.get_logger().info(f'GoTo completed (remaining: {remaining*100:.1f} cm)')
        return True

    # Startup nudge to overcome static friction
    turtle.mid["startup_counter"] += 1
    if turtle.mid["startup_counter"] <= 3:  # First 3 cycles (0.3 seconds)
        # Give a strong initial push to get moving
        turtle.publishVelocityCommand(0.15, 0.0)
        return False

    # Normal proportional control after startup
    # Ensure minimum velocity to keep moving
    min_velocity = 0.08  # Minimum velocity to maintain motion
    vx = clamp(Kp_lin * remaining, min_velocity, LIN_MAX)
    
    # For very short distances, use a higher minimum to ensure completion
    if remaining < 0.3:  # Less than 30cm remaining
        vx = max(vx, 0.1)

    # Check for obstacles in front
    ranges = turtle.valueLaserRaw.get('ranges', [])
    if ranges and len(ranges) == 360:
        rmax = turtle.valueLaserRaw.get('range_max', float('inf'))
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]
        
        # Check front sector (0 to 30 degrees and 330 to 360 degrees)
        front_ranges = ranges[0:30] + ranges[330:360]
        min_front_dist = min(front_ranges) if front_ranges else rmax
        
        # Safety threshold - stop if wall is too close
        WALL_SAFETY_DISTANCE = 0.15  # 15cm safety margin
        
        if min_front_dist < WALL_SAFETY_DISTANCE:
            turtle.publishVelocityCommand(0.0, 0.0)
            turtle.mid["active"] = False
            turtle.get_logger().warn(f'GoTo stopped early - wall detected at {min_front_dist*100:.1f}cm (remaining: {remaining*100:.1f}cm)')
            return True  # Return True to indicate completion (even if not at exact target)
        
        # Slow down if approaching wall
        WALL_SLOWDOWN_DISTANCE = 0.25  # 25cm - start slowing down
        if min_front_dist < WALL_SLOWDOWN_DISTANCE:
            slowdown_factor = (min_front_dist - WALL_SAFETY_DISTANCE) / (WALL_SLOWDOWN_DISTANCE - WALL_SAFETY_DISTANCE)
            vx = vx * slowdown_factor
            vx = max(vx, min_velocity * 0.5)  # Ensure some minimum movement

    # Heading correction to maintain straight line
    yaw_err = normalize_angle_rad(turtle.mid["ref_yaw"] - turtle.yaw)
    err_deg = abs(rad2deg(yaw_err))

    if err_deg <= DEAD_DEG:
        wz = 0.0
    else:
        wz = clamp(Kp_ang_track * yaw_err, -ANG_MAX_TRACK, ANG_MAX_TRACK)
        # Slow down if significantly off-heading, but maintain minimum
        if err_deg > OFF_DEG_SLOW:
            vx = max(vx * 0.5, min_velocity)

    turtle.publishVelocityCommand(vx, wz)
    return False

def GoUntil(turtle: Turtlebot3Controller, directions_to_monitor: dict[str, bool]) -> bool:
    """
    Move forward until a wall is detected in any of the specified directions that are set to True for monitoring.
    Returns True when a wall is detected in any of the given monitored directions.
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    # Initialize on first call or when switching from another behavior
    if not turtle.mid["active"] or turtle.mid["type"] != "gountil":
        turtle.mid.update({
            "active": True,
            "type": "gountil",
            "directions_to_monitor": directions_to_monitor,
            "startup_counter": 0
        })
        turtle.get_logger().info(f'Starting GoUntil, monitoring directions: {directions_to_monitor}')

    surroundings = DetectSurroundings(turtle)

    stop_condition_met = True
    for direction, monitor_status in turtle.mid["directions_to_monitor"].items():
        if monitor_status != surroundings[direction]:
            stop_condition_met = False
            break

    if stop_condition_met:
        turtle.publishVelocityCommand(0.0, 0.0)
        turtle.mid["active"] = False
        turtle.get_logger().info(f'Wall detected in one of the monitored directions. GoUntil completed.')
        return True

    # Startup nudge to overcome static friction
    turtle.mid["startup_counter"] += 1
    if turtle.mid["startup_counter"] <= 3:  # First 3 cycles (0.3 seconds)
        turtle.publishVelocityCommand(0.15, 0.0)
        return False

    # Continue moving forward
    turtle.publishVelocityCommand(0.1, 0.0) # Simple forward movement
    return False


def AlignSelf(turtle, side='left') -> bool:
    """
    Aligns the robot to be parallel to a wall on the given side.
    side: 'left'/'right' (legacy) or cardinal 'north'|'east'|'south'|'west'.
    Returns True when alignment is complete.
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    s = str(side).lower()
    mapping = {
        'north': 0,
        'east': 270,
        'south': 180,
        'west': 90,
        'right':270,   # legacy
        'left': 90,   # legacy
    }
    if s not in mapping:
        turtle.get_logger().warn(f'AlignSelf: unknown side "{side}"')
        return False
    RANGER_OFFSET = -4.5 # installation offset
    wall_angle_deg = (mapping[s] + RANGER_OFFSET) % 360
    angle_spread_deg = 20

    if not turtle.mid.get("active") or turtle.mid.get("type") != "align":
        turtle.mid.update({
            "active": True,
            "type": "align",
            "align_side": s,
        })
        turtle.get_logger().info(f'Starting alignment with {s} wall')

    ranges = turtle.valueLaserRaw.get('ranges', [])
    if len(ranges) != 360:
        return False  # Not ready yet

    angle1 = int((wall_angle_deg - angle_spread_deg) % 360)
    angle2 = int((wall_angle_deg + angle_spread_deg) % 360)

    rmax = turtle.valueLaserRaw.get('range_max', float('inf'))
    range1 = ranges[angle1]
    range2 = ranges[angle2]
    if range1 == 0.0 or math.isinf(range1): range1 = rmax
    if range2 == 0.0 or math.isinf(range2): range2 = rmax

    err = range1 - range2

    ALIGN_TOL = 0.005  # 0.5 cm
    if abs(err) <= ALIGN_TOL:
        turtle.publishVelocityCommand(0.0, 0.0)
        turtle.mid["active"] = False
        turtle.get_logger().info(f'Alignment complete to {s} (error: {err:.4f} m)')
        return True

    Kp_align = 4.0
    MIN_ROT = 0.05
    wz = clamp(err/abs(err) * max(abs(Kp_align * err), MIN_ROT), -ANG_MAX, ANG_MAX)
    turtle.publishVelocityCommand(0.0, wz)
    return False

def GoToNextNode(turtle, prefer_turn='left', node_distance_cm=30) -> bool:
    """
    Aligns to a wall, turns if needed, and moves forward to the next grid node.
    This is a stateful action using turtle.mid.
    
    Args:
        turtle: The TurtleBot3 controller instance
        prefer_turn: Preferred turn direction - 'front', 'left', 'right', or 'back' (default: 'left')
        node_distance_cm: Grid spacing in centimeters (default: 30)
    
    Returns True when complete.
    """
    # Initialize GoToNextNode state with a separate namespace
    if not hasattr(turtle, 'gtn_state') or turtle.gtn_state.get("active") != True:
        # Measure distance to front wall
        ranges = turtle.valueLaserRaw['ranges']
        if not ranges or len(ranges) != 360: 
            return False
        rmax = turtle.valueLaserRaw['range_max']
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]

        front_ranges = ranges[0:30] + ranges[330:360]
        front_dist_m = min(front_ranges) if front_ranges else rmax
        front_dist_cm = front_dist_m * 100.0
        
        # Calculate initial offset from grid
        initial_offset_cm = front_dist_cm % node_distance_cm
        
        turtle.get_logger().info(f'Front wall distance: {front_dist_cm:.1f}cm, initial offset: {initial_offset_cm:.1f}cm')

        left_ranges = ranges[85:95]
        right_ranges = ranges[265:275]
        back_ranges = ranges[175:185]

        left_dist = min(left_ranges) if left_ranges else rmax
        right_dist = min(right_ranges) if right_ranges else rmax
        back_dist = min(back_ranges) if back_ranges else rmax

        # Choose side: prefer the closer side wall (left/right).
        # If the back (south) wall is noticeably closer than both side walls, align to south.
        side = 'left' if left_dist < right_dist else 'right'
        SOUTH_MARGIN = 0.1  # require back to be this much closer to prefer south
        if back_dist + SOUTH_MARGIN < min(left_dist, right_dist):
            side = 'south'

        turtle.gtn_state = {
            "active": True,
            "state": "offsetting",
            "align_side": side,
            "prefer_turn": prefer_turn.lower(),
            "node_distance_cm": node_distance_cm,
            "initial_offset_cm": 15,
        }
        # Clear turtle.mid to allow sub-functions to initialize
        turtle.mid.clear()
        turtle.get_logger().info(f'GoToNextNode started: initial offset {initial_offset_cm:.1f}cm, will move one node ({node_distance_cm}cm)')

    state = turtle.gtn_state["state"]
    turtle.get_logger().info(f'GoToNextNode state: {state}')

    if state == "offsetting":
        # Move forward until (front_dist % node_distance) <= initial_offset
        # This means we've reached the next grid node
        ranges = turtle.valueLaserRaw.get('ranges', [])
        if not ranges or len(ranges) != 360:
            return False
            
        rmax = turtle.valueLaserRaw.get('range_max', float('inf'))
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]
        
        front_ranges = ranges[0:1] + ranges[359:360]
        current_front_dist_m = min(front_ranges) if front_ranges else rmax
        current_front_dist_cm = current_front_dist_m * 100.0
        current_offset_cm = current_front_dist_cm % turtle.gtn_state["node_distance_cm"]
        
        initial_offset_cm = turtle.gtn_state["initial_offset_cm"]
        
        # Check if we've reached a node (current offset <= initial offset)
        # Add small tolerance
        OFFSET_TOLERANCE = 0.2  # cm
        if current_offset_cm <= initial_offset_cm + OFFSET_TOLERANCE:
            turtle.publishVelocityCommand(0.0, 0.0)
            turtle.gtn_state["state"] = "aligning"
            turtle.mid.clear()
            turtle.get_logger().info(f'GoToNextNode: reached grid node (offset: {current_offset_cm:.1f}cm), now aligning')
            return False
        
        # Continue moving forward with obstacle avoidance
        # Check for wall safety
        WALL_SAFETY_DISTANCE = 0.15  # 15cm
        if current_front_dist_m < WALL_SAFETY_DISTANCE:
            turtle.publishVelocityCommand(0.0, 0.0)
            turtle.gtn_state["state"] = "aligning"
            turtle.mid.clear()
            turtle.get_logger().warn(f'GoToNextNode: wall too close at {current_front_dist_cm:.1f}cm, stopping offset')
            return False
        
        # Move forward at controlled speed
        forward_speed = 0.1
        turtle.publishVelocityCommand(forward_speed, 0.0)
        return False

    if state == "aligning":
        # Call AlignSelf and check if it completes
        if AlignSelf(turtle, side=turtle.gtn_state["align_side"]):
            turtle.gtn_state["state"] = "deciding"
            turtle.mid.clear()
            turtle.get_logger().info('GoToNextNode: alignment finished, moving to deciding state')
        return False
    
    if state == "deciding":
        surroundings = DetectSurroundings(turtle)
        prefer = turtle.gtn_state.get("prefer_turn", "left")
        
        # Build priority list based on preference
        if prefer == 'front':
            turn_priority = [
                ('north', 'front', 0),      # Front (no turn)
                ('west', 'left', 90),       # Left
                ('east', 'right', -90),     # Right
                ('south', 'back', 180),     # Back
            ]
        elif prefer == 'left':
            turn_priority = [
                ('west', 'left', 90),       # Left
                ('north', 'front', 0),      # Front
                ('east', 'right', -90),     # Right
                ('south', 'back', 180),     # Back
            ]
        elif prefer == 'right':
            turn_priority = [
                ('east', 'right', -90),     # Right
                ('north', 'front', 0),      # Front
                ('west', 'left', 90),       # Left
                ('south', 'back', 180),     # Back
            ]
        elif prefer == 'back':
            turn_priority = [
                ('south', 'back', 180),     # Back
                ('west', 'left', 90),       # Left
                ('east', 'right', -90),     # Right
                ('north', 'front', 0),      # Front
            ]
        else:
            # Default to front if invalid preference
            turn_priority = [
                ('north', 'front', 0),
                ('west', 'left', 90),
                ('east', 'right', -90),
                ('south', 'back', 180),
            ]
        
        # Try each direction in priority order
        turned = False
        for direction, turn_name, angle in turn_priority:
            if not surroundings[direction]:  # This direction is clear
                if angle == 0:  # No turn needed, go straight
                    turtle.gtn_state['state'] = 'moving'
                    turtle.mid.clear()  # Clear for GoTo
                    turtle.get_logger().info(f'GoToNextNode: front is clear, will move forward')
                else:  # Need to turn
                    turtle.gtn_state['state'] = 'turning'
                    turtle.gtn_state['turn_direction'] = turn_name
                    turtle.gtn_state['turn_angle'] = angle
                    turtle.mid.clear()  # Clear for TurnTo
                    turtle.get_logger().info(f'GoToNextNode: turning {turn_name.upper()} ({direction} is clear)')
                turned = True
                break
        
        if not turned:
            # Completely surrounded - this shouldn't happen, but handle it
            turtle.get_logger().error('GoToNextNode: robot is surrounded by walls!')
            turtle.gtn_state['state'] = 'turning'
            turtle.gtn_state['turn_direction'] = 'emergency'
            turtle.gtn_state['turn_angle'] = 90  # Try turning left anyway
            turtle.mid.clear()
        return False

    if state == "turning":
        turn_angle = turtle.gtn_state.get("turn_angle", 90)
        if TurnTo(turtle, turn_angle):
            turtle.gtn_state["state"] = "aligning_after_turn"
            turtle.mid.clear()  # Clear for AlignSelf
            turtle.get_logger().info(f'GoToNextNode: turned {turn_angle} degrees, now aligning')
        return False
    
    if state == "aligning_after_turn":
        # Determine which wall to align to after turning
        ranges = turtle.valueLaserRaw['ranges']
        if not ranges or len(ranges) != 360: 
            return False
        rmax = turtle.valueLaserRaw['range_max']
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]

        left_ranges = ranges[85:95]
        right_ranges = ranges[265:275]
        back_ranges = ranges[175:185]

        left_dist = min(left_ranges) if left_ranges else rmax
        right_dist = min(right_ranges) if right_ranges else rmax
        back_dist = min(back_ranges) if back_ranges else rmax

        # Choose side: prefer the closer side wall (left/right).
        # If the back (south) wall is noticeably closer than both side walls, align to south.
        align_side = 'left' if left_dist < right_dist else 'right'
        SOUTH_MARGIN = 0.1  # require back to be this much closer to prefer south
        if back_dist + SOUTH_MARGIN < min(left_dist, right_dist):
            align_side = 'south'
        
        if AlignSelf(turtle, side=align_side):
            turtle.gtn_state["state"] = "moving"
            turtle.mid.clear()  # Clear for GoTo
            turtle.get_logger().info(f'GoToNextNode: post-turn alignment complete')
        return False
    
    if state == "moving":
        # Move forward by approximately one node distance
        if GoTo(turtle, turtle.gtn_state["node_distance_cm"]):
            # Clean up state when completely done
            turtle.gtn_state = {"active": False}
            turtle.mid.clear()
            turtle.get_logger().info('GoToNextNode: reached next node, complete')
            return True
        return False
            
    return False

def main(args=None):
    rclpy.init(args=args)
    node = Turtlebot3Controller()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        robotStop(node)
        node.get_logger().info('Keyboard interrupt received')
    finally:
        robotStop(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
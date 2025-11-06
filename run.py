# Authors: Nopphakorn Subs. Niwatchai Wang. Supasate Wor. Narith Tha. Tawan Thaep. Napatharak Muan.
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan, BatteryState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import math

# ==================== TUNABLES ====================

# Control gains
Kp_ang = 2.0              # angular P gain (rad/s per rad of error)
Kp_lin = 0.6              # linear P gain (m/s per m of error)
Kp_ang_turn = 2.0         # angular P gain for TurnTo (rad/s per rad)
Kp_ang_track = 2.5        # angular P gain for GoTo heading correction (rad/s per rad)
Kp_align = 4.0            # angular P gain for AlignSelf

# Velocity limits
LIN_MAX = 0.25            # max linear velocity (m/s)
ANG_MAX = 1.2             # max angular velocity (rad/s)
ANG_MAX_TRACK = 1.2       # max angular velocity for heading correction
MIN_LINEAR_VEL = 0.08     # minimum linear velocity to maintain motion (m/s)
MIN_ANGULAR_VEL = 0.05    # minimum angular velocity for alignment (rad/s)

# Tolerances
YAW_TOL_DEG = 0.5         # stop turning within this tolerance (degrees)
POS_TOL_M = 0.05          # stop at goal if closer than this (m)
DEAD_DEG = 0.005          # no turning when nearly aligned (degrees)
OFF_DEG_SLOW = 20.0       # slow forward if off-heading more than this (degrees)
ALIGN_TOL_M = 0.005       # alignment tolerance (m)

# Wall detection
WALL_THRESHOLD = 0.18     # wall detection distance (m)
WALL_SAFETY_DISTANCE = 0.15  # safety margin before stopping (m)
WALL_SLOWDOWN_DISTANCE = 0.25  # start slowing down at this distance (m)

# Grid alignment
OFFSET_TOLERANCE_CM = 0.2  # tolerance for grid offset detection (cm)
DEFAULT_NODE_DISTANCE_CM = 30  # default grid spacing (cm)

# Sensor ranges
RANGER_OFFSET_DEG = -4.5  # LiDAR installation offset (degrees)
FRONT_SECTOR_DEG = 30     # front detection sector half-width (degrees)
SIDE_SECTOR_HALFWIDTH_DEG = 5  # side wall detection sector half-width (degrees)
BACK_SECTOR_HALFWIDTH_DEG = 5  # back wall detection sector half-width (degrees)
SOUTH_ALIGN_MARGIN = 0.1  # prefer south alignment if back wall is this much closer (m)

# Control timing
DT = 0.1                  # control period (s)
STARTUP_CYCLES = 3        # number of cycles for startup nudge

# ==================== UTILITY FUNCTIONS ====================

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

# ==================== TURTLEBOT3 CONTROLLER ====================

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

        # Shared state for motion primitives
        self.mid = {}
        self.pref_direction = ""

        # Timer for control loop
        self.timer = self.create_timer(DT, self.timerCallback)
        
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
            
        dir_map = {
            "w" : "front",
            "a" : "left",
            "s" : "back",
            "d" : "right"
        }
        if self.state == 0:
            self.pref_direction = dir_map[input("WASD:")]
            self.state = 1
        elif self.state == 1:
            if GoToNextNode(self, prefer_turn=self.pref_direction):
                self.state = 0
                self.get_logger().info("Continue")

# ==================== MOTION PRIMITIVES ====================

def robotStop(node: Turtlebot3Controller):
    """Stop the robot."""
    node.publishVelocityCommand(0.0, 0.0)

def DetectSurroundings(turtle, wall_threshold=WALL_THRESHOLD):
    """
    Detects if there is a wall in the North, South, East, and West directions
    relative to the robot's current orientation.
    
    Returns:
        dict: {'north': bool, 'east': bool, 'south': bool, 'west': bool}
              True = wall detected, False = clear
    """
    if not turtle.valueLaserRaw['ranges']:
        return {'north': False, 'east': False, 'south': False, 'west': False}

    ranges = list(turtle.valueLaserRaw['ranges'])
    rmax = turtle.valueLaserRaw['range_max']
    
    # Replace 0.0 (invalid scan) with max range
    ranges = [rmax if r == 0.0 else r for r in ranges]
    
    # North: front sector (±2 degrees around 0)
    north_ranges = ranges[0:2] + ranges[358:360]
    min_north_dist = min(north_ranges) if north_ranges else rmax

    # East: right sector (270 degrees ±2)
    east_ranges = ranges[268:272]
    min_east_dist = min(east_ranges) if east_ranges else rmax

    # South: back sector (180 degrees ±2)
    south_ranges = ranges[178:182]
    min_south_dist = min(south_ranges) if south_ranges else rmax

    # West: left sector (90 degrees ±2)
    west_ranges = ranges[88:92]
    min_west_dist = min(west_ranges) if west_ranges else rmax

    surroundings = {
        'north': min_north_dist < wall_threshold,
        'east': min_east_dist < wall_threshold,
        'south': min_south_dist < wall_threshold,
        'west': min_west_dist < wall_threshold,
    }
    
    return surroundings

def TurnTo(turtle, rel_angle_deg) -> bool:
    """
    Relative turn: rotate by 'rel_angle_deg' CCW from the pose at first call.
    
    Args:
        turtle: TurtleBot3Controller instance
        rel_angle_deg: Angle to turn (degrees, positive = CCW)
    
    Returns:
        bool: True when turn is complete
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    # Initialize on first call
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
    Includes wall collision avoidance.
    
    Args:
        turtle: TurtleBot3Controller instance
        rel_distance_cm: Distance to move forward (cm)
    
    Returns:
        bool: True when movement is complete
    """
    if not turtle.have_odom:
        turtle.publishVelocityCommand(0.0, 0.0)
        return False

    # Initialize on first call
    if not turtle.mid.get("active") or turtle.mid.get("type") != "goto":
        turtle.mid.update({
            "active": True,
            "type": "goto",
            "ref_x": turtle.pos_x,
            "ref_y": turtle.pos_y,
            "ref_yaw": turtle.yaw,
            "target": rel_distance_cm / 100.0,  # Convert cm to meters
            "startup_counter": 0
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
    if turtle.mid["startup_counter"] <= STARTUP_CYCLES:
        turtle.publishVelocityCommand(0.15, 0.0)
        return False

    # Calculate velocity with proportional control
    vx = clamp(Kp_lin * remaining, MIN_LINEAR_VEL, LIN_MAX)
    
    # For very short distances, use a higher minimum
    if remaining < 0.3:
        vx = max(vx, 0.1)

    # Check for obstacles in front
    ranges = turtle.valueLaserRaw.get('ranges', [])
    if ranges and len(ranges) == 360:
        rmax = turtle.valueLaserRaw.get('range_max', float('inf'))
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]
        
        front_ranges = ranges[0:FRONT_SECTOR_DEG] + ranges[360-FRONT_SECTOR_DEG:360]
        min_front_dist = min(front_ranges) if front_ranges else rmax
        
        # Stop if wall is too close
        if min_front_dist < WALL_SAFETY_DISTANCE:
            turtle.publishVelocityCommand(0.0, 0.0)
            turtle.mid["active"] = False
            turtle.get_logger().warn(f'GoTo stopped early - wall at {min_front_dist*100:.1f}cm')
            return True
        
        # Slow down if approaching wall
        if min_front_dist < WALL_SLOWDOWN_DISTANCE:
            slowdown_factor = (min_front_dist - WALL_SAFETY_DISTANCE) / (WALL_SLOWDOWN_DISTANCE - WALL_SAFETY_DISTANCE)
            vx = vx * slowdown_factor
            vx = max(vx, MIN_LINEAR_VEL * 0.5)

    # Heading correction to maintain straight line
    yaw_err = normalize_angle_rad(turtle.mid["ref_yaw"] - turtle.yaw)
    err_deg = abs(rad2deg(yaw_err))

    if err_deg <= DEAD_DEG:
        wz = 0.0
    else:
        wz = clamp(Kp_ang_track * yaw_err, -ANG_MAX_TRACK, ANG_MAX_TRACK)
        if err_deg > OFF_DEG_SLOW:
            vx = max(vx * 0.5, MIN_LINEAR_VEL)

    turtle.publishVelocityCommand(vx, wz)
    return False

def AlignSelf(turtle, side='left') -> bool:
    """
    Aligns the robot to be parallel to a wall on the given side.
    
    Args:
        turtle: TurtleBot3Controller instance
        side: 'left'/'right' or cardinal 'north'/'east'/'south'/'west'
    
    Returns:
        bool: True when alignment is complete
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
        'right': 270,
        'left': 90,
    }
    
    if s not in mapping:
        turtle.get_logger().warn(f'AlignSelf: unknown side "{side}"')
        return False
    
    wall_angle_deg = (mapping[s] + RANGER_OFFSET_DEG) % 360
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
        return False

    angle1 = int((wall_angle_deg - angle_spread_deg) % 360)
    angle2 = int((wall_angle_deg + angle_spread_deg) % 360)

    rmax = turtle.valueLaserRaw.get('range_max', float('inf'))
    range1 = ranges[angle1] if ranges[angle1] > 0 and not math.isinf(ranges[angle1]) else rmax
    range2 = ranges[angle2] if ranges[angle2] > 0 and not math.isinf(ranges[angle2]) else rmax

    err = range1 - range2

    if abs(err) <= ALIGN_TOL_M:
        turtle.publishVelocityCommand(0.0, 0.0)
        turtle.mid["active"] = False
        turtle.get_logger().info(f'Alignment complete to {s} (error: {err:.4f} m)')
        return True

    wz = clamp(err/abs(err) * max(abs(Kp_align * err), MIN_ANGULAR_VEL), -ANG_MAX, ANG_MAX)
    turtle.publishVelocityCommand(0.0, wz)
    return False

# ==================== HIGH-LEVEL BEHAVIORS ====================

def GoToNextNode(turtle, prefer_turn='left', node_distance_cm=DEFAULT_NODE_DISTANCE_CM) -> bool:
    """
    Aligns to a wall, turns if needed, and moves forward to the next grid node.
    Uses a state machine to orchestrate: offsetting -> aligning -> deciding -> 
    turning -> aligning_after_turn -> moving.
    
    Args:
        turtle: TurtleBot3Controller instance
        prefer_turn: Preferred turn direction - 'front', 'left', 'right', or 'back'
        node_distance_cm: Grid spacing in centimeters
    
    Returns:
        bool: True when complete
    """
    # Initialize state
    if not hasattr(turtle, 'gtn_state') or turtle.gtn_state.get("active") != True:
        ranges = turtle.valueLaserRaw['ranges']
        if not ranges or len(ranges) != 360: 
            return False
        
        rmax = turtle.valueLaserRaw['range_max']
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]

        turtle.gtn_state = {
            "active": True,
            "state": "aligning", # TODO: CHANGE THIS TO offsetting
            "prefer_turn": prefer_turn.lower(),
            "node_distance_cm": node_distance_cm,
            "initial_offset_cm": 18,  # Fixed offset for grid alignment
        }
        
        turtle.mid.clear()
        turtle.get_logger().info("GoToNextNode started")

    state = turtle.gtn_state["state"]
    turtle.get_logger().info(f'GoToNextNode state: {state}')

    # State: Offsetting - move to grid node
    if state == "offsetting":
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
        
        # Check if reached grid node
        if current_offset_cm <= initial_offset_cm + OFFSET_TOLERANCE_CM:
            turtle.publishVelocityCommand(0.0, 0.0)
            turtle.gtn_state["state"] = "aligning"
            turtle.mid.clear()
            turtle.get_logger().info(f'Reached grid node (offset: {current_offset_cm:.1f}cm)')
            return False
        
        # Safety check
        if current_front_dist_m < WALL_SAFETY_DISTANCE:
            turtle.publishVelocityCommand(0.0, 0.0)
            turtle.gtn_state["state"] = "aligning"
            turtle.mid.clear()
            turtle.get_logger().warn(f'Wall too close at {current_front_dist_cm:.1f}cm')
            return False
        
        turtle.publishVelocityCommand(0.1, 0.0)
        return False

    # State: Aligning - align to wall
    if state == "aligning":
        ranges = turtle.valueLaserRaw['ranges']
        if not ranges or len(ranges) != 360: 
            return False
        
        rmax = turtle.valueLaserRaw['range_max']
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]

        left_ranges = ranges[90-SIDE_SECTOR_HALFWIDTH_DEG:90+SIDE_SECTOR_HALFWIDTH_DEG]
        right_ranges = ranges[270-SIDE_SECTOR_HALFWIDTH_DEG:270+SIDE_SECTOR_HALFWIDTH_DEG]
        back_ranges = ranges[180-BACK_SECTOR_HALFWIDTH_DEG:180+BACK_SECTOR_HALFWIDTH_DEG]
        front_ranges = ranges[0:BACK_SECTOR_HALFWIDTH_DEG] + ranges[360-BACK_SECTOR_HALFWIDTH_DEG:360]

        left_dist = min(left_ranges) if left_ranges else rmax
        right_dist = min(right_ranges) if right_ranges else rmax
        back_dist = min(back_ranges) if back_ranges else rmax
        front_dist = min(front_ranges) if front_ranges else rmax

        align_side = 'left' if left_dist < right_dist else 'right'
        if back_dist + SOUTH_ALIGN_MARGIN < min([left_dist, right_dist, front_dist]):
            align_side = 'south'
        elif front_dist + SOUTH_ALIGN_MARGIN < min(left_dist, right_dist):
            align_side = 'north'
    
        if AlignSelf(turtle, side=align_side):
            turtle.gtn_state["state"] = "deciding"
            turtle.mid.clear()
            turtle.get_logger().info('Alignment finished, deciding direction')
        return False
    
    # State: Deciding - choose direction
    if state == "deciding":
        surroundings = DetectSurroundings(turtle)
        prefer = turtle.gtn_state.get("prefer_turn", "left")
        
        # Build priority list
        priority_map = {
            'front': [('north', 'front', 0), ('west', 'left', 90), ('east', 'right', -90), ('south', 'back', 180)],
            'left': [('west', 'left', 90), ('north', 'front', 0), ('east', 'right', -90), ('south', 'back', 180)],
            'right': [('east', 'right', -90), ('north', 'front', 0), ('west', 'left', 90), ('south', 'back', 180)],
            'back': [('south', 'back', 180), ('west', 'left', 90), ('east', 'right', -90), ('north', 'front', 0)],
        }
        turn_priority = priority_map.get(prefer, priority_map['front'])
        
        # Try each direction
        turned = False
        for direction, turn_name, angle in turn_priority:
            if not surroundings[direction]:
                if angle == 0:
                    turtle.gtn_state['state'] = 'moving'
                    turtle.get_logger().info('Front clear, moving forward')
                else:
                    turtle.gtn_state['state'] = 'turning'
                    turtle.gtn_state['turn_direction'] = turn_name
                    turtle.gtn_state['turn_angle'] = angle
                    turtle.get_logger().info(f'Turning {turn_name.upper()} ({direction} clear)')
                turtle.mid.clear()
                turned = True
                break
        
        if not turned:
            turtle.get_logger().error('Robot surrounded by walls!')
            turtle.gtn_state['state'] = 'turning'
            turtle.gtn_state['turn_angle'] = 90
            turtle.mid.clear()
        return False

    # State: Turning
    if state == "turning":
        turn_angle = turtle.gtn_state.get("turn_angle", 90)
        if TurnTo(turtle, turn_angle):
            turtle.gtn_state["state"] = "aligning_after_turn"
            turtle.mid.clear()
            turtle.get_logger().info(f'Turned {turn_angle} degrees')
        return False
    
    # State: Aligning after turn
    if state == "aligning_after_turn":
        ranges = turtle.valueLaserRaw['ranges']
        if not ranges or len(ranges) != 360: 
            return False
        
        rmax = turtle.valueLaserRaw['range_max']
        ranges = [rmax if r == 0.0 or math.isinf(r) else r for r in ranges]

        left_ranges = ranges[90-SIDE_SECTOR_HALFWIDTH_DEG:90+SIDE_SECTOR_HALFWIDTH_DEG]
        right_ranges = ranges[270-SIDE_SECTOR_HALFWIDTH_DEG:270+SIDE_SECTOR_HALFWIDTH_DEG]
        back_ranges = ranges[180-BACK_SECTOR_HALFWIDTH_DEG:180+BACK_SECTOR_HALFWIDTH_DEG]
        front_ranges = ranges[0:BACK_SECTOR_HALFWIDTH_DEG] + ranges[360-BACK_SECTOR_HALFWIDTH_DEG:360]

        left_dist = min(left_ranges) if left_ranges else rmax
        right_dist = min(right_ranges) if right_ranges else rmax
        back_dist = min(back_ranges) if back_ranges else rmax
        front_dist = min(front_ranges) if front_ranges else rmax

        align_side = 'left' if left_dist < right_dist else 'right'
        if back_dist + SOUTH_ALIGN_MARGIN < min([left_dist, right_dist, front_dist]):
            align_side = 'south'
        elif front_dist + SOUTH_ALIGN_MARGIN < min(left_dist, right_dist):
            align_side = 'north'
        
        if AlignSelf(turtle, side=align_side):
            turtle.gtn_state["state"] = "moving"
            turtle.mid.clear()
            turtle.get_logger().info('Post-turn alignment complete')
        return False
    
    # State: Moving - move forward one node
    if state == "moving":
        if GoTo(turtle, turtle.gtn_state["node_distance_cm"]):
            turtle.gtn_state = {"active": False}
            turtle.mid.clear()
            turtle.get_logger().info('Reached next node, complete')
            return True
        return False
            
    return False

# ==================== MAIN ====================

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
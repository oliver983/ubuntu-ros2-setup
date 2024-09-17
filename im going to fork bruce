import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import math


class navigationControl(Node):
    def __init__(self):
        super().__init__('exploration')
        
        # Subscriptions for map, odometry, and laser scan topics
        self.map_subscription = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)  # Nav2 goal topic
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize variables to store map, odometry, and scan data
        self.map_data = None
        self.resolution = None
        self.originX = None
        self.originY = None
        self.width = None
        self.height = None
        self.data = None
        self.grid = None
        
        # List to track visited frontiers
        self.visited_points = []

    def scan_callback(self, msg):
        # Store laser scan data
        self.scan_data = msg
        self.scan = msg.ranges

    def map_callback(self, msg):
        # Store map data
        self.map_data = msg
        self.resolution = msg.info.resolution
        self.originX = msg.info.origin.position.x
        self.originY = msg.info.origin.position.y
        self.width = msg.info.width
        self.height = msg.info.height
        self.data = msg.data
        
        # Format the map into a 2D grid
        self.map_formatting()
        
        # Process the map for frontier detection
        if self.map_data is not None:
            frontiers = self.get_frontiers(self.map_data)
            if frontiers:
                
                chosen_frontier = self.choose_frontier(frontiers)
                if chosen_frontier:
                    point = self.decide_point(chosen_frontier)
                    self.get_logger().info(f'Navigating to point: {point}')
                    self.navigate_to_point(point)

    def odom_callback(self, msg):
        # Store odometry data
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

    def map_formatting(self):
        # Convert flat map data to a 2D grid
        map_data = self.data
        gridcounter = 0
        xl = self.width
        yl = self.height
        self.grid = [[0 for _ in range(yl)] for _ in range(xl)]
        for x in range(xl):
            for y in range(yl):
                self.grid[x][y] = map_data[gridcounter]
                gridcounter += 1

    def get_frontiers(self, map_data):
        # Converts map data to a NumPy array
        map_array = np.array(map_data.data).reshape((map_data.info.height, map_data.info.width))
        # Call the find_frontiers method to detect all frontiers
        frontiers = self.find_frontiers(map_array)
        
        # Return all the detected frontiers
        return frontiers

    def find_frontiers(self, map_array):
        frontiers = []
        height, width = map_array.shape

        for y in range(height):
            for x in range(width):
                # Check if the current cell is free space (0 represents free space)
                if map_array[y, x] == 0:  
                    # Get neighbors of the current cell
                    neighbors = self.get_neighbors(x, y, width, height)
                    
                    # Check if any of the neighboring cells are unknown (-1 represents unknown)
                    if any(map_array[ny, nx] == -1 for nx, ny in neighbors):  
                        # Add this cell as part of the frontier
                        frontier = (x, y)
                        
                        # Only add if the point is not already in the frontiers list
                        if frontier not in frontiers:
                            frontiers.append(frontier)
        
        return frontiers  # Return all frontier points

    def choose_frontier(self, frontiers):
        # Select a frontier that hasn't been visited yet
        for frontier in frontiers:
            if frontier not in self.visited_points:
                return frontier  # Return the first unvisited frontier
        
        return None  # If all frontiers are visited, return None

    def decide_point(self, frontier):
        # Pick a point in the frontier (you could also use the middle of the frontier as before)
        return frontier

    def get_neighbors(self, x, y, width, height):
        neighbors = []
        
        # Check all the neighboring cells in an 8-connected grid (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the cell itself (dx == 0 and dy == 0)
                if dx == 0 and dy == 0:
                    continue
                
                # Calculate the neighbor's coordinates
                nx, ny = x + dx, y + dy
                
                # Ensure the neighbor is within the map bounds
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))
        
        return neighbors

    def navigate_to_point(self, point):
        # Check if the point has already been visited
        if point in self.visited_points:
            self.get_logger().info(f"Point {point} already visited, skipping.")
            return
        
        # Convert grid coordinates to world coordinates using map resolution and origin
        goal_x = self.originX + point[0] * self.resolution
        goal_y = self.originY + point[1] * self.resolution

        # Create a PoseStamped message for the goal
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"  # Ensure the goal is in the "map" frame
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.orientation.w = 1.0  # Facing forward, assuming no rotation for simplicity

        # Publish the goal to Nav2
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"Published goal: x = {goal_x}, y = {goal_y}")
        
        # Add the point to the visited list
        self.visited_points.append(point)

    def euler_from_quaternion(self, x, y, z, w):
        """Convert a quaternion into euler angles (roll, pitch, yaw)"""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians


def main(args=None):
    # Initialize ROS 2 communication
    rclpy.init(args=args)
    
    # Create an instance of your navigationControl node
    navigation_control = navigationControl()
    
    # Keep the node alive and listen to incoming topics
    try:
        rclpy.spin(navigation_control)
    except KeyboardInterrupt:
        pass

    # Shutdown the node gracefully when finished
    navigation_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

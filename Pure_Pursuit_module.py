import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from queue import Queue

class MessageQueueManager:
    def __init__(self):
        self.queueList = {
            "Critical": Queue(),
            "Warning": Queue(),
            "General": Queue(),
            "Config": Queue(),
        }

    def add_message(self, queue_name, message):
        """
        Add a message to the specified queue.
        
        Parameters:
            queue_name (str): Name of the queue to add the message to.
            message (str): The message to be added.
        """
        if queue_name in self.queueList:
            self.queueList[queue_name].put(message)
        else:
            print(f"Queue '{queue_name}' does not exist.")

    def get_message(self, queue_name):
        """
        Get a message from the specified queue.
        
        Parameters:
            queue_name (str): Name of the queue to get the message from.
        
        Returns:
            str: The message retrieved from the queue.
        """
        if queue_name in self.queueList:
            if not self.queueList[queue_name].empty():
                return self.queueList[queue_name].get()
            else:
                print(f"Queue '{queue_name}' is empty.")
                return None
        else:
            print(f"Queue '{queue_name}' does not exist.")
            return None

class PurePursuit:
    def __init__(self, graph_file_path, start_node, goal_node):
        self.graph = self.parse_graphml(graph_file_path)
        self.start_node = start_node
        self.goal_node = goal_node
        self.planned_path = nx.shortest_path(self.graph, source=start_node, target=goal_node)
        self.planned_path_positions = np.array([self.graph.nodes[node]['pos'] for node in self.planned_path])
        self.planned_path_graph = self.graph.subgraph(self.planned_path)
        self.smoothed_path = self.smoothing(self.planned_path_positions, 0.1, 0.3, 0.001)
        
        self.currentPos = self.planned_path_positions[0]
        self.currentHeading = 0
        self.lastFoundIndex = 0
        self.lookAheadDis = 0.8
        self.using_rotation = False
        self.numOfFrames = 400

        self.fig = plt.figure()
        self.trajectory_lines = plt.plot([], '-', color='orange', linewidth=4)
        self.trajectory_line = self.trajectory_lines[0]
        self.poses = plt.plot([], 'o', color='black', markersize=10)
        self.pose = self.poses[0]
        
        plt.plot(self.planned_path_positions[:, 0], self.planned_path_positions[:, 1], '--', color='grey')
        plt.axis("scaled")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        self.dt = 100
        self.xs = [self.planned_path_positions[0, 0]]
        self.ys = [self.planned_path_positions[0, 1]]

        self.anim = animation.FuncAnimation(self.fig, self.pure_pursuit_animation, frames=self.numOfFrames, interval=50)

    def parse_graphml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        graph = nx.DiGraph()

        # Add nodes to the graph with X and Y coordinates
        for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
            node_id = node.get('id')

            # Check if X and Y attributes are present
            x_elem = node.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="d0"]')
            y_elem = node.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="d1"]')

            if x_elem is not None and y_elem is not None:
                x = float(x_elem.text)
                y = float(y_elem.text)
                graph.add_node(node_id, pos2=[x, y])
                y = -y
                graph.add_node(node_id, pos=[x, y])

        # Add edges to the graph with True/False indicating the ability to change lanes
        for edge in root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge'):
            source = edge.get('source')
            target = edge.get('target')
            data_value = edge.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="d2"]').text

            graph.add_edge(source, target, change_lane=data_value)

        return graph

    def visualize_map(self, title=''):
        pos = nx.get_node_attributes(self.graph, 'pos')

        if not pos:
            print("No nodes with valid positions found.")
            return

        # Draw nodes with customized settings
        nx.draw_networkx_nodes(self.graph, pos, node_size=300, node_color='skyblue', edgecolors='black')

        # Draw edges with customized settings
        for edge, change_lane in nx.get_edge_attributes(self.graph, 'change_lane').items():
            edge_color = 'blue' if change_lane else 'red'
            nx.draw_networkx_edges(self.graph, pos, edgelist=[edge], edge_color=edge_color, width=2)

        # Draw labels with coordinates, node ID, and the ability to change lanes
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_color='black')

        plt.title(title)
        plt.axis('off')  # Turn off the axis
        plt.show()

    def smoothing(self, path, weight_data, weight_smooth, tolerance):
        smoothed_path = path.copy()
        change = tolerance

        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(0, len(path[i])):
                    aux = smoothed_path[i][j]
                    smoothed_path[i][j] += weight_data * (path[i][j] - smoothed_path[i][j]) + weight_smooth * (
                                smoothed_path[i - 1][j] + smoothed_path[i + 1][j] - (2.0 * smoothed_path[i][j]))
                    change += np.abs(aux - smoothed_path[i][j])
        return smoothed_path

    def pt_to_pt_distance(self, pt1, pt2):
        distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        return distance

    def sgn(self, num):
        if num >= 0:
            return 1
        else:
            return -1

    def pure_pursuit_step(self, path, currentPos, currentHeading, lookAheadDis, LFindex):
        currentX = currentPos[0]
        currentY = currentPos[1]

        lastFoundIndex = LFindex
        foundIntersection = False
        startingIndex = lastFoundIndex

        for i in range(startingIndex, len(path) - 1):
            x1 = path[i][0] - currentX
            y1 = path[i][1] - currentY
            x2 = path[i + 1][0] - currentX
            y2 = path[i + 1][1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx ** 2 + dy ** 2)
            D = x1 * y2 - x2 * y1
            discriminant = (lookAheadDis ** 2) * (dr ** 2) - D ** 2

            if discriminant >= 0:
                sol_x1 = (D * dy + self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_x2 = (D * dy - self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
                sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2
                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

                minX = min(path[i][0], path[i + 1][0])
                minY = min(path[i][1], path[i + 1][1])
                maxX = max(path[i][0], path[i + 1][0])
                maxY = max(path[i][1], path[i + 1][1])

                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                    linearVel = 40  # percentage
                    foundIntersection = True

                    if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and (
                            (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                        if self.pt_to_pt_distance(sol_pt1, path[i + 1]) < self.pt_to_pt_distance(sol_pt2, path[i + 1]):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2
                    else:
                        if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    if self.pt_to_pt_distance(goalPt, path[i + 1]) < self.pt_to_pt_distance([currentX, currentY], path[i + 1]):
                        lastFoundIndex = i
                        break
                    else:
                        steeringAngle = 0
                        linearVel = 0
                        lastFoundIndex = i + 1

                else:
                    linearVel = 0  # percentage
                    steeringAngle = 0
                    foundIntersection = False
                    goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

            else:
                linearVel = 0  # percentage
                foundIntersection = False
                goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

        if foundIntersection:
            absTargetAngle = math.atan2(goalPt[1] - currentPos[1], goalPt[0] - currentPos[0]) * 180 / np.pi
            if absTargetAngle < 0:
                absTargetAngle += 360
            turnError = absTargetAngle - currentHeading

            if turnError > 180 or turnError < -180:
                turnError = -1 * self.sgn(turnError) * (360 - abs(turnError))

            if turnError > 20:
                steeringAngle = 20
                linearVel = 10
            elif turnError < -20:
                steeringAngle = -20
                linearVel = 10
            else:
                steeringAngle = turnError

            return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection

        else:
            turnError = 0
            linearVel = 0
            steeringAngle = 0
            goalPt = [currentPos[0], currentPos[1]]
            lastFoundIndex = lastFoundIndex

            return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection

    def pure_pursuit_animation(self, frame):
        goalPt, self.lastFoundIndex, SteeringAngle, turnError, linearVel, foundIntersection = self.pure_pursuit_step(self.smoothed_path, self.currentPos, self.currentHeading, self.lookAheadDis, self.lastFoundIndex)

        if not foundIntersection or np.all(self.currentPos == self.planned_path_positions[-1]):
            return  # Stop the animation 

        maxLinVelfeet = (200 / 60 * np.pi * 4 / 12) * 0.47  # in feet per second (scaled to cm/s)
        maxTurnVelDeg = (200 / 60 * np.pi * 4 / 9 * 180 / np.pi) * 0.18124573  # in degrees per second

        TurnVel = (SteeringAngle / maxTurnVelDeg) * maxTurnVelDeg
        stepDis = linearVel / 100 * maxLinVelfeet * self.dt / 1000
        self.currentPos[0] += stepDis * np.cos(self.currentHeading * np.pi / 180)
        self.currentPos[1] += stepDis * np.sin(self.currentHeading * np.pi / 180)

        self.currentHeading += TurnVel * (self.dt / 1000)
        if not self.using_rotation:
            self.currentHeading = self.currentHeading % 360
            if self.currentHeading < 0:
                self.currentHeading += 360

        self.xs.append(self.currentPos[0])
        self.ys.append(self.currentPos[1])

        self.pose.set_data([self.currentPos[0]], [self.currentPos[1]])
        self.trajectory_line.set_data(self.xs, self.ys)
        R = (self.lookAheadDis*30.48/2)/ math.sin(np.radians(turnError))
        # Update car orientation
        pose_angle = self.currentHeading
        self.pose.set_marker((3, 0, pose_angle))
        print(f"Current Position: ({self.currentPos[0]}, {self.currentPos[1]})")
        print(f"Turn Error: {turnError} degrees")
        print(f"Steering Angle: {SteeringAngle} degrees")
        print(f"Velocity: {((linearVel/100) * maxLinVelfeet)*30.48} cm/s\n")
        print(f"Radius {R}cm")
        return self.trajectory_line, self.pose

# Define the file path to your GraphML file
graph_file_path = 'Competition_track_graph.graphml'
start_node = "409"
goal_node = "408"

car_animation = PurePursuit(graph_file_path, start_node, goal_node)
plt.show()

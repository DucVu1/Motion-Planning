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
# -------------------Map generating-------------------
def parse_graphml(file_path):
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
            y=-y

            graph.add_node(node_id, pos=[x, y])

    # Add edges to the graph with True/False indicating the ability to change lanes
    for edge in root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge'):
        source = edge.get('source')
        target = edge.get('target')
        data_value = edge.find('.//{http://graphml.graphdrawing.org/xmlns}data[@key="d2"]').text

        graph.add_edge(source, target, change_lane=data_value)

    return graph


def visualize_map(graph, title=''):
    pos = nx.get_node_attributes(graph, 'pos')

    if not pos:
        print("No nodes with valid positions found.")
        return

    # Draw nodes with customized settings
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='skyblue', edgecolors='black')

    # Draw edges with customized settings
    for edge, change_lane in nx.get_edge_attributes(graph, 'change_lane').items():
        edge_color = 'blue' if change_lane else 'red'
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color=edge_color, width=2)

    # Draw labels with coordinates, node ID, and the ability to change lanes
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black')

    plt.title(title)
    plt.axis('off')  # Turn off the axis
    plt.show()

# Replace 'Competition_track_graph.graphml' with the actual path to your GraphML file
graph = parse_graphml('Competition_track_graph.graphml')

# Example: Plan a path from node '472' to '23'
start_node = "472"
goal_node = "82"
planned_path = nx.shortest_path(graph, source=start_node, target=goal_node)
planned_path_positions = np.array([graph.nodes[node]['pos'] for node in planned_path])

# Create a graph for the planned path
planned_path_graph = graph.subgraph(planned_path)
# Visualize the original map
visualize_map(graph, title='Original Map')
# Visualize the planned path in a separate graph
visualize_map(planned_path_graph, title='Planned Path')


# -------------------Pure Pursuit Algorithm-------------------
# def add_line(path):
#     for i in range(0, len(path)):
#         plt.plot(path[i][0], path[i][1], '.', color='red', markersize=10)

#     for i in range(0, len(path) - 1):
#         plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='b')

#     plt.axis('scaled')

# def add_complicated_line(path, lineStyle, lineColor, lineLabel):
#     for i in range(0, len(path)):
#         plt.plot(path[i][0], path[i][1], '.', color='red', markersize=10)

#     for i in range(0, len(path) - 1):
#         if (i == 0):
#             plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], lineStyle, color=lineColor,
#                      label=lineLabel)
#         else:
#             plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], lineStyle, color=lineColor)

#     plt.axis('scaled')

# def highlight_points(points, pointColor):
#     for point in points:
#         plt.plot(point[0], point[1], '.', color=pointColor, markersize=10)

# def draw_circle(x, y, r, circleColor):
#     xs = []
#     ys = []
#     angles = np.arange(0, 2.2 * np.pi, 0.2)

#     for angle in angles:
#         xs.append(r * np.cos(angle) + x)
#         ys.append(r * np.sin(angle) + y)

#     plt.plot(xs, ys, '-', color=circleColor)

currentPos = planned_path_positions[0]
currentHeading = 330
lastFoundIndex = 0
lookAheadDis = 0.8
MaxSteering = 20 #degree
using_rotation = False
numOfFrames = 400
def smoothing (path,weight_data,weight_smooth,tolerance):
    
    smoothed_path = path.copy()
    change = tolerance
    
    while change >= tolerance :
        change = 0.0

        for i in range (1,len(path)-1):
            
            for j in range (0,len(path[i])):
                aux = smoothed_path[i][j]

                smoothed_path[i][j] += weight_data * (path[i][j] - smoothed_path[i][j]) + weight_smooth * (smoothed_path[i-1][j] + smoothed_path[i+1][j] - (2.0 * smoothed_path[i][j]))
                change += np.abs(aux - smoothed_path[i][j])
                
    return smoothed_path
def pt_to_pt_distance(pt1, pt2):
    distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    return distance

def sgn(num):
    if num >= 0:
        return 1
    else:
        return -1

def pure_pursuit_step(path, currentPos, currentHeading, lookAheadDis, LFindex):
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
            sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
            sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
            sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
            sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2
            sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
            sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

            minX = min(path[i][0], path[i + 1][0]) # the min value from the current node to the next node
            minY = min(path[i][1], path[i + 1][1])
            maxX = max(path[i][0], path[i + 1][0])
            maxY = max(path[i][1], path[i + 1][1]) # the max value from the current node to the next node
            # condition for at least one suitable solution, x and y in range of max and min value
            if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                    (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                linearVel = 40 #percentage, this means that this are 10% of the maximum velocity
                foundIntersection = True
                # if both of the two solutions are true then pick the closer one
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

                    if pt_to_pt_distance(sol_pt1, path[i + 1]) < pt_to_pt_distance(sol_pt2, path[i + 1]):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2
                # if only one of the solution is true then pick the appropriate one
                else:
                    if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2
                # check condition to update for the lastFoundIndex
                if pt_to_pt_distance(goalPt, path[i + 1]) < pt_to_pt_distance([currentX, currentY], path[i + 1]):
                    lastFoundIndex = i
                    break
                else:
                    steeringAngle = 0
                    linearVel = 0
                    lastFoundIndex = i + 1

            else:
                linearVel = 0 #percentage, this means that this are 10% of the maximum velocity
                steeringAngle = 0
                foundIntersection = False
                goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

        else:
            linearVel = 0 #percentage, this means that this are 10% of the maximum velocity
            foundIntersection = False
            goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]
    if foundIntersection:
        #absTargetAngle is the angle of the goal point compare with the x axis in degree
        absTargetAngle = math.atan2(goalPt[1] - currentPos[1], goalPt[0] - currentPos[0]) * 180 / pi #atan2 return value in pi 
        if absTargetAngle < 0: absTargetAngle += 360 #atan2 = [-pi,pi] -> absTargetAngle =[-180,180]
        # turnError is the value of the angle the car need to turn
        turnError = absTargetAngle - currentHeading
        # find the minimum turn error
        if turnError > 180 or turnError < -180:
            turnError = -1 * sgn(turnError) * (360 - abs(turnError))   
        if turnError >20:
            steeringAngle =20
            linearVel =10
        elif turnError <-20:
            steeringAngle =-20
            linearVel =10
        else:
            steeringAngle = turnError
        return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection
    else:
        # Handle the case when no intersection is found
        turnError = 0  # Or any other appropriate default value
        linearVel = 0   # Or any other appropriate default value
        steeringAngle = 0  # Or any other appropriate default value
        goalPt = [currentPos[0], currentPos[1]]
        lastFoundIndex = lastFoundIndex  # No need to change lastFoundIndex
        return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection

    

pi = np.pi

fig = plt.figure()
trajectory_lines = plt.plot([], '-', color='orange', linewidth=4)
trajectory_line = trajectory_lines[0]
heading_lines = plt.plot([], '-', color='red')
heading_line = heading_lines[0]
connection_lines = plt.plot([], '-', color='green')
connection_line = connection_lines[0]
poses = plt.plot([], 'o', color='black', markersize=10)
pose = poses[0]

plt.plot(planned_path_positions[:, 0], planned_path_positions[:, 1], '--', color='grey')

plt.axis("scaled")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
dt = 100
xs = [currentPos[0]]
ys = [currentPos[1]]

smoothed_path = smoothing(planned_path_positions,0.1,0.3,0.001)
def pure_pursuit_animation(frame):
    global currentPos
    global currentHeading
    global lastFoundIndex
 

    goalPt, lastFoundIndex, SteeringAngle, turnError, linearVel, foundIntersection = pure_pursuit_step(smoothed_path, currentPos, currentHeading, lookAheadDis, lastFoundIndex)

    if not foundIntersection or np.all(currentPos == planned_path_positions[-1]):
        return  # Stop the animation 

   # model: 200rpm drive with 18" width
  #               rpm   /s  circ   feet 
    maxLinVelfeet = (200 / 60 * pi*4 / 12)*0.47 # the original is in feet therefore to set the appropriate velocity we have to scaled it so that it = 50 cm/s
  #               rpm   /s  center angle   deg
    maxTurnVelDeg = (200 / 60 * pi*4 / 9 *180/pi)*0.18124573
    TurnVel = (SteeringAngle/maxTurnVelDeg)* maxTurnVelDeg
    stepDis = linearVel/100 * maxLinVelfeet * dt/1000
    currentPos[0] += stepDis * np.cos(currentHeading * pi / 180)
    currentPos[1] += stepDis * np.sin(currentHeading * pi / 180)

    heading_line.set_data([currentPos[0], currentPos[0] + 0.5 * np.cos(currentHeading / 180 * pi)],
                          [currentPos[1], currentPos[1] + 0.5 * np.sin(currentHeading / 180 * pi)])
    connection_line.set_data([currentPos[0], goalPt[0]], [currentPos[1], goalPt[1]])

    currentHeading += TurnVel* (dt / 1000)
    if using_rotation == False:
        currentHeading = currentHeading % 360
        if currentHeading < 0:
            currentHeading += 360

    xs.append(currentPos[0])
    ys.append(currentPos[1])

    pose.set_data([currentPos[0]], [currentPos[1]])
    trajectory_line.set_data(xs, ys)

    # Print current position, turn error, and velocity
    print(f"Current Position: ({currentPos[0]}, {currentPos[1]})")
    print(f"Turn Error: {turnError} degrees")
    print(f"Steering Angle: {SteeringAngle} degrees")
    print(f"Velocity: {((linearVel/100) * maxLinVelfeet)*30.48} cm/s\n")

    return trajectory_line, heading_line, connection_line, pose  # Return updated plot elements

anim = animation.FuncAnimation(fig, pure_pursuit_animation, frames=numOfFrames, interval=50)
plt.show()


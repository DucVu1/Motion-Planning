import sys
sys.path.append(r"C:\Users\Duc\Downloads\Dashboard")
import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from queue import Queue
from CarCommunication.threadwithstop import ThreadWithStop
from multiprocessing import Pipe
import time
class PurePursuit(ThreadWithStop):
    def __init__(self, graph_file_path, start_node, goal_node,pipeRecv,pipeSend):
        
        self.linear_vel=10
        self.steer_angle=None
        self.pipeRecv=pipeRecv
        self.pipeSend=pipeSend
        super(PurePursuit, self).__init__()
        self.graph = self.parse_graphml(graph_file_path)
        self.start_node = start_node
        self.goal_node = goal_node
        self.planned_path = nx.shortest_path(self.graph, source=start_node, target=goal_node)
    
        self.planned_path_positions = np.array([self.graph.nodes[node]['pos'] for node in self.planned_path])
        self.planned_path_graph = self.graph.subgraph(self.planned_path)
        # self.visualize_map(self.graph,"")
        # self.visualize_map(self.planned_path_graph,"Planned")
        self.smoothed_path = self.smoothing(self.planned_path_positions, 0.1, 0.3, 0.001)

        self.currentPos=[15.477, -13.07]
        self.currentHeading = 220
        self.lastFoundIndex = 0
        self.lookAheadDis = 0.265
        self.using_rotation = False
        self.steeringAngle=0
        self.currentIndex = 0
        self.xs = [self.planned_path_positions[0, 0]]
        self.ys = [self.planned_path_positions[0, 1]]

        

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

    def visualize_map(self, graph,title=''):
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

    def pure_pursuit_step(self, path, currentHeading, lookAheadDis, LFindex):
        currentX = self.currentPos[0]
        currentY = self.currentPos[1]
        lastFoundIndex = LFindex
        foundIntersection = False
        startingIndex = lastFoundIndex

        while foundIntersection == False:
            if self.currentIndex >= len(path)-1:
               self.currentIndex-=1
            x1 = path[self.currentIndex][0] - currentX
            y1 = path[self.currentIndex][1] - currentY
            x2 = path[self.currentIndex + 1][0] - currentX
            y2 = path[self.currentIndex + 1][1] - currentY
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
                minX = min(path[self.currentIndex][0], path[self.currentIndex + 1][0])
                minY = min(path[self.currentIndex][1], path[self.currentIndex + 1][1])
                maxX = max(path[self.currentIndex][0], path[self.currentIndex + 1][0])
                maxY = max(path[self.currentIndex][1], path[self.currentIndex + 1][1])

                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)): 
                    linearVel =10
                    if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and (
                            (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                        if self.pt_to_pt_distance(sol_pt1, path[self.currentIndex + 1]) < self.pt_to_pt_distance(sol_pt2, path[self.currentIndex + 1]):
                            goalPt = sol_pt1
                            foundIntersection = True
                        else:
                            goalPt = sol_pt2
                            foundIntersection = True

                    else:
                        if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                            goalPt = sol_pt1
                            foundIntersection = True
    
                        else:
                            goalPt = sol_pt2
                            foundIntersection = True
            

                    if self.pt_to_pt_distance(goalPt, path[self.currentIndex + 1]) < self.pt_to_pt_distance([currentX, currentY], path[self.currentIndex + 1]):
                        lastFoundIndex = self.currentIndex
                        break
                    else:
                        linearVel = 0
                        lastFoundIndex = self.currentIndex + 1

                else:
                    linearVel = 0  # cm/s
                    goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]
            else:
                linearVel = 0  # cm/s
                goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]
                break
        condition = (self.currentIndex < len(path)-1)

        if foundIntersection and (not condition ^ foundIntersection):
            absTargetAngle = math.atan2(goalPt[1] - self.currentPos[1], goalPt[0] - self.currentPos[0]) * 180 / np.pi
            if absTargetAngle < 0:
                absTargetAngle += 360
            turnError = absTargetAngle - currentHeading
            if turnError > 180 or turnError < -180:
                turnError = -1 * self.sgn(turnError) * (360 - abs(turnError))
            if (turnError <0.01 and self.sgn(turnError) >0) or ((turnError >-0.01)and self.sgn(turnError)<0):
                R  =  float('inf')
            else:
                R = (self.lookAheadDis/2)/ math.sin(np.radians(turnError))
            steeringAngle = np.degrees((math.atan(0.265/R)))
            if steeringAngle >20:
                steeringAngle =20
                linearVel = 10
            elif steeringAngle < -20:
                steeringAngle =-20
                linearVel = 10
            else:
                steeringAngle = steeringAngle
            self.steeringAngle=steeringAngle    
            data2 = {"action": "speed", "value": 10}
            self.pipeSend.send(data2)
            data2 = {"action": "steer", "value": steeringAngle}
            self.pipeSend.send(data2)    
            return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection,R
        elif foundIntersection and ((condition) ^ foundIntersection):
            if self.currentPos[0] == path[-1][0] and self.currentPos[1] == path[-1][1]:
                R=0
                turnError = 0
                linearVel = 0
                steeringAngle = 0
                goalPt = [self.currentPos[0], self.currentPos[1]]
                lastFoundIndex = lastFoundIndex
                data2 = {"action": "speed", "value": 0}
                self.pipeSend.send(data2)
                data2 = {"action": "steer", "value": steeringAngle}
                self.pipeSend.send(data2)
                return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection,R
            else:
                absTargetAngle = math.atan2(goalPt[1] - self.currentPos[1], goalPt[0] - self.currentPos[0]) * 180 / np.pi
                if absTargetAngle < 0:
                    absTargetAngle += 360
                turnError = absTargetAngle - currentHeading

                if turnError > 180 or turnError < -180:
                    turnError = -1 * self.sgn(turnError) * (360 - abs(turnError))
                if (turnError <0.01 and self.sgn(turnError) >0) or ((turnError >-0.01)and self.sgn(turnError)<0):
                    R  =  float('inf')
                else:
                    R = (self.lookAheadDis/2)/ math.sin(np.radians(turnError))
                steeringAngle = np.degrees((math.atan(0.265/R)))
                if steeringAngle >20:
                    steeringAngle =20
                    linearVel = 10
                elif steeringAngle < -20:
                    steeringAngle =-20
                    linearVel = 10
                else:
                    steeringAngle = steeringAngle
                self.steeringAngle=steeringAngle    
                data2 = {"action": "speed", "value": 10}
                self.pipeSend.send(data2)
                data2 = {"action": "steer", "value": steeringAngle}
                self.pipeSend.send(data2)    
                return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection,R

        else:
            R=0
            turnError = 0
            linearVel = 0
            steeringAngle = 0
            self.steeringAngle = steeringAngle
            goalPt = [self.currentPos[0], self.currentPos[1]]
            lastFoundIndex = lastFoundIndex
            data2 = {"action": "speed", "value": 10}
            self.pipeSend.send(data2)
            data2 = {"action": "steer", "value": steeringAngle}
            self.pipeSend.send(data2)
            return goalPt, lastFoundIndex, steeringAngle, turnError, linearVel, foundIntersection,R

    def run(self):      
        while True:
            self.continuos_update()
            if self.currentPos is not None:
                self.pure_pursuit_step(self.planned_path_positions, currentHeading=self.currentHeading, lookAheadDis=self.lookAheadDis, LFindex=self.currentIndex)
            self.currentPos =None
    def continuos_update(self):
        if self.pipeRecv.poll():
            msg = self.pipeRecv.recv()
            if msg["action"] == "location":
                self.planned_path = msg["value"]
            self.currentPos=msg["value"]
            self.currentHeading +=((10/(0.265)))*0.5*math.tan(np.radians(self.steeringAngle))#ackerman odometry
            print(self.currentHeading)
                

if __name__ == "__main__":
    pipe1,pipe2=Pipe()
    pipe3,pipe4=Pipe()
# Define the file path to your GraphML file
    graph_file_path = 'Competition_track_graph.graphml'
    start_node = "424"
    goal_node = "401"
    car_animation = PurePursuit(graph_file_path, start_node, goal_node,pipe2,pipe3)
    car_animation.start()
    current_pos = [15.685, -9.7]
    currentHeading = 0
    while True:
        data = {"action": "location", "value":[current_pos[0],current_pos[1]]}
        pipe1.send(data)
        time.sleep(0.5)
        current_pos[0]+= 0
        current_pos[1] +=0
        while pipe4.poll():
            data=pipe4.recv()
            print(data)

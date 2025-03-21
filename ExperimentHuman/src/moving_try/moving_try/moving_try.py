#!/usr/bin/env python3

""" This simple mapper is loosely based on both the bitcraze cflib point cloud example
 https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/multiranger/multiranger_pointcloud.py
 and the webots epuck simple mapper example:
 https://github.com/cyberbotics/webots_ros2

 Originally from https://github.com/knmcguire/crazyflie_ros2_experimental/



<pose>6.0 0 0.5 0 0 0</pose>
<name>Box1</name>
</include>

    <include>
<uri>
model://Box
</uri>
<pose>2.5 -5.0 0.5 0 0 0</pose>
<name>Box2</name>
</include>

    <include>
<uri>
model://Box
</uri>
<pose>-3.0 -3.0 0.5 0 0 0</pose>
<name>Box3</name>

 """

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from std_msgs.msg import Int16, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Twist
from tf2_ros import StaticTransformBroadcaster
from std_srvs.srv import Trigger

import tf_transformations
import math
import numpy as np
import time
import random 

GLOBAL_SIZE_X = 20.0
GLOBAL_SIZE_Y = 20.0
MAP_RES = 0.1

import itertools
import numpy as np
import math

def expected(search, hide):  # k = 1
    pexp = []
    expected = 0 
    j = 1
    for i in search:
        pexp.append(p[i])
        if hide[0] == i:
            expected += math.prod(pexp[0:j])
        j += 1
    return expected

###
n = 3
k = 1

S = [i for i in range(n)]
p = [1/2, 3/4, 2/3]

search_strats = list(itertools.permutations(S))
hide_strats = list(itertools.combinations(S, k))

N = 10000

rho = 1  # parameter that depreciates past experiences
lamb = 9.5  # measure of sensitivity of players to attractions

# Initial event distribution (occurences)
Ns = [1,0,0,0,0,0]  # initial strategies chosen searcher
Nh = [1,0,0]  # initial strategies chosen hider
Nbelief = sum(Ns)  # initial count

# Initial belief of the players based on the initial distribution of events
Bsh = [i/sum(Nh) for i in Nh]  # Belief searcher has of hider
Bhs = [i/sum(Ns) for i in Ns]  # Belief hider has of searcher

def Eh(B, stratj):
    Eh = 0
    for i in range(len(search_strats)):
        Eh += -1*expected(search_strats[i], stratj)*B[i]  # *-1 if zero-sum?
    return Eh

def Es(B, stratj):
    Es = 0
    for i in range(len(hide_strats)):
        Es += expected(stratj, hide_strats[i])*B[i]
    return Es

# Attractions for each strategy given the belief of the players about the other players
Eh = [Eh(Bhs, i) for i in hide_strats]
Es = [Es(Bsh, i) for i in search_strats]

# Calculate the probabilities of choosing a strategy based on the attractions
Ph = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in Eh]) for j in Eh]
Ps = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in Es]) for j in Es]

#expected number of hidders
#0.40912



class Moving(Node):
    def __init__(self):

        super().__init__('simple_mapper_multiranger')
        
        self.declare_parameter('robot_prefix', '/crazyflie')
        robot_prefix = self.get_parameter('robot_prefix').value
        self.declare_parameter('delay', 5.0)
        self.delay = self.get_parameter('delay').value
        self.declare_parameter('max_turn_rate', 0.5)
        max_turn_rate = self.get_parameter('max_turn_rate').value
        self.declare_parameter('max_forward_speed', 0.5)
        max_forward_speed = self.get_parameter('max_forward_speed').value
        self.declare_parameter('wall_following_direction', 'right')
        self.wall_following_direction = self.get_parameter('wall_following_direction').value


        self.droneHid = False
        self.droneSea = True
        
        self.Es = Es
        self.Eh = Eh
        self.Ph = Ph
        self.Ps = Ps
        self.Nbelief = Nbelief
        self.Ns = Ns
        self.Nh = Nh
        self.S = S
        self.k = k
        self.n = n
        self.p = p
        self.a = [[1/2, 3/8, 1/4], [1/2, 1/4, 1/3], [3/8, 3/4 , 1/4], [1/4, 3/4, 1/2], [1/3, 1/4, 2/3], [1/4, 1/2, 2/3]]
        self.search_strats = list(itertools.permutations(self.S))
        self.hide_strats = list(itertools.combinations(self.S, self.k))
        
        if self.droneHid:
            H_index = np.random.choice([x[0] for x in itertools.combinations(self.S, self.k)], p=self.Ph).item()
            self.hidder = H_index
            self.Nh[H_index] += 1
        else:
            self.hidder= -1
        

        if self.droneSea:
            search_index = np.random.choice(list(range(math.factorial(self.n))), p=self.Ps).item()
            self.strategy = list(search_strats[search_index])
            self.Ns[search_index] += 1
        else:
            self.strategy = -1

        self.results = []
        self.expected_hiders_found_strats = []

        self.oldstrategy = -1
        self.history = []


        self.declare_parameter('trials',1)
        self.trials = self.get_parameter('trials').value
        self.counter = self.trials
        self.og_strategy = self.strategy

        self.get_logger().info(f"Hidder is set on {self.hidder} using the strategy {self.strategy}")

        self.hidders_found = 0

        self.odom_subscriber = self.create_subscription(
            Odometry, robot_prefix + '/odom', self.odom_subscribe_callback, 10)
        self.ranges_subscriber = self.create_subscription(
            LaserScan, robot_prefix + '/scan', self.scan_subscribe_callback, 10)
        self.searcher_subscriber = self.create_subscription(
            Int16, '/target', self.searcher_subscribe_callback, 10)
        self.hidder_subscriber = self.create_subscription(
            Int16, '/hidder', self.hidder_subscribe_callback, 10)


        self.searched_publisher = self.create_publisher(Int16, 'searched', 10)

        self.start_pos = False


        # add service to stop wall following and make the crazyflie land
        self.srv = self.create_service(Trigger, robot_prefix + '/stop_wall_following', self.stop_wall_following_cb)

        self.position = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.ranges = [0.0, 0.0, 0.0, 0.0]
        self.avoiding = False
        self.count = 5
        self.target = False
        self.boxes = [(5.0, 0.3 , 0.5),(-2.0, 7.2  , 0.25),(-2.2, -2.6, 0.33)]
        self.fail = False
        self.succ = False
        self.stat = -1
        
        self.score = [0,0,0,0,0,0,0,0,0,0,0]
        

        self.twist_publisher = self.create_publisher(Twist, robot_prefix+'/cmd_vel', 10)

        # Create a timer to run the wall following state machine
        self.timer = self.create_timer(0.01, self.timer_callback)

        # Give a take off command but wait for the delay to start the wall following
        self.wait_for_start = True
        self.start_clock = self.get_clock().now().nanoseconds * 1e-9
        msg = Twist()
        msg.linear.z = 1.5
        self.twist_publisher.publish(msg)


        self.status_publisher = self.create_publisher(String, '/status', 10)
        self.scoreS_publisher = self.create_publisher(String, '/score', 10)

    def expected(self,search, hide):  # k = 1
        pexp = []
        expected = 0 
        j = 1
        for i in search:
            pexp.append(self.p[i])
            if hide[0] == i:
                expected += math.prod(pexp[0:j])
            j += 1
        return expected



    def stop_wall_following_cb(self, request, response):
        self.get_logger().info('Stopping')
        self.timer.cancel()
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = -0.2
        msg.angular.z = 0.0
        self.twist_publisher.publish(msg)

        response.success = True

        return response

    def search(self):
        rho = 1  # parameter that depreciates past experiences
        lamb = 9.5


        rand = random.random()

        targetP = self.boxes[self.strategy[0]][2]
        index = self.strategy.pop(0)

        if self.start_pos:
            self.start_pos = False
            self.target = False
            self.boxes.pop(0)

            list_hid = [self.hidder]
            self.expected_hiders_found_strats.append(self.expected(self.og_strategy, list_hid))
            
            self.Es = [(rho * self.Nbelief * self.Es[i] + self.expected(self.search_strats[i], list_hid))/(rho * self.Nbelief + 1) for i in range(len(self.Es))]
            self.Eh = [(rho * self.Nbelief * self.Eh[i] - self.expected(self.og_strategy, self.hide_strats[i]))/(rho * self.Nbelief + 1) for i in range(len(self.Eh))]
            
            self.Ph = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in self.Eh]) for j in self.Eh]
            self.Ps = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in self.Es]) for j in self.Es]
    

            if self.droneHid:
                H_index = np.random.choice([x[0] for x in itertools.combinations(self.S, self.k)], p=self.Ph).item()
                self.hidder = H_index
                self.Nh[H_index] += 1
            else:
                self.hidder= -1

            if self.droneSea:
                search_index = np.random.choice(list(range(math.factorial(self.n))), p=self.Ps).item()
                self.strategy = list(search_strats[search_index])
                self.Ns[search_index] += 1
                self.og_strategy = self.strategy[:]
            else:
                self.strategy = -1
                
            self.stat = -1
            self.Nbelief += 1
            msg = Twist()
            self.twist_publisher.publish(msg)

            msg = String()
            msg.data = "Waiting for new strategy selection"
            self.status_publisher.publish(msg)

            return
        

        if rand > targetP:
            if self.hidder == index:
                self.start_pos = True
                self.score[self.hidder+6]+= 1
                self.hidders_found +=1
                self.strategy = [0]
                self.boxes.insert(0,(0, 0, 1.1))
                self.get_logger().info('Found')
                self.counter -=1
                self.stat = 0
                self.score[10] += 1
                self.score[9] += 1
                self.score[index]+= 1
                self.results.append(1)



            self.target=False

        else:
            self.target=False
            self.start_pos = True
            self.score[self.hidder+6]+= 1
            self.strategy = [0]
            self.boxes.insert(0,(0, 0, 1.1))
            self.get_logger().info('Failed')
            self.counter -=1
            self.stat = 1
            self.score[9] += 1
            self.score[index]+= 1
            self.results.append(0)
        


    def fall(self):
        self.get_logger().info('Done')
        self.timer.cancel()
        msg1 = Twist()
        msg1.linear.x = 0.0
        msg1.linear.y = 0.0
        msg1.linear.z = -0.2
        self.twist_publisher.publish(msg1)
        return
        



    def timer_callback(self):
        #self.get_logger().info(f'{self.strategy}')
        m = String()
        m.data = ','.join(str(i) for i in self.score)
        self.scoreS_publisher.publish(m)
        
        if self.counter == 0 and not self.start_pos:
            self.fall()
            msg = String()
            msg.data = "The Game has ended. Thank you for playing"
            self.status_publisher.publish(msg)

            VV = np.dot(self.Ps, np.dot(self.a, self.Ph))
            V_perc_diff = float(round((VV - 0.4090909090)/((VV + 0.4090909090)/2) * 100,1))

            self.get_logger().info("V percentage difference: {}".format(V_perc_diff))
            self.get_logger().info("Average expected hiders found from strats {}".format(sum(self.expected_hiders_found_strats)/len(self.expected_hiders_found_strats)))
            self.get_logger().info("Average hiders found: {}".format(sum(self.results)/len(self.results)))
            return
        

        # wait for the delay to pass and then start wall following
        if self.strategy == -1:
            msg = Twist()
            error = 1.5 - self.position[2]
            msg.linear.z = error
            self.twist_publisher.publish(msg)

            msg = String()
            msg.data = "Waiting for search strategy selection"
            self.status_publisher.publish(msg)
            return

        elif self.hidder == -1:
            msg = Twist()
            error = 1.5 - self.position[2]
            msg.linear.z = error
            self.twist_publisher.publish(msg)

            msg = String()
            msg.data = "Waiting for hidding spot selection"
            self.status_publisher.publish(msg)
            return

        elif self.stat == 0:
            msg = String()
            msg.data = f"Found Hidder at spot {self.strategy[0]+1}, returning to start"
            self.status_publisher.publish(msg)
        elif self.stat == 1:
            msg = String()
            msg.data = f"Failed the search. They hid at {self.hidder+1}, returning to start"
            self.status_publisher.publish(msg)
        else:
            msg = String()
            msg.data = f"Traveling to Spot {self.strategy[0]+1} and search"
            self.status_publisher.publish(msg)




        front_range = self.ranges[2]

        msg = Twist()
        time_now = self.get_clock().now().nanoseconds * 1e-9
        

        if front_range < 0.3:
            self.avoiding = True
            self.count = 20
        else:
            self.avoiding = False
        

        targetX = self.boxes[self.strategy[0]][0]
        targetY = self.boxes[self.strategy[0]][1]

        distX = (targetX-self.position[0])
        distY = (targetY-self.position[1])

        z_temp = 0.0
        x_temp = 0.0
        angle_to_obj = math.atan2(distX,distY)
        angle_to= angle_to_obj+math.pi*(6/4) if angle_to_obj < - math.pi/2 else angle_to_obj-math.pi/2
        calc = angle_to + self.angles[2]

        if abs(calc) > math.pi:
            sig = 1 if calc < 0 else -1
            res = abs(calc) % math.pi
            calc = res*sig
        


        if calc > 0.2:
            z_temp = -0.5
        elif calc < -0.2:
            z_temp = 0.5
        else:
            z_temp = 0.0
            self.target=True

        if self.target:
            x_temp= 0.8
        else:
            x_temp= 0.0

        #right_range = self.ranges[1]
        #front_range = self.ranges[2]
        #left_range = self.ranges[3]
        

        if self.avoiding or self.count > 0 or self.ranges[3] < 0.1 or self.ranges[1] < 0.1:
            if self.ranges[3] > self.ranges[1]:
                msg.linear.y = 0.5
            else:
                msg.linear.y = -0.5
            msg.angular.z = 0.0
            msg.linear.x = 0.0
            self.count-=1
        else:
            msg.angular.z = z_temp
            msg.linear.x = x_temp
            msg.linear.y = 0.0
        error = 1.5 - self.position[2]
        msg.linear.z = error

        self.twist_publisher.publish(msg)

        if abs(distX) < 0.2 and abs(distY) < 0.2:
            self.search()



    def odom_subscribe_callback(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        euler = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.angles[0] = euler[0]
        self.angles[1] = euler[1]
        self.angles[2] = euler[2]
        self.position_update = True

    def scan_subscribe_callback(self, msg):
        self.ranges = msg.ranges

    def searcher_subscribe_callback(self, msg):
        strats = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        ind = msg.data
        self.get_logger().info(f'strategy {strats[ind]}')
        self.strategy = strats[ind]
        self.og_strategy = self.strategy[:]

    def hidder_subscribe_callback(self, msg):
        ind = msg.data
        self.get_logger().info(f'hidding spot {ind}')
        self.hidder = ind



def main(args=None):

    rclpy.init(args=args)
    mover = Moving()
    rclpy.spin(mover)
    rclpy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

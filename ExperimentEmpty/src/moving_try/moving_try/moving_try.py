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
import pickle

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

def EhF(B, stratj):
    Eh = 0
    for i in range(len(search_strats)):
        Eh += -1*expected(search_strats[i], stratj)*B[i]  # *-1 if zero-sum?
    return Eh

def EsF(B, stratj):
    Es = 0
    for i in range(len(hide_strats)):
        Es += expected(stratj, hide_strats[i])*B[i]
    return Es

# Attractions for each strategy given the belief of the players about the other players
Eh = [EhF(Bhs, i) for i in hide_strats]
Es = [EsF(Bsh, i) for i in search_strats]

# Calculate the probabilities of choosing a strategy based on the attractions
Ph = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in Eh]) for j in Eh]
Ps = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in Es]) for j in Es]

#expected number of hidders
#0.40912



class Moving(Node):
    def __init__(self):
        super().__init__('simple_mapper_multiranger')
        self.fileName = 'ExperimentSup'
        self.historyList = []

        self.exists = True

        if self.exists:
            with open('/home/coders/ExperimentEmpty/'+self.fileName+'.pkl', 'rb') as f:
                self.historyList= pickle.load(f)

        self.droneHid = False
        self.droneSea = False
        self.partSearch = False
        
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
        

        self.stratInt = -1
        if self.droneSea:
            search_index = np.random.choice(list(range(math.factorial(self.n))), p=self.Ps).item()
            self.strategy = list(search_strats[search_index])
            self.Ns[search_index] += 1
            self.stratInt = search_index
        else:
            self.strategy = -1

        self.results = []
        self.expected_hiders_found_strats = []

        self.oldstrategy = -1
        self.history = []

        self.swtichTimer = 200



        self.found = [0,0,0]
        self.captured = [0,0,0]
        self.score = [0,0,0,0,0,0,0,0,0,0,0]
        self.og_strategies =[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        


        self.declare_parameter('trials',200)
        self.trials = self.get_parameter('trials').value
        self.counter = self.trials
        self.og_strategy = self.strategy
        self.allH = ['naive', 'robot']
        self.allS = ['naive1', 'naive2','robot']
        self.Op = -1

        if self.droneHid:
            self.Op = self.allH
        elif self.droneSea:
            self.Op = self.allS
        else:
            self.Op = ['Human']

        self.main = 'Participant3'


        self.get_logger().info(f"Hidder is set on {self.hidder} using the strategy {self.strategy}")

        self.hidders_found = 0
        self.searcher_subscriber = self.create_subscription(
            Int16, '/target', self.searcher_subscribe_callback, 10)
        self.hidder_subscriber = self.create_subscription(
            Int16, '/hidder', self.hidder_subscribe_callback, 10)


        self.searched_publisher = self.create_publisher(Int16, 'searched', 10)
        self.opponentH_publisher = self.create_publisher(String, 'opponentH', 10)
        self.opponentS_publisher = self.create_publisher(String, 'opponentS', 10)
        self.prev_publisher = self.create_publisher(String, 'prev', 10)

        self.last = 'No iteration run yet'

        self.start_pos = False
        # Create a timer to run the wall following state machine
        self.timer = self.create_timer(0.01, self.timer_callback)


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
        





    def timer_callback(self):
        m = String()
        m.data = ','.join(str(i) for i in self.score)
        self.scoreS_publisher.publish(m)

        m = String()
        m.data = self.last
        self.prev_publisher.publish(m)

        if len(self.Op)==0:
            self.timer.cancel()
            msg = String()
            msg.data = "Game ended, all opponents played"
            self.status_publisher.publish(msg)


            with open('/home/coders/ExperimentEmpty/'+self.fileName+'.pkl', 'wb') as f:
                pickle.dump(self.historyList, f)

            
            return


        if self.counter == 0:
            if self.swtichTimer == 0:
                self.swtichTimer = 200
                self.counter+=self.trials
                self.Op.pop(0)
                self.score = [0,0,0,0,0,0,0,0,0,0,0]
                

                VV = np.dot(self.Ps, np.dot(self.a, self.Ph))
                V_perc_diff = float(round((VV - 0.4090909090)/((VV + 0.4090909090)/2) * 100,1))
                self.get_logger().info("V percentage diff: {}".format(V_perc_diff))
                self.get_logger().info("Average expected hiders found from strats {}".format(sum(self.expected_hiders_found_strats)/len(self.expected_hiders_found_strats)))
                self.get_logger().info("Succes percentage: {}".format(sum(self.results)/len(self.results)))

                self.results = []
                self.expected_hiders_found_strats = []

                self.Eh = [EhF(Bhs, i) for i in self.hide_strats]
                self.Es = [EsF(Bsh, i) for i in self.search_strats]
                
                # Calculate the probabilities of choosing a strategy based on the attractions
                self.Ph = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in self.Eh]) for j in self.Eh]
                self.Ps = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in self.Es]) for j in self.Es]
                return
            else :
                msg = String()
                msg.data = "Current opponent finished. Switching to next one"
                self.status_publisher.publish(msg)
                self.swtichTimer -= 1
                return

            

        
        if self.droneHid:
            msg = String()
            msg.data = self.Op[0]
            self.opponentH_publisher.publish(msg)
        elif self.droneSea:
            msg = String()
            msg.data = self.Op[0]
            self.opponentS_publisher.publish(msg)
        else:
            msg = String()
            msg.data = self.Op[0]
            self.opponentS_publisher.publish(msg)
            self.opponentH_publisher.publish(msg)

        if self.strategy == -1:
            msg = String()
            msg.data = "Waiting for search strategy selection"
            self.status_publisher.publish(msg)
            return

        elif self.hidder == -1:
            msg = String()
            msg.data = "Waiting for hidding spot selection"
            self.status_publisher.publish(msg)
            return

        


        msg = String()
        msg.data = "searching"
        self.status_publisher.publish(msg)

        msg = String()
        H = [self.hidder]
        self.expected_hiders_found_strats.append(expected(self.strategy, [self.hidder]))
        for i in self.strategy:  # iterate through the search strategy
            if self.p[i] > np.random.uniform(0,1):  
                # searcher is not captured
                if i in H: 
                    # found hider
                    H.remove(i)
                    if len(H) == 0:
                        # all hiders found, end game
                        self.results.append(1)
                        self.found[i] += 1
                        self.score[self.hidder+6]+= 1
                        self.score[10] += 1
                        self.score[9] += 1
                        self.score[self.stratInt]+= 1
                        msg.data = "Hider found at location {}".format(i+1)

                        t = -1
                        if self.droneHid or self.partSearch:
                            t = [self.Op[0],self.main]
                        else:
                            t = [self.main,self.Op[0]]


                        self.historyList.append([t[0],t[1],201-self.counter,self.stratInt,self.hidder,1,i])
                        break                     
                    else:
                        # still hiders remaining
                        continue
                else:
                    # didn't found a hider
                    continue 
            else:
                # searcher is captured, end game
                self.results.append(0)
                self.captured[i] += 1
                self.score[self.hidder+6]+= 1
                self.score[9] += 1
                self.score[self.stratInt]+= 1
                msg.data =  "Searcher captured at location {}".format(i+1)
                t = -1
                if self.droneHid or self.partSearch:
                    t = [self.Op[0],self.main]
                else:
                    t = [self.main,self.Op[0]]
                self.historyList.append([t[0],t[1],201-self.counter,self.stratInt,self.hidder,0,i])
                break


        self.prev_publisher.publish(msg)
        self.last = msg.data

        self.Es = [(rho * self.Nbelief * self.Es[i] + self.expected(self.search_strats[i], [self.hidder]))/(rho * self.Nbelief + 1) for i in range(len(self.Es))]
        self.Eh = [(rho * self.Nbelief * self.Eh[i] - self.expected(self.og_strategy, self.hide_strats[i]))/(rho * self.Nbelief + 1) for i in range(len(self.Eh))]
        
        self.Ph = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in self.Eh]) for j in self.Eh]
        self.Ps = [np.exp(j*lamb)/sum([np.exp(i*lamb) for i in self.Es]) for j in self.Es]
        self.Nbelief += 1
        self.counter -= 1

        if self.droneHid:
            if self.Op[0] == 'naive':
                H_index = 0
            elif self.Op[0] == 'robot':
                H_index = np.random.choice([x[0] for x in itertools.combinations(self.S, self.k)], p=self.Ph).item()
            
            self.hidder = H_index
            self.Nh[H_index] += 1
        else:
            self.hidder= -1

        if self.droneSea:
            if self.Op[0] == 'naive1':
                search_index = 1  # risky player
            elif self.Op[0] == 'naive2':
                search_index = 3 
            elif self.Op[0] == 'robot':
                search_index = np.random.choice(list(range(math.factorial(self.n))), p=self.Ps).item()
            
            self.strategy = list(search_strats[search_index])
            self.Ns[search_index] += 1
            self.og_strategy = self.strategy[:]
            self.stratInt = search_index
        else:
            self.strategy = -1

        


    def searcher_subscribe_callback(self, msg):
        strats = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        ind = msg.data
        self.strategy = strats[ind]
        self.og_strategy = self.strategy[:]
        self.stratInt = ind
        self.Ns[ind] += 1

    def hidder_subscribe_callback(self, msg):
        ind = msg.data
        self.hidder = ind
        self.Nh[ind] += 1



def main(args=None):

    rclpy.init(args=args)
    mover = Moving()
    rclpy.spin(mover)
    rclpy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

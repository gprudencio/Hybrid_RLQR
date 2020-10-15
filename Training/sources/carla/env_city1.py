from sources.carla.control import *
sys.path.append('./sources/carla/PythonAPI/')
from sources.carla.matrix import observation_matrix

from scipy.stats import norm

import random
import time

# ==============================================================================
# -- CARLA ENV ---------------------------------------------------------------
# ==============================================================================

class CarlaEnv:

    # INIT
    def init( self, args ):

        # Connect
        pygame.init()
        pygame.font.init()

        result = None
        while result is None:
            try:
                print('Trying to connect client')
                self.client = carla.Client(args.host, args.port)
                self.client.set_timeout(10.0)
                result = self.client.get_world()
            except:
                 pass

        # FIXED PARAMETERS
        self.continuous = args.continuous
        # Image size
        self.w, self.h = args.width, args.height
        # World coordinates
        spawn_points = self.client.get_world().get_map().get_spawn_points()
        #self.spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.spawn_point = spawn_points[1] if spawn_points else carla.Transform()                     # Spawn point
        self.start_pos_x, self.start_pos_y = self.spawn_point.location.x, self.spawn_point.location.y # Start points
        self.final_pos_x, self.final_pos_y = -7.5299, 300                                                 # Final points
        # Intersection points
        self.inter_x_left  = -5.628998
        self.inter_x_right = -9.409000
        # Auxiliar
        self.low_vel = 0
        self.desired_speed = args.desired_speed
        self.old_car_y     = self.start_pos_y
        self.old_car_speed = 0
        self.old_collision = 0
        self.old_car_x     = 0
        self.step_count = 0
        self.transition = False
        self.throttle_intensity = 0.7
        self.steer = 0
        self.brake = 0
        # Module
        self.old_rs = 0
        self.old_rc = 0
        self.old_rm = 0
        self.old_ri = 0
        self.SIC_lo = 0
        self.SIC_la = 0
        self.actn_list = []
        self.m_actn_list = []
        self.car_x_list = []
        self.save_list = 0
        self.vel_list = []
        
        self.loop_iter = 0

        # Init
        self.display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(args.width, args.height)
        self.world = World(self.client.get_world(), self.hud, 'vehicle.*', self.spawn_point)
        self.invade = False
        self.controller = KeyboardControl(self.world, args.autopilot)
        self.clock = pygame.time.Clock()
        
        # Waypoints
        #self.waypoints = self.world.get_map().get_waypoint(self.world.vehicle.get_location())
        
        self.waypoint = self.client.get_world().get_map().get_waypoint(self.world.player.get_location())

        # Actor list
        self.actor_list = self.world.world.get_actors()

        # Spawn NPCs
        self.world.add_npcs(args.number_of_vehicles)
        
        # Semantic segmentation
        self.world.camera_manager._index = 5        
        
        
    def behaviour_module( self, rs, rc, rm, ri, old_rs, old_rc, old_rm, old_ri ):

        # Longitudional sensitivity
        di = rs
        ds = rs - old_rs
        if ds >= 0: delta_1 = di
        else:       delta_1 = -di

        # Lateral sensitivity
        ki = rm + ri
        ks = (rm + ri) - (old_rm + old_ri)
        if ks >= 0: delta_2 = ki
        else:       delta_2 = -ki

        if delta_1 == 0: ES1 = 0 # [neutral]
        if delta_1 > 0:  ES1 = 1 # [motivation]  - reference approach
        if delta_1 < 0:  ES1 = 2 # [frustration] - reference deviation

        if delta_2 == 0: ES2 = 0 # [neutral]
        if delta_2 > 0:  ES2 = 1 # [relief]  - reference approach
        if delta_2 < 0:  ES2 = 2 # [anguish] - reference deviation

        return ES1, ES2

    # STEP
    def step( self, actn ):

#        self.invade = self.world.lane_invasion_sensor.invade
#        self.waypoint = self.client.get_world().get_map().get_waypoint(self.world.player.get_location())        

        # Car info list
        self.car_id, self.car_theta, self.car_x, self.car_y, self.car_speed = self.get_car_info()
        self.car_info = [self.car_id, self.car_theta, self.car_x, self.car_y, self.car_speed, self.start_pos_x, self.start_pos_y]

        # Map actn to pressed keys
        self.keys = list(pygame.key.get_pressed())
        self.keys = self.map_actions(self.keys, actn)
        
        # Parse events and render
        self.clock.tick_busy_loop(60)
        if self.controller.parse_events(self.world, self.clock, self.keys, self.throttle_intensity, self.steer, self.brake, self.continuous):
            return
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()

        # Return info (obs, rewd, done)
        obsv = self.get_obsv()
        done = self.get_done()
        rewd = self.get_rewd(actn) # Vector of rewards

        # Module (no actions)
        ES1, ES2 = self.behaviour_module(rewd[0], rewd[1], rewd[2], rewd[3], self.old_rs, self.old_rc, self.old_rm, self.old_ri)

        # Update
        self.old_rs, self.old_rc, self.old_rm, self.old_ri = rewd[0], rewd[1], rewd[2], rewd[3]
        obsv.append([ES1,ES2])

        return obsv, rewd, done

    # GET OBSV
    def get_obsv( self ):

        rgb_image = self.world.camera_manager.image
        collision = self.world.collision_sensor.collision
        distance_to_goal = (self.car_y - self.start_pos_y) / np.abs(self.final_pos_y - self.start_pos_y)
        matrix = observation_matrix( self.actor_list, self.car_info, self.inter_x_left, self.inter_x_right )[:,:,0]

        return [rgb_image, matrix, collision, self.car_speed*3.6/100, distance_to_goal]

	# REWARDS
    def get_rewd( self, actn ): # immediate rewards (1)

        # ====================================================================
        # Variables and weights
        # ====================================================================
        rs, rc, ri, rb = 0,0,0,0
        ws, wc, wi, wb = 0,100,0,10

        # Car position
        car_x = self.world.player.get_transform().location.x
        car_y = self.world.player.get_transform().location.y
        rot_c = self.world.player.get_transform().rotation.yaw 
        # Waypoints position
        way_x = []
        way_y = []
        
        # Time
        start = time.time()
        time.clock()    
        elapsed = time.time() - start
        result = None
        self.waypoint = []
        while elapsed <= 7:
            elapsed = time.time() - start     
            try:
                self.waypoint = self.client.get_world().get_map().get_waypoint(self.world.player.get_location())
                result = self.waypoint.transform.location.x
                way_x = self.waypoint.transform.location.x
                way_y = self.waypoint.transform.location.y
                rot_w = self.waypoint.transform.rotation.yaw 
            except:
                 pass
            if result is not None:
                break
            
        if result is None:
            print('Error Sensor') 
            self.low_vel == 200    
            self.waypoint = []
            way_x = []
            way_y = []
            rot_w = []
            way_x = car_x 
            way_y = car_y
            rot_w = rot_c
   
        car_speed = self.car_speed * 3.6

                
        # ====================================================================
        # Longitudinal reward
        # ====================================================================
        
        ds = 15 - car_speed
        rs = norm.pdf(round(ds,3),0,10)        
        rs = rs * 25 
                            
        # ====================================================================
        # Lateral reward - Traditional One
        # ==================================================================== 
        
        # Orientation    
        if (rot_c >= -5 and rot_c <= 5) or (rot_c >= -185 and rot_c <= -150) :
            rot_c = np.abs(rot_c)
        if (rot_w >= -185 and rot_w <= -150) :
            rot_w = np.abs(rot_w)        
        if rot_w >= 180: 
            rot_w = rot_w - 360 
            
        # Vehicle against
        D = np.absolute(rot_c - rot_w)  
        d = 0
        if D >= 150: 
            d = 4
            
        # Centrality reward
        dx = np.sqrt((np.abs(car_x + d - way_x))**2 + (np.abs(car_y + d - way_y))**2)
        rc = norm.pdf(round(dx,3),0,1) 
        rc = rc * 2.5  
        
        # ====================================================================
        # Colision reward
        # ====================================================================
        rb = -1 * (np.abs(self.world.collision_sensor.collision) / 100)
        
        # ====================================================================
        # Position
        # ====================================================================
#        print('Carx',self.car_x,'Cary',self.car_y)
        
        
        return ws*rs, wc*rc, wi*ri, wb*rb
    
    # Reference points to control
    def reference(self):

        tr, st = self.loop_iter, 0
       
        return tr, st

	# DONE CONDITIONS
    def get_done( self ):

        done = False
        if (280 < self.car_x < 282 and self.car_y < 134) or \
           self.world.collision_sensor.collision / 100 > 50   or \
           self.low_vel == 100:

            done = True
            
            self.loop_iter += 1
            self.reference()
                        
            self.world.restart()
            self.low_vel = 0

            self.old_car_y        = self.start_pos_y
            self.old_car_speed    = 0
            self.old_collision    = 0
            self.old_car_x        = 0

            self.inter_x_left  = -5.628998
            self.inter_x_right = -9.409000

        return done

	# GET CURRENT CAR INFO
    def get_car_info( self ):

        # Get car id
        car_id = self.world.player.id
        # Position
        car_x = self.world.player.get_transform().location.x
        car_y = self.world.player.get_transform().location.y
        # Theta
        car_theta = self.world.player.get_transform().rotation.yaw - 90
        # Speed
        vel = self.world.player.get_velocity()
        car_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        if car_speed < 0.2: self.low_vel += 1

        # Update
        car_x = self.world.player.get_transform().location.x
        car_y = self.world.player.get_transform().location.y

        dist_esquerda = -1.9010
        dist_direita = 1.8790

        self.world.lane_invasion_sensor.invade = True

        if car_x <= 0 and car_y < 295:

            self.inter_x_left  = self.start_pos_x - dist_esquerda
            self.inter_x_right = self.start_pos_x - dist_direita

            if np.abs(car_x) > np.abs(self.inter_x_left) and np.abs(car_x) < np.abs(self.inter_x_right):
                self.world.lane_invasion_sensor.invade = False

        if car_x <= 0 and car_y > 295:

            theta = np.arcsin((car_x - self.final_pos_x)/11.5)
            #print('theta     ',theta*180/3.141)

            self.inter_x_left  = self.start_pos_x - dist_esquerda * np.cos(theta)
            self.inter_x_right = self.start_pos_x - dist_direita * np.cos(theta)

            if np.abs(car_x) > np.abs(self.inter_x_left) and np.abs(car_x) < np.abs(self.inter_x_right):
                self.world.lane_invasion_sensor.invade = False

        if car_x > 0:

            self.inter_x_left = 306.5 - np.abs(dist_esquerda)
            self.inter_x_right = 306.5 + np.abs(dist_direita)

            if np.abs(car_y) > np.abs(self.inter_x_left) and np.abs(car_y) < np.abs(self.inter_x_right):
                self.world.lane_invasion_sensor.invade = False

        #print(car_x,self.inter_x_left, self.inter_x_right, car_y)

        return car_id, car_theta, car_x, car_y, car_speed

    # MAP ACTIONS
    def map_actions( self, keys, actn ):
               
        tr, st = 0.3, 0
        
        if keys[K_LEFT] == 1:
            st = - .3 #.35
        if keys[K_RIGHT] == 1:
            st = .3 #.35
        
        if keys[K_UP] == 1:
            tr = 0.2 
        if keys[K_DOWN] == 1: 
            tr = 0
                       
        self.throttle_intensity = tr 
        self.brake = 0       
        self.steer = np.clip(actn[2],-0.6,0.6)                

        return keys

    # OPEN SERVER
    def open_server( self, args ):

        with open('./sources/carla/CarlaLogs.txt', "wb") as out:

#            cmd = [os.path.join('./sources/carla/', 'CarlaUE4.sh'), '/Game/Carla/Maps/Town01']
            cmd = [os.path.join('./sources/carla/', 'CarlaUE4.sh'), '/Game/Carla/Maps/Town01', \
                "-benchmark -fps=10 -quality-level=Low -windowed -ResX={} -ResY={}".format(300, 300)]
            p = subprocess.Popen(cmd, stdout=out, stderr=out, env=None)

        return p

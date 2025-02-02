import math
import numpy as np
import random

EXTRA_ROTATION = 0#math.pi

def add_circle(center_x, center_y, radius, num_gates, rad_ang):
  positions = []

  angle_increment = rad_ang / num_gates

  for i in range(int(num_gates)):
      # Calculate the angle for this position
      angle = i * angle_increment
      
      # Calculate the position coordinates
      x = center_x + radius * math.cos(angle)
      y = center_y + radius * math.sin(angle)
      
      # Calculate the orientation (in radians) based on the angle
      orientation = angle - math.pi / 2 

      # Append the position and orientation to the list
      positions.append((x, y, 0, 0, 0, orientation + EXTRA_ROTATION))
  
  return positions

def add_line(center_x, center_y, length, num_gates, orientation):
    positions = []

    # Calculate the angle increment based on the orientation
    angle_increment = orientation

    # Calculate the step size along the line
    step = length / num_gates

    for i in range(int(num_gates)):
        # Calculate the position coordinates
        x = center_x + i * step * math.cos(angle_increment)
        y = center_y + i * step * math.sin(angle_increment)

        # Append the position and orientation to the list
        positions.append((x, y, 0, 0, 0, orientation + EXTRA_ROTATION))

    return positions

def ellipse():
  # Random poses
  num_gates = 30
  theta = np.linspace(0, 2 * np.pi, num_gates, endpoint=False)  # Angle for each gate

  # Define the radius and center for the oval path
  a = 50  # Semi-major axis (horizontal radius)
  b = 30  # Semi-minor axis (vertical radius)
  center_x = 0
  center_y = 0

  xposes = center_x + a * np.cos(theta)
  yposes = center_y + b * np.sin(theta)
  zposes = []

  # Calculate the orientation of each point
  orientations = []
  for i in range(num_gates):
      next_i = (i + 1) % num_gates  # Next index, wrapping around for the last point
      dx = xposes[next_i] - xposes[i]
      dy = yposes[next_i] - yposes[i]
      orientation = np.arctan2(dy, dx)
      orientations.append(orientation)

      zposes.append(random.randrange(-1, 4))
  poses = [(xposes[i], yposes[i], zposes[i], 0, 0, orientations[i]) for i in range(num_gates)]

  return poses


def randomLine():
  n_gates = 50
  poses = [(3, 0, 0, 0, 0, math.pi)]

  for i in range(n_gates):
    x = random.randint(8, 20) + poses[-1][0]
    y = random.randint(-10, 10) + poses[-1][1]
    z = random.randint(-1, 4) 
    orient = np.arctan2(poses[-1][1] - y, poses[-1][0] - x)

    poses.append((x, y, z, 0, 0, orient)) 
  
  return poses


# poses = randomLine()

## Defined circle poses
poses = [
        [-6.47404, -6.98526, 2.7153, 0, 0, -2.96],
        [-13.5561, -9.59085, -0.4432, 0, 0, 0],
        [-18.1517, -15.3409, 1.2568, 0, 0, 0],
        [-19.799, -25.3114, -0.3211, 0, 0, 0],
        [-23.6129, -32.0429, 0.7894, 0, 0, 0],
        [-31.958, -34.5389, 2.4873, 0, 0, 0],
        [-38.6226, -31.5022, 1.9345, 0, 0, 0],
        [-39.2746, -22.6062, 0.5432, 0, 0, 0],
        [-37.9813, -13.5741, 1.8765, 0, 0, 0],
        [-32.0881, -6.03558, -0.1245, 0, 0, 0],
        [-24.4027, 0.172298, 2.1534, 0, 0, 0],
        [-16.109, 3.8169, 1.4532, 0, 0, 0],
        [-9.17197, 9.86202, -0.3214, 0, 0, 0],
        [-3.33669, 16.8607, 0.8765, 0, 0, 0],
        [2.46026, 23.7222, 2.6543, 0, 0, 0],
        [6.6366, 33.7054, 1.1234, 0, 0, 0],
        [13.2114, 42.9063, -0.9876, 0, 0, 0],
        [24.221, 48.7494, 1.2345, 0, 0, 0],
        [35.1856, 46.0491, 0.5432, 0, 0, 0],
        [42.5478, 39.8838, 1.8765, 0, 0, 0],
        [44.1371, 32.2892, -0.1245, 0, 0, 0],
        [41.0288, 22.9648, 2.1534, 0, 0, 0],
        [37.2517, 14.0092, 1.4532, 0, 0, 0],
        [30.4535, 6.98368, -0.3214, 0, 0, 0],
        [21.9232, 1.62634, 0.8765, 0, 0, 0],
        [12.027, -2.23824, 2.6543, 0, 0, -2.79],
        [2.09309, -4.03768, 1.1234, 0, 0, -2.853]
]


for i in range(1, len(poses)- 1):

  if poses[i][5] == 0:
    poses[i][5] = ((np.arctan2(poses[i][1] - poses[i-1][1], poses[i][0] - poses[i-1][0]) + np.arctan2(poses[i+1][1] - poses[i][1], poses[i+1][0] - poses[i][0]))/2) - math.pi

  #poses[i][2] = random.randrange(-1, 3) 
  #poses[i][2] = 0.0


xml_template = '''
<include>
    <uri>model://aruco_gate_1</uri>
    <name>aruco_gate_{}</name>
    <pose>{}</pose>
</include>
'''

# Generate XML for each pose
xml_instances = [xml_template.format(i, ' '.join(map(str, pose))) for i, pose in enumerate(poses, start=1)]


#ground_plane or grass_plane


# Generate the final XML content
xml_content = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="generatedWorld">

    <include> 
      <uri>model://grass_plane</uri>
    </include>
	<scene>
      <grid>false</grid>
    </scene>

    <include>
      <uri>model://sun</uri>
    </include>

{"".join(xml_instances)}

    <!-- Gazebo required specifications -->
	<physics name='default_physics' default='0' type='ode'>
		<gravity>0 0 -9.8066</gravity>
		<ode>
			<solver>
				<type>quick</type>
				<iters>10</iters>
				<sor>1.3</sor>
				<use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
			</solver>
			<constraints>
				<cfm>0</cfm>
				<erp>0.2</erp>
				<contact_max_correcting_vel>100</contact_max_correcting_vel>
				<contact_surface_layer>0.001</contact_surface_layer>
			</constraints>
		</ode>
		<max_step_size>0.004</max_step_size>
		<real_time_factor>1</real_time_factor>
		<real_time_update_rate>250</real_time_update_rate>
		<magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
	</physics>

  </world>
</sdf>
'''

# Write the content to the generatedWorld.world file
with open('../worlds/generatedWorld.world', 'w') as file:
    file.write(xml_content)
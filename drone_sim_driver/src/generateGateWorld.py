import math

EXTRA_ROTATION = math.pi

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
     
     
  
# Random poses
poses = [
    (5, 0, 0, 0, 0, 0),
    (12, 30, 0, 0, 0, 0),
    (9, 12, 0, 0, 0, 0),
    (32, 13, 0, 0, 0, 0),
    (25, -9, 0, 0, 0, 2),
    (27, -25, 0, 0, 0, 1.8),
    (7, -26, 0, 0, 0, 1),
]

# Center of the circle
radius = 10
num_gates = 20

# Adds the circle
poses.extend(add_circle(-24, 0.17, radius, num_gates, 2* math.pi))
poses.extend(add_circle(-24, -40.17, radius*1.5, num_gates/3, math.pi))

# Adds a line
poses.extend(add_line(-45, 20,70, 25, -math.pi/2))


xml_template = '''
<include>
    <uri>model://aruco_gate_1</uri>
    <name>aruco_gate_{}</name>
    <pose>{}</pose>
</include>
'''

# Generate XML for each pose
xml_instances = [xml_template.format(i, ' '.join(map(str, pose))) for i, pose in enumerate(poses, start=1)]

# Generate the final XML content
xml_content = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="ocean">

    <include>
      <uri>model://ground_plane</uri>
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

# Write the content to the ocean.world file
with open('../worlds/ocean.world', 'w') as file:
    file.write(xml_content)
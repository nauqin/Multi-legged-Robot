# make_robot.py
# Running this script will generate 'hugo_hexapod.urdf' in the same directory.

def generate_urdf():
    # Robot Base Configuration (Body Size: 1.6m x 0.8m x 0.2m)
    urdf = """<?xml version="1.0"?>
<robot name="hugo_hexapod">
  <material name="yellow"><color rgba="0.8 0.7 0.1 1.0"/></material>
  <material name="grey"><color rgba="0.6 0.6 0.6 1.0"/></material>
  <material name="dark_grey"><color rgba="0.3 0.3 0.3 1.0"/></material>
  <material name="red"><color rgba="0.8 0.2 0.2 1.0"/></material>

  <link name="base_link">
    <visual>
      <geometry><box size="1.6 0.8 0.2"/></geometry>
      <material name="yellow"/>
    </visual>
    <collision>
      <geometry><box size="1.6 0.8 0.2"/></geometry>
    </collision>
    <inertial>
      <mass value="40.0"/>
      <inertia ixx="8.6" ixy="0" ixz="0" iyy="2.2" iyz="0" izz="10.6"/>
    </inertial>
  </link>
"""

    # Leg Template Function (Primitive shapes based on user specifications)
    def make_leg(prefix, x, y):
        return f"""
  <joint name="{prefix}_joint1_roll" type="revolute">
    <parent link="base_link"/>
    <child link="{prefix}_hip_dummy"/>
    <origin xyz="{x} {y} 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.8" upper="0.8" effort="200" velocity="2.0"/>
  </joint>

  <link name="{prefix}_hip_dummy">
    <inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="{prefix}_joint2_pitch" type="revolute">
    <parent link="{prefix}_hip_dummy"/>
    <child link="{prefix}_sphere1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="200" velocity="2.0"/>
  </joint>

  <link name="{prefix}_sphere1">
    <visual><geometry><sphere radius="0.04"/></geometry><material name="grey"/></visual>
    <collision><geometry><sphere radius="0.04"/></geometry></collision>
    <inertial><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="{prefix}_sphere1_to_inside" type="fixed">
    <parent link="{prefix}_sphere1"/>
    <child link="{prefix}_inside"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="{prefix}_inside">
    <visual><origin xyz="0 0 -0.15" rpy="0 0 0"/><geometry><cylinder radius="0.03" length="0.3"/></geometry><material name="dark_grey"/></visual>
    <collision><origin xyz="0 0 -0.15" rpy="0 0 0"/><geometry><cylinder radius="0.03" length="0.3"/></geometry></collision>
    <inertial><mass value="2.0"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="{prefix}_prismatic1" type="prismatic">
    <parent link="{prefix}_inside"/>
    <child link="{prefix}_outside"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="0.0" upper="0.2" effort="500" velocity="1.0"/>
  </joint>

  <link name="{prefix}_outside">
    <visual><origin xyz="0 0 -0.175" rpy="0 0 0"/><geometry><cylinder radius="0.04" length="0.35"/></geometry><material name="grey"/></visual>
    <collision><origin xyz="0 0 -0.175" rpy="0 0 0"/><geometry><cylinder radius="0.04" length="0.35"/></geometry></collision>
    <inertial><mass value="3.0"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.002"/></inertial>
  </link>

  <joint name="{prefix}_outside_to_sphere2" type="fixed">
    <parent link="{prefix}_outside"/>
    <child link="{prefix}_sphere2_base"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
  </joint>

  <link name="{prefix}_sphere2_base">
    <inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="{prefix}_joint3_pitch" type="revolute">
    <parent link="{prefix}_sphere2_base"/>
    <child link="{prefix}_sphere2"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="200" velocity="2.0"/>
  </joint>

  <link name="{prefix}_sphere2">
    <visual><geometry><sphere radius="0.04"/></geometry><material name="grey"/></visual>
    <collision><geometry><sphere radius="0.04"/></geometry></collision>
    <inertial><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="{prefix}_sphere2_to_inside2" type="fixed">
    <parent link="{prefix}_sphere2"/>
    <child link="{prefix}_inside2"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="{prefix}_inside2">
    <visual><origin xyz="0 0 -0.15" rpy="0 0 0"/><geometry><cylinder radius="0.03" length="0.3"/></geometry><material name="dark_grey"/></visual>
    <collision><origin xyz="0 0 -0.15" rpy="0 0 0"/><geometry><cylinder radius="0.03" length="0.3"/></geometry></collision>
    <inertial><mass value="2.0"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="{prefix}_prismatic2" type="prismatic">
    <parent link="{prefix}_inside2"/>
    <child link="{prefix}_outside2"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="0.0" upper="0.2" effort="500" velocity="1.0"/>
  </joint>

  <link name="{prefix}_outside2">
    <visual><origin xyz="0 0 -0.175" rpy="0 0 0"/><geometry><cylinder radius="0.04" length="0.35"/></geometry><material name="grey"/></visual>
    <collision><origin xyz="0 0 -0.175" rpy="0 0 0"/><geometry><cylinder radius="0.04" length="0.35"/></geometry></collision>
    <inertial><mass value="3.0"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.002"/></inertial>
  </link>

  <joint name="{prefix}_outside2_to_feet" type="fixed">
    <parent link="{prefix}_outside2"/>
    <child link="{prefix}_feet"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
  </joint>

  <link name="{prefix}_feet">
    <visual><origin xyz="0 0 -0.01" rpy="0 0 0"/><geometry><cylinder radius="0.03" length="0.02"/></geometry><material name="red"/></visual>
    <collision><origin xyz="0 0 -0.01" rpy="0 0 0"/><geometry><cylinder radius="0.03" length="0.02"/></geometry></collision>
    <inertial><mass value="0.5"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  </link>
"""

    # Y-axis Offset (Left/Right ends of the 800mm wide chassis)
    Y_LEFT = 0.4
    Y_RIGHT = -0.4

    # X-axis Positions (Chassis length 1600mm -> spans from -0.8m to +0.8m)
    # 20cm (0.2m) inward from both front and rear ends, and exactly at the center (0.0m)
    x_positions = [0.6, 0.0, -0.6]

    # Generate 3 Left Legs (L1: Front, L2: Middle, L3: Rear)
    for i, x in enumerate(x_positions):
        urdf += make_leg(f"L{i+1}", x, Y_LEFT)

    # Generate 3 Right Legs (R1: Front, R2: Middle, R3: Rear)
    for i, x in enumerate(x_positions):
        urdf += make_leg(f"R{i+1}", x, Y_RIGHT)

    urdf += "</robot>\n"

    # Save the output file with explicit utf-8 encoding
    with open("hugo_hexapod.urdf", "w", encoding="utf-8") as f:
        f.write(urdf)
    
    # Removed Korean characters from print statement to avoid UnicodeEncodeError in non-utf8 terminals
    print("SUCCESS: 'hugo_hexapod.urdf' file has been successfully generated!")

if __name__ == "__main__":
    generate_urdf()
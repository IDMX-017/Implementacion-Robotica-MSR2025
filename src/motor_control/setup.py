from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'motor_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.[yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mario Martinez',
    maintainer_email='mario.mtz@manchester-robotics.com',
    description='This package generates a DC motor simulation, Set Point Generator and Controller nodes for simulating and controlling a DC Motor',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dc_motor = motor_control.dc_motor:main',
            'set_point = motor_control.set_point:main',
	        'ctrl = motor_control.controller:main',
            'trajectory = motor_control.trajectory:main', 
            'inverse_kinematics = motor_control.inverse_kinematics:main', 
            'odometry = motor_control.odometry:main', 
            'cmd_vel_mux = motor_control.cmd_vel_mux:main',
            'teleop_twist_keyboard = motor_control.teleop_twist_keyboard:main', 
            'escaneo_3d = motor_control.escaneo_3d_node:main'
        ],
    },
)

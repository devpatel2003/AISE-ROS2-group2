from setuptools import find_packages, setup

package_name = 'pybullet_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devpatel',
    maintainer_email='devpatel@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'pybullet_lidar_sim = pybullet_sim.pybullet_lidar_sim:main',
            'view_gym = pybullet_sim.view_gym:main',
        ],
    },
)

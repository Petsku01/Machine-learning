# Battery Monitor ROS 2 Package

# https://docs.ros.org/en/jazzy/Installation.html
# Not https://docs.ros.org/en/kilted/Installation.html

## Overview
This ROS 2 package (`battery_monitor`) demonstrates the integration of ROS 2 with machine learning for battery level prediction in a robotic systems. It consists of two nodes:
- **Battery Sensor Node**: Publishes simulated sensor data (current and voltage) to the `/battery_sensors` topic at 1 Hz, mimicking a 12V battery.
- **Battery Predictor Node**: Configures to `/battery_sensors`, uses a Linear Regression model to predict the battery percentage, and publishes the result to `/battery_level` topic.

The package uses `scikit-learn` for the ML model and is designed to run in a ROS 2 Jazzy Jalisco environment.

## Prerequisites
- **OS**: Linux Ubuntu 24.04 (Noble Numbat) or compatible.
- **ROS 2**: ROS 2 Jazzy Jalisco (install instructions: [ROS 2 Documentation](https://docs.ros.org/en/jazzy/Installation.html)).
- **Python**: Python 3.8+ (included with ROS 24.04).
- **Dependencies**: `scikit-learn`, `numpy`.
- **ROS 2 Workspace**: A configured ROS 2 workspace (e.g., `~/ros2_ws`).

## Installation
1. **Create a ROS 2 Workspace** (if not already created):
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```
2. **Clone or Create the Package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python battery_monitor
   ```
3. **Add the Code**:
   - Copy the provided `battery_predictor.py` to `battery_monitor/battery_monitor/`.

4. **Update `setup.py`**:
   - Ensure the `setup.py` in `battery_monitor/` includes the following `entry_points`:
     ```python
     entry_points={
         'console_scripts': [
             'battery_predictor = battery_monitor.battery_predictor:main',
         ],
     }
     ```
   - Full `setup.py` example:
     ```python
     from setuptools import setup

     package_name = 'battery_monitor'

     setup(
         name=package_name,
         version='0.0.1',
         packages=[package_name],
         data_files=[
             ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
             ('share/' + package_name, ['package.xml']),
         ],
         install_requires=['setuptools'],
         zip_safe=True,
         maintainer='Your Name',
         maintainer_email='your.email@example.com',
         description='ROS 2 battery level prediction',
         license='Apache License 2.0',
         tests_require=['pytest'],
         entry_points={
             'console_scripts': [
                 'battery_predictor = battery_monitor.battery_predictor:main',
             ],
         },
     )
     ```
5. **Install Python Dependencies**:
   ```bash
   pip install scikit-learn numpy
   ```
6. **Build the Workspace**:
   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

## Usage
1. **Run the Nodes**:
   ```bash
   ros2 run battery_monitor battery_predictor
   ```
   - Optionally, set a custom timeout threshold (default: 10 seconds):
     ```bash
     ros2 run battery_monitor battery_predictor --ros-args -p timeout_threshold:=15.0
     ```
2. **Monitor Topics**:
   - View sensor data:
     ```bash
     ros2 topic echo /battery_sensors
     ```
   - View predicted battery level:
     ```bash
     ros2 topic echo /battery_level
     ```
3. **Expected Output**:
   - Sensor node: `Published sensor data: Current=2.34A, Voltage=11.8V`
   - Predictor node: `Predicted battery level: 65.2%`
   - Warnings (if applicable):
     - `Out-of-range sensor data: Current=6.00A, Voltage=11.0V`
     - `No valid sensor data received for 10.0 seconds`
     - `Invalid timeout_threshold -5.0, setting to 1.0`

## Code Details
- **File**: `battery_monitor/battery_monitor/battery_predictor.py`
- **Nodes**:
  - `BatterySensorNode`: Publishes `std_msgs/Float32MultiArray` (current, voltage) at 1 Hz.
  - `BatteryPredictorNode`: Subscribes to `/battery_sensors`, uses a Linear Regression model to predict battery percentage, and publishes `std_msgs/Float32` to `/battery_level`.
- **ML Model**: Trained on synthetic data mapping [current, voltage] to battery percentage.
- **Features**:
  - Input validation for sensor data.
  - Configurable timeout for detecting missing sensor data.
  - Robust/fixed error handling for ML training and prediction.

## Extending the Project (future?)
- **Real Sensors**: Replace `BatterySensorNode` with a node reading from a battery management system.
- **Advanced ML**: Use a neural network (e.g., TensorFlow) for more complex predictions.
- **Visualization**: Integrate with RViz to display battery levels.
- **Parameters**: Add ROS 2 parameters for sensor ranges or timer periods.

## Troubleshooting
- **No output on topics**: Ensure nodes are running and the workspace is sourced (`source ~/ros2_ws/install/setup.bash`).
- **Dependency errors**: Verify `scikit-learn` and `numpy` are installed (`pip show scikit-learn numpy`).
- **Invalid data warnings**: Check publisher data format or adjust range checks in `sensor_callback`.
- **Timeout warnings**: Ensure the sensor node is publishing or check `timeout_threshold`.

## License
Apache License 2.0


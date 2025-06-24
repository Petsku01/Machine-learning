import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import time

class BatterySensorNode(Node):
    # Initialize the sensor node for publishing simulated battery sensor data
    def __init__(self):
        # Call parent class constructor with node name
        super().__init__('battery_sensor_node')
        # Define QoS profile for reliable communication
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # Create publisher for sensor data (current, voltage) on 'battery_sensors' topic
        self.publisher_ = self.create_publisher(Float32MultiArray, 'battery_sensors', qos)
        # Create a timer to publish data every 1 second
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Battery sensor node initialized (publishing at 1 Hz)')

    # Callback function for timer, publishes simulated sensor data
    def timer_callback(self):
        try:
            # Create a Float32MultiArray message for sensor data
            msg = Float32MultiArray()
            # Simulate current (0.5-5.0 A) and voltage (10.0-12.6 V for a 12V battery)
            msg.data = [random.uniform(0.5, 5.0), random.uniform(10.0, 12.6)]
            # Publish the sensor data
            self.publisher_.publish(msg)
            # Log the published data for debugging
            self.get_logger().info(f'Published sensor data: Current={msg.data[0]:.2f}A, Voltage={msg.data[1]:.2f}V')
        except Exception as e:
            # Log any errors during publishing
            self.get_logger().error(f'Timer callback error: {str(e)}')

class BatteryPredictorNode(Node):
    # Initialize the predictor node for ML-based battery level prediction
    def __init__(self):
        # Call parent class constructor with node name
        super().__init__('battery_predictor_node')
        # Define QoS profile for reliable communication
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # Declare parameter for timeout threshold (default: 10 seconds)
        self.declare_parameter('timeout_threshold', 10.0)
        # Validate timeout threshold
        timeout_threshold = self.get_parameter('timeout_threshold').get_parameter_value().double_value
        if timeout_threshold <= 0.0:
            self.get_logger().warn(f'Invalid timeout_threshold {timeout_threshold}, setting to 1.0')
            timeout_threshold = 1.0
        # Create subscription to 'battery_sensors' topic
        self.subscription = self.create_subscription(
            Float32MultiArray, 'battery_sensors', self.sensor_callback, qos)
        # Create publisher for predicted battery level on 'battery_level' topic
        self.publisher_ = self.create_publisher(Float32, 'battery_level', qos)
        # Initialize last message time for timeout detection
        self.last_message_time = time.time()
        # Create timer to check for message timeout (period: timeout_threshold / 2, min 0.5s)
        timer_period = max(0.5, timeout_threshold / 2.0)
        self.timeout_timer = self.create_timer(timer_period, self.timeout_callback)
        # Store timeout threshold
        self.timeout_threshold = timeout_threshold
        # Train the ML model during initialization
        self.model = self.train_model()
        self.get_logger().info('Battery predictor node initialized (LinearRegression model ready, timeout_threshold={:.1f}s)'.format(timeout_threshold))

    # Train a Linear Regression model with synthetic data
    def train_model(self):
        try:
            # Synthetic training data: [current, voltage] -> battery percentage (0-100)
            X = np.array([
                [0.5, 12.6], [1.0, 12.4], [2.0, 12.0],  # High battery
                [3.0, 11.5], [4.0, 11.0], [5.0, 10.5]   # Low battery
            ])
            y = np.array([90.0, 80.0, 70.0, 50.0, 30.0, 10.0])
            # Validate input data shapes
            if X.shape[0] != y.shape[0] or X.shape[1] != 2:
                raise ValueError(f'Invalid training data shapes: X={X.shape}, y={y.shape}')
            # Initialize and train the Linear Regression model
            model = LinearRegression()
            model.fit(X, y)
            # Verify model training by checking coefficients
            if not hasattr(model, 'coef_'):
                raise RuntimeError('Model training failed: no coefficients found')
            return model
        except Exception as e:
            # Log and re-raise training errors
            self.get_logger().error(f'Model training error: {str(e)}')
            raise

    # Callback function for processing incoming sensor data
    def sensor_callback(self, msg):
        # Check if the sensor data has the expected length (2 values: current, voltage)
        if len(msg.data) != 2:
            self.get_logger().error(f'Invalid sensor data format: expected 2 values, got {len(msg.data)}')
            return
        
        try:
            # Validate sensor data ranges
            current, voltage = msg.data
            if not (0.5 <= current <= 5.0 and 10.0 <= voltage <= 12.6):
                self.get_logger().warn(f'Out-of-range sensor data: Current={current:.2f}A, Voltage={voltage:.2f}V')
                return
            
            # Prepare sensor data for ML prediction
            data = np.array([msg.data]).reshape(1, -1)
            # Predict battery percentage using the trained model
            prediction = self.model.predict(data)[0]
            # Clamp prediction to valid range (0-100%)
            prediction = max(0.0, min(100.0, prediction))
            
            # Create and publish the prediction result
            result_msg = Float32()
            result_msg.data = float(prediction)
            self.publisher_.publish(result_msg)
            # Log the prediction for debugging
            self.get_logger().info(f'Predicted battery level: {result_msg.data:.1f}%')
            # Update last message time after successful processing
            self.last_message_time = time.time()
        except Exception as e:
            # Log any errors during prediction
            self.get_logger().error(f'Prediction error: {str(e)}')

    # Callback to check for message timeout
    def timeout_callback(self):
        # Warn if no valid sensor data received for the specified threshold
        if time.time() - self.last_message_time > self.timeout_threshold:
            self.get_logger().warn(f'No valid sensor data received for {self.timeout_threshold:.1f} seconds')

def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)
    
    # Create instances of both nodes
    sensor_node = BatterySensorNode()
    predictor_node = BatteryPredictorNode()
    
    # Use a multi-threaded executor to run both nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(sensor_node)
    executor.add_node(predictor_node)
    
    try:
        # Spin the executor to process callbacks
        executor.spin()
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        pass
    finally:
        # Clean up nodes and shutdown ROS 2 if initialized
        executor.shutdown()
        sensor_node.destroy_node()
        predictor_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
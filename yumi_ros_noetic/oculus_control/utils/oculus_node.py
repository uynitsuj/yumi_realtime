from oculus_reader import OculusReader
from tf.transformations import quaternion_from_matrix
import rospy
import tf2_ros
import geometry_msgs.msg
from vr_policy.msg import OculusData


def publish_transform(transform, name):
    translation = transform[:3, 3]

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'world'
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = quaternion_from_matrix(transform)
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    br.sendTransform(t)


def main():
    oculus_reader = OculusReader()
    rospy.init_node('oculus_reader')

    # Publisher for OculusData message
    oculus_publisher = rospy.Publisher('/oculus_reader/data', OculusData, queue_size=10)
    
    import time
    start = time.time()
    while not rospy.is_shutdown():
        rospy.sleep(1/120)
        transformations, buttons = oculus_reader.get_transformations_and_buttons()

        # Create an OculusData message
        oculus_data_msg = OculusData()

        if 'r' in transformations:
            right_controller_pose = transformations['r']
            oculus_data_msg.right_controller_transform.header.stamp = rospy.Time.now()
            oculus_data_msg.right_controller_transform.header.frame_id = 'world'
            oculus_data_msg.right_controller_transform.transform.translation.x = right_controller_pose[0, 3]
            oculus_data_msg.right_controller_transform.transform.translation.y = right_controller_pose[1, 3]
            oculus_data_msg.right_controller_transform.transform.translation.z = right_controller_pose[2, 3]
            quat = quaternion_from_matrix(right_controller_pose)
            oculus_data_msg.right_controller_transform.transform.rotation.x = quat[0]
            oculus_data_msg.right_controller_transform.transform.rotation.y = quat[1]
            oculus_data_msg.right_controller_transform.transform.rotation.z = quat[2]
            oculus_data_msg.right_controller_transform.transform.rotation.w = quat[3]
            oculus_data_msg.right_controller_present = True
            publish_transform(right_controller_pose, 'oculus_r')
        else:
            oculus_data_msg.right_controller_present = False

        if 'l' in transformations:
            left_controller_pose = transformations['l']
            oculus_data_msg.left_controller_transform.header.stamp = rospy.Time.now()
            oculus_data_msg.left_controller_transform.header.frame_id = 'world'
            oculus_data_msg.left_controller_transform.transform.translation.x = left_controller_pose[0, 3]
            oculus_data_msg.left_controller_transform.transform.translation.y = left_controller_pose[1, 3]
            oculus_data_msg.left_controller_transform.transform.translation.z = left_controller_pose[2, 3]
            quat = quaternion_from_matrix(left_controller_pose)
            oculus_data_msg.left_controller_transform.transform.rotation.x = quat[0]
            oculus_data_msg.left_controller_transform.transform.rotation.y = quat[1]
            oculus_data_msg.left_controller_transform.transform.rotation.z = quat[2]
            oculus_data_msg.left_controller_transform.transform.rotation.w = quat[3]
            oculus_data_msg.left_controller_present = True
            publish_transform(left_controller_pose, 'oculus_l')
        else:
            oculus_data_msg.left_controller_present = False

        # Set button states
        # Set button states
        oculus_data_msg.A = buttons.get('A', False)
        oculus_data_msg.B = buttons.get('B', False)
        oculus_data_msg.X = buttons.get('X', False)
        oculus_data_msg.Y = buttons.get('Y', False)
        oculus_data_msg.RThU = buttons.get('RThU', False)
        oculus_data_msg.LThU = buttons.get('LThU', False)
        oculus_data_msg.RJ = buttons.get('RJ', False)
        oculus_data_msg.LJ = buttons.get('LJ', False)
        oculus_data_msg.RG = buttons.get('RG', False)
        oculus_data_msg.LG = buttons.get('LG', False)
        oculus_data_msg.RTr = buttons.get('RTr', False)
        oculus_data_msg.LTr = buttons.get('LTr', False)

        # Set joystick and trigger values if available
        right_joystick = buttons.get('rightJS', (0.0, 0.0))
        oculus_data_msg.right_joystick_x = right_joystick[0]
        oculus_data_msg.right_joystick_y = right_joystick[1]

        left_joystick = buttons.get('leftJS', (0.0, 0.0))
        oculus_data_msg.left_joystick_x = left_joystick[0]
        oculus_data_msg.left_joystick_y = left_joystick[1]

        oculus_data_msg.right_grip = buttons.get('rightGrip', (0.0,))[0]
        oculus_data_msg.left_grip = buttons.get('leftGrip', (0.0,))[0]
        oculus_data_msg.right_trigger = buttons.get('rightTrig', (0.0,))[0]
        oculus_data_msg.left_trigger = buttons.get('leftTrig', (0.0,))[0]

        # Publish the OculusData message
        oculus_publisher.publish(oculus_data_msg)

        # Print transformations and buttons for debugging
        print("freq: ", 1/(time.time() - start))
        start = time.time()
        print('transformations', transformations)
        print('buttons', buttons)


if __name__ == '__main__':
    main()

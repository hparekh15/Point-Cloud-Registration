#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "ros/ros.h"
#include <Eigen/Dense>

class PositionInterface
{
    private:
        ros::NodeHandle n;
        ros::Publisher des_twist_pub;
        ros::Subscriber task_state_sub;
        ros::Publisher des_pos_pub;
        ros::Timer timer;

        Eigen::Matrix<double, 3, 3> matA;
        Eigen::Matrix<double, 3, 1> matB;

        Eigen::Vector3d curr_pos;

        // std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
        // std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        int count;

        int time_count;
    
    public:
        PositionInterface(ros::NodeHandle nh):
        timer(nh.createTimer(ros::Duration(0.01), &PositionInterface::main_loop, this))
        {
            // des_twist_pub = n.advertise<geometry_msgs::Twist>("/passiveDS/desired_twist", 1);
            // task_state_sub = n.subscribe<geometry_msgs::Pose>("/franka_state_controller/ee_pose", 1, boost::bind(&PositionInterface::updateTaskState,this,_1), ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());

            des_pos_pub = n.advertise<geometry_msgs::PoseStamped>("/cartesian_impedance_controller/desired_pose", 1);
            // tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
            // transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            matA << -1.0,  0,  0,
                    0, -1.0,  0,
                    0,  0, -1.0;

            matB << 0.45,
                    0.0,
                    0.4;

            count = 0;
            time_count = 1;
        }


        // void updateTaskState(const geometry_msgs::Pose::ConstPtr& msg){
        //     curr_pos << (double)msg->position.x, (double)msg->position.y, (double)msg->position.z;
        // }

        // void updateDesiredState(const geometry_msgs::Pose::ConstPtr& msg){
        //     matB << (double)msg->position.x,
        //             (double)msg->position.y,
        //             (double)msg->position.z;
        // }
        void publishDesiredFixedPose(){
            geometry_msgs::PoseStamped poseStamped;
            poseStamped.header.stamp = ros::Time::now();

            geometry_msgs::Pose pose;



            if (count == 0){
                pose.position.x = matB(0);
                pose.position.y = matB(1) + 0.3;
                pose.position.z = matB(2);
                pose.orientation.x = 0.925;
                pose.orientation.y = 0.0;
                pose.orientation.z = 0.0;
                pose.orientation.w = 0.381;
            }else if (count == 1){
                pose.position.x = matB(0);
                pose.position.y = matB(1);
                pose.position.z = matB(2);
                pose.orientation.x = 1.0;
                pose.orientation.y = 0.0;
                pose.orientation.z = 0.0;
                pose.orientation.w = 0.0;
            }else{
                pose.position.x = matB(0);
                pose.position.y = matB(1) - 0.3;
                pose.position.z = matB(2);
                pose.orientation.x = -0.925;
                pose.orientation.y = 0.0;
                pose.orientation.z = 0.0;
                pose.orientation.w = 0.381;
            }

            poseStamped.pose = pose;
            des_pos_pub.publish(poseStamped);
        }


        void main_loop(const ros::TimerEvent &){
            publishDesiredFixedPose();
            

            if (time_count % 1000 == 0){
                count++;
                count = count % 3;

                std::cout << count << std::endl;
            }

            time_count++;
        }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "PositionInterface");
    ros::NodeHandle n;


    PositionInterface ds = PositionInterface(n);
    ros::spin();
    return 0;
}
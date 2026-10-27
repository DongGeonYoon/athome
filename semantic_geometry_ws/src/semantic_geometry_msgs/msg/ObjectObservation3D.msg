std_msgs/Header header
int32 detection_id
int32 object_id
string label
float32 confidence
sensor_msgs/PointCloud2 points
geometry_msgs/Point centroid
geometry_msgs/Vector3 size
float32 yaw
float32[] clip_feature
uint32 num_observations
# bbox_3d.center of the Scene Graph JSON (differs from centroid = point mean); size / yaw above
geometry_msgs/Point bbox_center

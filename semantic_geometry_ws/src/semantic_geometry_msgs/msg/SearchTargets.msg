# /athome/search/targets (latched, transient_local). Target object names of the current command; an empty
# list ends the search. Perception adds the names to the detector query and prefers them in the label vote.
std_msgs/Header header
string command_id
string[] targets

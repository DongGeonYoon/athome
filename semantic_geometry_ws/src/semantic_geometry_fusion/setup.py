from setuptools import find_packages, setup

package_name = "semantic_geometry_fusion"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/fusion_params.yaml", "config/robot.yaml", "config/fastdds_shm.xml"]),
        ("share/" + package_name + "/launch", ["launch/perception.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="LiDAR geometry and CG-D CLIP feature fusion without DBSCAN/clustering.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "geometry_fusion_node = semantic_geometry_fusion.geometry_fusion_node:main",
            "gsa_pkl_bridge = semantic_geometry_fusion.gsa_pkl_bridge:main",
            "inspect_gsa = semantic_geometry_fusion.inspect_gsa:main",
            "synthetic_test = semantic_geometry_fusion.synthetic_test:main",
        ],
    },
)

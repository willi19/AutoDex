import os
import numpy as np

from autodex.planner import PlanResult


class SimExecutor:
    """
    Visualize planned trajectory in viser viewer.

    Usage:
        executor = SimExecutor()
        executor.visualize(plan_result, obj_name="bottle", obj_pose=T)
    """

    def visualize(self, plan_result: PlanResult, obj_name: str, obj_pose: np.ndarray):
        """Show planned trajectory in viser viewer."""
        import trimesh
        from paradex.visualization.visualizer.viser import ViserViewer
        from rsslib.path import urdf_path, obj_path

        if not plan_result.success:
            print("Planning failed — nothing to visualize.")
            return

        viewer = ViserViewer()

        # robot
        urdf_full = os.path.join(urdf_path, "xarm_allegro.urdf")
        viewer.add_robot("robot", urdf_full)

        # object mesh (static)
        mesh_path = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
        mesh = trimesh.load(mesh_path)
        viewer.add_trimesh("object", mesh, obj_pose)

        # scene
        viewer.add_grid()

        # trajectory: robot moves, object stays static
        traj = plan_result.traj  # (T, 22)
        obj_traj = np.tile(obj_pose[None], (len(traj), 1, 1))  # (T, 4, 4)

        viewer.add_traj(
            "plan",
            robot_traj={"robot": traj},
            obj_traj={"object": obj_traj},
        )
        viewer.start_viewer()
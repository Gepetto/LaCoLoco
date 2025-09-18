import os
import sys
import copy
import glob
import pickle
import bvhsdk
import hppfcl
import numpy as np
import casadi as ca
from tqdm import tqdm
import pinocchio as pin
import pinocchio.visualize
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pinocchio import casadi as cpin
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

############################################
MULTIPROCESSING = False
VISUALIZE = True
############################################


# Quaternion in xyzw form to MRP:
def quat2mrp(xyzw: ca.SX) -> ca.SX:
    normalized = xyzw / ca.norm_2(xyzw)
    return normalized[:3] / (1 + normalized[3])


# MRP to quaternion in xyzw form:
def mrp2quat(xyz: ca.SX) -> ca.SX:
    normsq = xyz.T @ xyz
    w = (ca.SX.ones(1) - normsq) / (1 + normsq)
    return ca.vertcat(2 * xyz / (1 + normsq), w)


# State using MRP for base orientation to state with quaternion.
# We assume the floating base is at offsets [0:6]: x, y, z, mrp
def q_mrp_to_quat(q_mrp: ca.SX) -> ca.SX:
    return ca.vertcat(q_mrp[:3], mrp2quat(q_mrp[3:6]), q_mrp[6:])


# State using quaternion for base orientation to state with MRP.
# We assume the floating base is at offsets [0:7]: x, y, z, xyzw
def q_quat_to_mrp(q_quat: ca.SX) -> ca.SX:
    return ca.vertcat(q_quat[:3], quat2mrp(q_quat[3:7]), q_quat[7:])


# Daft thing converting a CasADi SX to a numpy array.
# For some reason, np.array(SX) doesn't work? Neither .full()?
def ca_to_np(x: ca.SX) -> np.ndarray:
    result = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j]

    return result


class Solo12:
    def _load_robot(self):
        pkg_path = os.path.join("./Projects/trajopt/")
        urdf_path = "./Projects/trajopt/example-robot-data/robots/solo_description/robots/solo12.urdf"

        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[pkg_path], root_joint=pin.JointModelFreeFlyer()
        )

    def _create_visualizer(self, floor_z: float = 0.0):
        self.visualizer = pin.visualize.MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )

        self.robot.setVisualizer(self.visualizer)
        self.robot.initViewer()
        self.robot.loadViewerModel()

        # # Add floor visual geometry:
        # floor_obj = pin.GeometryObject("floor", 0, 0, hppfcl.Box(2, 2, 0.005), pin.SE3.Identity())
        # self.visualizer.loadViewerGeometryObject(floor_obj, pin.GeometryType.VISUAL, np.array([0.3, 0.3, 0.3, 1]))

        # # Manually set the floor transform because the GeometryObject() constructor doesn't work:
        # floor_obj_name = self.visualizer.getViewerNodeName(floor_obj, pin.GeometryType.VISUAL)
        # self.visualizer.viewer[floor_obj_name].set_transform(
        #     pin.SE3(np.eye(3), np.array([0, 0, floor_z])).homogeneous
        # )

        # Display an initial pose:
        self.robot.display(pin.neutral(self.robot.model))

    def __init__(self, floor_z: float = -0.226274, visualize: bool = False):
        self._load_robot()

        if visualize:
            self._create_visualizer(floor_z)

        # Create autodiff robot model:
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata = self.cmodel.createData()

        # self.frames = {
        #     "feet":      ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"],
        #     "shoulders": ["FR_SHOULDER", "FL_SHOULDER", "HR_SHOULDER", "HL_SHOULDER"],
        #     "knees":     ["FR_LOWER_LEG", "FL_LOWER_LEG", "HR_LOWER_LEG", "HL_LOWER_LEG"],
        # }

        # # Skip 'universe' and 'root_joint' as they're not actuated:
        # self.actuated_joints = [j.id for j in self.robot.model.joints[2:]]

    def q_off(self, joint: str):
        return 6 + list(self.robot.model.names)[2:].index(joint)


class ADFrameKinematics:
    def __init__(self, robot: Solo12, frames=None):
        self.cmodel, self.cdata = robot.cmodel, robot.cdata

        self.frame_ids = [
            self.cmodel.getFrameId(frame)
            for frame in (frames if frames is not None else FRAMES.values())
        ]

    def __call__(self, q_mrp: ca.SX):
        q = q_mrp_to_quat(q_mrp)
        cpin.forwardKinematics(self.cmodel, self.cdata, q)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        # Get all frame positions (3x1) and concatenate them into a (3xN) matrix.
        # Return its transpose (Nx3).
        return ca.horzcat(
            *iter(self.cdata.oMf[fid].translation for fid in self.frame_ids)
        ).T


FRAMES = OrderedDict(
    [
        ("RightHand", "FR_FOOT"),
        ("LeftHand", "FL_FOOT"),
        ("RightFoot", "HR_FOOT"),
        ("LeftFoot", "HL_FOOT"),
        # ("RightForeArm", "FR_LOWER_LEG"),
        # ("LeftForeArm",  "FL_LOWER_LEG"),
        # ("RightLeg",     "HR_LOWER_LEG"),
        # ("LeftLeg",      "HL_LOWER_LEG"),
        ("RightShoulder", "FR_SHOULDER"),
        ("LeftShoulder", "FL_SHOULDER"),
        ("RightUpLeg", "HR_SHOULDER"),
        ("LeftUpLeg", "HL_SHOULDER"),
        # ("Spine1", "root_joint"),
        # mj.name: ""
        # for mj in mocap.getlistofjoints()
    ]
)

DEFAULT_ANGLES = {
    "FR_KFE": -np.pi / 2,
    "FL_KFE": -np.pi / 2,
    "HR_KFE": -np.pi / 2,
    "HL_KFE": -np.pi / 2,
    "FR_HFE": np.pi / 4,
    "FL_HFE": np.pi / 4,
    "HR_HFE": np.pi / 4,
    "HL_HFE": np.pi / 4,
    "FR_HAA": 0,
    "FL_HAA": 0,
    "HR_HAA": 0,
    "HL_HAA": 0,
}

NVIDIA_JOINT_ORDER = [
    "FL_HAA",
    "FR_HAA",
    "HL_HAA",
    "HR_HAA",
    "FL_HFE",
    "FR_HFE",
    "HL_HFE",
    "HR_HFE",
    "FL_KFE",
    "FR_KFE",
    "HL_KFE",
    "HR_KFE",
]


def rearrange_joints(vec, solo):
    pinocchio_order = list(solo.robot.model.names)[2:]

    return np.array([vec[pinocchio_order.index(j)] for j in NVIDIA_JOINT_ORDER])


JOINT_LIMIT_MIN = {
    "FR_KFE": -ca.inf,
    "FL_KFE": -ca.inf,
    "HR_KFE": -ca.inf,
    "HL_KFE": -ca.inf,
    "FR_HFE": -5 * np.pi / 4,
    "FL_HFE": -5 * np.pi / 4,
    "HR_HFE": -np.pi / 2,
    "HL_HFE": -np.pi / 2,
    "FR_HAA": -np.pi / 2,
    "FL_HAA": -np.pi / 4,
    "HR_HAA": -np.pi / 2,
    "HL_HAA": -np.pi / 4,
}

JOINT_LIMIT_MAX = {
    "FR_KFE": ca.inf,
    "FL_KFE": ca.inf,
    "HR_KFE": ca.inf,
    "HL_KFE": ca.inf,
    "FR_HFE": np.pi / 2,
    "FL_HFE": np.pi / 2,
    "HR_HFE": 5 * np.pi / 4,
    "HL_HFE": 5 * np.pi / 4,
    "FR_HAA": np.pi / 4,
    "FL_HAA": np.pi / 2,
    "HR_HAA": np.pi / 4,
    "HL_HAA": np.pi / 2,
}


def retarget(filename):
    print(f"Processing {filename}...")
    solo = Solo12(visualize=VISUALIZE)

    # input("Press ENTER to start!")

    if VISUALIZE:
        solo.robot.display(pin.neutral(solo.robot.model))

    import code

    code.interact(local=locals())

    frame_corrector = R.from_matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    PATH = f"MotionCapture/{filename}"
    mocap = bvhsdk.ReadFile(PATH)

    global_positions = {mj: [] for mj in FRAMES}

    for mj in FRAMES:
        joint = mocap.getJoint(mj)

        for fidx in range(mocap.frames):
            p_mcoords = joint.getPosition(fidx)
            global_positions[mj].append(frame_corrector.apply(p_mcoords) / 100.0)

    q_def = np.array(q_quat_to_mrp(pin.neutral(solo.robot.model)).T)[0]
    for j, ang in DEFAULT_ANGLES.items():
        q_def[solo.q_off(j)] = ang

    q_sym = ca.SX.sym("q_sym", (18, 1))
    # z_knees = ca.SX.sym("z_knee_sym", (4, 1))
    # scales_sym = ca.SX.sym("scales_sym", (3, 1))

    ad_fk_targets = ADFrameKinematics(solo)
    num_fk_targets = ca.Function("num_fk_targets", [q_sym], [ad_fk_targets(q_sym)])

    ad_fk_knees = ADFrameKinematics(solo, ["FL_KFE", "FR_KFE", "HL_KFE", "HR_KFE"])
    num_fk_knees = ca.Function("num_fk_knees", [q_sym], [ad_fk_knees(q_sym)])

    def errf(q, target_pos):
        actual_pos = num_fk_targets(q)

        error = 0.0
        for f_idx in range(len(target_pos)):
            error += ca.sum((np.array(target_pos[f_idx]) - actual_pos[f_idx, :].T) ** 2)

        return error

    def solve(idx, q0, plot=False):
        # p = [scaled_positions[f][idx] for f in FRAMES.keys()]
        p = [global_positions[f][idx] for f in FRAMES.keys()]

        aa_regularizer = 0.0
        for ang in ["FR_HAA", "FL_HAA", "HR_HAA", "HL_HAA"]:
            aa_regularizer += q_sym[solo.q_off(ang)] ** 2

        # Ipopt doesn't like the c(q) = x, x >= 0 formulation...
        variables = {
            # "x": ca.vertcat(z_knees, q_sym),
            "x": q_sym,
            "f": errf(q_sym, p)
            + 0.005 * ca.sum((q_sym - q_def)[6:] ** 2),  # + 0.06*aa_regularizer,
            "g": num_fk_knees(q_sym)[:, 2],
        }

        prob = ca.nlpsol(
            "S",
            "ipopt",
            variables,
            {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"},
        )

        # Joint limits:
        lbx, ubx = [-ca.inf] * 18, [ca.inf] * 18

        for joint in JOINT_LIMIT_MIN.keys():
            idx = solo.q_off(joint)
            lbx[idx] = JOINT_LIMIT_MIN[joint]
            ubx[idx] = JOINT_LIMIT_MAX[joint]

        soln = prob(
            # x0 = np.concatenate((np.zeros(4), q0)),
            x0=q0,
            lbx=lbx,
            ubx=ubx,
            lbg=[0] * 4,
            ubg=[ca.inf] * 4,
        )

        return soln
        # print(num_fk(soln["x"])[:4, :])

    qs_solver = []
    grounded_states = []

    # for idx in tqdm(range(mocap.frames)):
    for idx in range(mocap.frames):
        soln = solve(idx, q_def if idx == 0 else qs_solver[-1], plot=True)
        # qs_solver.append(ca_to_np(soln["x"][4:]).T[0])
        qs_solver.append(ca_to_np(soln["x"]).T[0])

        # Force the lower foot to be on the ground because of the offset ¯\_(ツ)_/¯
        feet_z = ca_to_np(num_fk_targets(qs_solver[-1])[:4, 2])
        q_g = copy.deepcopy(qs_solver[-1])
        q_g[2] -= min(feet_z)[0]

        grounded_states.append(q_g)

        if VISUALIZE:
            solo.robot.display(ca_to_np(q_mrp_to_quat(grounded_states[-1])).T[0])

    def calculate_observations(qs):
        observations, states = [], []
        dt = mocap.frametime

        for idx in range(len(qs) - 1):
            q_cur = qs[idx]
            q_next = qs[idx + 1]

            # Local frame at q_k (xyzw quaternion):
            # TODO: Interpolate between the two frames
            q_cur_quat = ca_to_np(q_mrp_to_quat(q_cur)).T[0]
            F = R.from_quat(q_cur_quat[3:7], scalar_first=False)
            F_inv = F.inv()

            obs, state = {}, {}

            # --------- Observations -------------
            # Finite differences approximation:
            linvel_g = (q_next[:3] - q_cur[:3]) / dt

            # Local frame measurements:
            obs["base_linvel"] = F_inv.apply(linvel_g)

            # Angular velocity in robot local frame:
            # https://math.stackexchange.com/questions/2282938/converting-from-quaternion-to-angular-velocity-then-back-to-quaternion
            # https://arxiv.org/pdf/0811.2889
            q_next_quat = ca_to_np(q_mrp_to_quat(q_next)).T[0]
            F_next = R.from_quat(q_next_quat[3:7], scalar_first=False)
            obs["base_angvel"] = (
                2 * (F.inv() * F_next).as_quat(scalar_first=False) / dt
            )[:3]

            # Projected normalized gravity in local frame:
            grav_g = [0, 0, -1.0]
            obs["proj_gravity"] = F_inv.apply(grav_g)

            # Base height, average for better approximation:
            obs["base_height"] = (q_cur[2] + q_next[2]) / 2.0

            # Joint positions and velocities:
            obs["joint_pos"] = rearrange_joints(q_cur[6:], solo)
            obs["joint_vel"] = rearrange_joints((q_next[6:] - q_cur[6:]) / dt, solo)
            # ---------------------------------------

            # ------------ States ------------
            state["base_pos"] = (q_cur[:3] + q_next[:3]) / 2.0
            state["base_orientation"] = F.as_quat(scalar_first=True)  # wxyz

            state["base_linvel"] = obs["base_linvel"]
            state["base_angvel"] = obs["base_angvel"]
            state["joint_pos"] = obs["joint_pos"]
            state["joint_vel"] = obs["joint_vel"]
            # --------------------------------

            observations.append(obs)
            states.append(state)

        return observations, states

    o, s = calculate_observations(grounded_states)

    with open(f"output/{filename.rstrip('.bvh')}.pkl", "wb") as wf:
        pickle.dump(
            {
                "observations": o,
                "states": s,
                "qs_mrp_grounded": grounded_states,
                "qs_mrp_solver": qs_solver,
            },
            wf,
        )

    O_FIELDS = [
        "base_height",
        "proj_gravity",
        "base_linvel",
        "base_angvel",
        "joint_pos",
        "joint_vel",
    ]

    obs_matrix = np.vstack(
        [
            np.array(
                [[np.concatenate([np.array(o[t][of], ndmin=1) for of in O_FIELDS])]]
            )
            for t in range(len(o))
        ]
    )

    np.save(f"observations/{filename.rstrip('.bvh')}_obs", obs_matrix)

    S_FIELDS = [
        "base_pos",
        "base_orientation",
        "base_linvel",
        "base_angvel",
        "joint_pos",
        "joint_vel",
    ]

    st_matrix = np.vstack(
        [
            np.array(
                [[np.concatenate([np.array(s[t][sf], ndmin=1) for sf in S_FIELDS])]]
            )
            for t in range(len(s))
        ]
    )

    np.save(f"states/{filename.rstrip('.bvh')}_state", st_matrix)


if __name__ == "__main__":
    # data link: https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018
    # files = [
    #     os.path.basename(path)
    #     for path in glob.glob("MotionCapture/*.bvh")
    # ]

    files = ["D1_031_KAN01_001.bvh"]

    if MULTIPROCESSING:
        with Pool(20) as pool:
            pool.map(retarget, files)
    else:
        for file in files:
            retarget(file)
            # break

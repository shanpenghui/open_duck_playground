# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base

# from playground.common.utils import LowPassActionFilter
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    reward_alive,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

# if set to false, won't require the reference data to be present and won't compute the reference motions polynoms for nothing
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                joint_vel=2.5,  # rad/s # Was 1.5
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=2.5,
                tracking_ang_vel=6.0,
                torques=-1.0e-3,
                action_rate=-0.5,  # was -1.5
                stand_still=-0.2,  # was -1.0 TODO try to relax this a bit ?
                alive=20.0,
                imitation=1.0,
            ),
            tracking_sigma=0.01,  # was working at 0.01
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.0, 1.0],  # [-1.0, 1.0]
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        head_range_factor=1.0,  # to make it easier
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            )

        # Note: First joint is freejoint.
        # get the range of the joints
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # # noise in the simu?
        qpos_noise_scale = np.zeros(self._actuators)

        hip_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j
        ]
        knee_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j
        ]
        ankle_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j
        ]

        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # self.action_filter = LowPassActionFilter(
        #     1 / self._config.ctrl_dt, cutoff_frequency=37.5
        # )

    # 功能简述：
    #
    # 重置环境，包括设置初始位姿、速度、控制命令，以及模仿参考数据和扰动初始化。对应强化学习中的 episode 起始状态构造。
    # reset() 主要逻辑：
    # ┌───────────────────────────────┐
    # │    随机初始化 base pose/vel   │
    # ├───────────────────────────────┤
    # │    采样 joystick 控制命令      │
    # ├───────────────────────────────┤
    # │    初始化 imitation 参考轨迹   │
    # ├───────────────────────────────┤
    # │    构造 info 状态 / 奖励统计   │
    # ├───────────────────────────────┤
    # │    返回 state 用于 RL         │
    # └───────────────────────────────┘
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # 从关键帧“home”中获取初始的完整位置（浮动基 + 关节）
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv) # 初始化所有速度为0（nv为自由度数量）

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        # 在x/y平面添加小的随机偏移 [-0.05, 0.05]，模拟起始位置不确定性
        # 类似 sim2real 的 domain randomization（详见：Tobin et al., 2017. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"）
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # 提取浮动基的qpos，并加上x/y的随机偏移
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        # 添加z轴上的yaw旋转扰动 [-π, π]，并叠加到原始姿态上（四元数乘法）
        # 四元数乘法 q' = q ⊗ Δq 实现旋转扰动叠加。详见《Quaternions and Rotation Sequences》 by J. B. Kuipers。
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw noise

        base_qpos = base_qpos.at[3:7].set(new_quat)

        # 更新完整 qpos 中的浮动基位置
        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        # print(f'DEBUG1 base qpos: {qpos}')
        # init joint position
        # qpos[7:]=*U(0.0, 0.1)
        rng, key = jax.random.split(rng)

        # 给每个实际关节添加随机缩放噪声 [0.5, 1.5]，增强动作多样性
        #📘【应用目的】：
        #   增加 exploration 初期策略的泛化能力。
        # multiply actual joints with noise (excluding floating base and backlash)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
        # print(f'DEBUG2 joint qpos: {qpos}')
        # init joint vel
        # d(xyzrpy)=U(-0.05, 0.05)
        # 给浮动基加速度扰动 [-0.05, 0.05]，模拟初始动态变化
        rng, key = jax.random.split(rng)
        # qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        # print(f'DEBUG3 base qvel: {qvel}')
        # 初始的控制量设置为目标角度值（即关键帧中的actuator默认位姿）
        ctrl = self.get_actuator_joints_qpos(qpos)
        # print(f'DEBUG4 ctrl: {ctrl}')
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

        # 采样一个 joystick 命令：[vx, vy, yaw_rate, neck_pitch, head_pitch, head_yaw, head_roll]
        # 🎮 控制命令仿真目标是：让机器人学会通过模仿轨迹来执行 joystick 提示的方向性行为。
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        # 采样随机推力间隔（用于外部扰动模拟），单位为仿真秒数
        # 📘【现实模拟】：
        #    仿真真实机器人训练中突然遭受撞击（perturbation）的鲁棒性考察。
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        # 如果启用 imitation reward，则从多项式中获取当前参考轨迹（基于命令 + 相位 = 0）
        #📘【论文参考】：
        #    Peng et al. 2021. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], 0
            )
        else:
            current_reference_motion = jp.zeros(0)

        # info 是强化学习中 agent 可见/隐藏状态，包括模仿阶段、历史动作、IMU缓存等
        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            # imitation related
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
        }

        # 奖励项初始化（根据 reward_config 中的 scale 项设定指标种类）
        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        # 检测当前接触状态，返回 [left_contact, right_contact]
        contact = jp.array(
            [
                # 📘 geoms_colliding() 是封装的 collision check 函数，检测几何体是否重合。
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        # 获取观测向量（state 和 privileged_state）
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)  # 初始无奖励，未终止
        # 打包为 mjx_env.State 返回，用于强化学习迭代
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # 主环境步进函数。输入当前状态和 agent 动作，返回新状态。
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        if USE_IMITATION_REWARD:
            # imitation_i 表示当前模仿周期内的步数索引，每步递增
            state.info["imitation_i"] += 1
            # 对步数取模，保证 phase 周期性（一个周期内 imitation_i 在 [0, nb_steps_in_period)）
            state.info["imitation_i"] = (
                    state.info["imitation_i"] % self.PRM.nb_steps_in_period
            )  # 已在 get_reference_motion 内部做了模运算，这里是显式同步

            # 使用 cos/sin 编码将模仿周期位置信息转化为可学习的向量（相当于时间 embedding）
            # 对应 AMP 论文中 imitation phase 表示法：
            # \phi_i = [cos(2π * t_i / T), sin(2π * t_i / T)]
            state.info["imitation_phase"] = jp.array(
                [
                    jp.cos(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                ]
            )
        else:
            # imitation reward 未启用时，phase 和步数重置
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            # 根据 joystick 指令 vx, vy, yaw 和 imitation_i 获取当前参考动作
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0],  # vx
                state.info["command"][1],  # vy
                state.info["command"][2],  # yaw_rate
                state.info["imitation_i"], # 当前模仿帧编号
            )
        else:
            # imitation reward 关闭时返回零参考轨迹
            state.info["current_reference_motion"] = jp.zeros(0)

        # 拆分随机种子，分别用于推力方向、推力强度、动作延迟
        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # 滚动动作历史缓存（FIFO），插入当前动作
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history

        # 从指定动作延迟范围内采样一个索引值
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        # 根据延迟步数从历史队列中取出实际使用的动作（用于模拟控制链条时延）
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]

        # 采样推力方向（θ ∈ [0, 2π]）和推力大小（∈ magnitude_range）
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        # 构造单位向量方向的推力
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        # 仅在推力步数触发点施加推力，否则为零
        push *= (
                jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        # 如果全局开关关闭，则清零推力
        push *= self._config.push_config.enable

        # 将推力添加到浮动基底的 qvel[x,y] 上
        qvel = state.data.qvel
        qvel = qvel.at[
               self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
               ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # 根据动作生成 motor target，action 是归一化输出，因此乘以 action_scale 并偏移至默认位置
        # motor_targets = a_default + a_normalized * α，其中 α 是 self._config.action_scale
        motor_targets = (
                self._default_actuator + action_w_delay * self._config.action_scale
        )

        # 如果启用了电机最大速度限制，则对 motor_targets 做限速裁剪
        # motor_targets_clipped = clip(motor_targets, prev ± v_max * dt)
        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,  # 下限
                prev_motor_targets + self._config.max_motor_velocity * self.dt,  # 上限
            )

        # 用 motor_targets 执行仿真步进，执行 self.n_substeps 个子步
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets  # 存储该步目标值（用于限速）

        # 检测双脚是否与地面接触
        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        # 记录首次落地帧（上一帧非接触，本帧接触）
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt

        # 记录两脚空中持续时间与最高离地高度（swing_peak）
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]  # 所有脚的世界坐标
        p_fz = p_f[..., -1]  # 提取 z 高度
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        # 获取当前观测
        obs = self._get_obs(data, state.info, contact)
        # 判断是否摔倒
        done = self._get_termination(data)

        # 计算所有子奖励项（未加权）
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        # 每项乘以配置中的权重，reward_config.scales[k] 对应每一项 k 的系数
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        # 总奖励为 sum(rewards) * dt，并裁剪最大值，避免不稳定
        # reward_total = clip(Σ_k r_k * dt, 0, 10000)
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # 更新状态记录
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        # 每 500 步重新采样命令（模拟 joystick 变化）
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
            )

        # 达到终止条件或步数上限后清零 step
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
            )

        # 若接触则 feet_air_time 归零；未接触则保持累加
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        # 将奖励/惩罚项记录进 metrics，分 reward 和 cost
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v

        # 记录该帧最大抬腿高度
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        # accelerometer[0] += 1.3 # TODO testing
        accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Handle IMU delay
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        # joint_angles = data.qpos[7:]

        # Handling backlash
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)

        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)

        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        # joint_vel = data.qvel[6:]
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)
        # info["rng"], noise_rng = jax.random.split(info["rng"])
        # noisy_linvel = (
        #     linvel
        #     + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.linvel
        # )

        state = jp.hstack(
            [
                # noisy_linvel,  # 3
                # noisy_gyro,  # 3
                # noisy_gravity,  # 3
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                info["command"],  # 3
                noisy_joint_angles - self._default_actuator,  # 10
                noisy_joint_vel * self._config.dof_vel_scale,  # 10
                info["last_act"],  # 10
                info["last_last_act"],  # 10
                info["last_last_last_act"],  # 10
                info["motor_targets"],  # 10
                contact,  # 2
                # info["current_reference_motion"],
                # info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 10
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    # 内部调用的奖励函数，输出所有 reward 名称对应的值（未加权）
    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        #当前未使用 metrics 参数，避免 PyLint 报错。
        del metrics  # Unused.

        ret = {
            # 奖励机器人以正确的 x/y 平移速度移动。
            # 公式：r = exp( - ||v_target - v_actual||^2 / (2 * σ^2) )
            # 来自论文 DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"], # joystick 命令中的线速度目标 vx, vy
                self.get_local_linvel(data), # 当前身体的本地线速度
                self._config.reward_config.tracking_sigma,  # reward 平滑系数, 高斯惩罚项的σ
            ),

            # 奖励机器人以正确的角速度（转向）移动。
            # 公式：r = exp( - ||ω_target - ω_actual||^2 / (2 * σ^2) )
            # 用于鼓励 agent 原地或行进中转向正确。
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],# joystick 命令中的期望角速度 yaw_rate
                self.get_gyro(data),# 当前陀螺仪角速度（通常绕 z）
                self._config.reward_config.tracking_sigma,
            ),

            # "orientation": cost_orientation(self.get_gravity(data)),
            # 惩罚使用过大的关节力矩，鼓励节能控制。
            # 公式：cost = λ * Σ_i τ_i^2
            # 参考：Tassa et al., "Synthesis and stabilization of complex behaviors" (2012)
            "torques": cost_torques(data.actuator_force),

            # 惩罚连续两步动作差异过大，鼓励控制平滑。
            # 公式：cost = λ * ||a_t - a_{t-1}||^2
            # 可防止振荡、降低硬件磨损、增强 sim2real 鲁棒性。
            "action_rate": cost_action_rate(action, info["last_act"]),

            # 给予基础“存活奖励”，鼓励 agent 不摔倒、持续运动。
            # 通常返回固定常数，如 1 或 20。
            # 若发生摔倒、nan、翻滚等，会在环境 step() 中被置为 done。
            "alive": reward_alive(),

            # imitation 奖励：鼓励 agent 模仿参考动作。
            # 通常包括：
            # - 关节角度误差：Σ ||q_ref - q||^2
            # - 关节速度误差：Σ ||dq_ref - dq||^2
            # - 根部位置/速度差异
            # 可根据 AMP 框架替代 discriminator，直接使用 imitation reward。
            # 参考论文：
            # - AMP: https://arxiv.org/abs/2104.02180
            # - DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
            "imitation": reward_imitation(  # FIXME, this reward is so adhoc...
                self.get_floating_base_qpos(data.qpos),  # 根部位置 + 姿态（四元数）
                self.get_floating_base_qvel(data.qvel),  # 根部速度
                self.get_actuator_joints_qpos(data.qpos),# 所有关节角度
                self.get_actuator_joints_qvel(data.qvel),# 所有关节速度
                contact, # 当前接触状态（用于平衡脚步）
                info["current_reference_motion"],# 多项式拟合生成的参考动作轨迹
                info["command"],# 当前 joystick 控制命令
                USE_IMITATION_REWARD,# imitation 奖励是否启用
            ),

            # 当目标命令为零时，惩罚机器人移动。
            # 用于鼓励 agent 在“静止命令”下保持站立不动。
            # 实现逻辑：若 ||command|| ≈ 0，则对 q, dq 与默认值进行 L2 惩罚
            "stand_still": cost_stand_still(
                # info["command"], data.qpos[7:], data.qvel[6:], self._default_pose
                info["command"],# joystick 命令
                self.get_actuator_joints_qpos(data.qpos), # 当前关节角度
                self.get_actuator_joints_qvel(data.qvel),# 当前关节速度
                self._default_actuator, # 默认静止角度（关键帧“home”姿态）
                ignore_head=False, # 是否忽略头部（这里不忽略）
            ),
        }

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 8)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        neck_pitch = jax.random.uniform(
            rng5,
            minval=self._config.neck_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.neck_pitch_range[1] * self._config.head_range_factor,
        )

        head_pitch = jax.random.uniform(
            rng6,
            minval=self._config.head_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.head_pitch_range[1] * self._config.head_range_factor,
        )

        head_yaw = jax.random.uniform(
            rng7,
            minval=self._config.head_yaw_range[0] * self._config.head_range_factor,
            maxval=self._config.head_yaw_range[1] * self._config.head_range_factor,
        )

        head_roll = jax.random.uniform(
            rng8,
            minval=self._config.head_roll_range[0] * self._config.head_range_factor,
            maxval=self._config.head_roll_range[1] * self._config.head_range_factor,
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(7),
            jp.hstack(
                [
                    lin_vel_x,
                    lin_vel_y,
                    ang_vel_yaw,
                    neck_pitch,
                    head_pitch,
                    head_yaw,
                    head_roll,
                ]
            ),
        )

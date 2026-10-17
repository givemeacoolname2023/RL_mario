import gym

class SkipFrameWrapper(gym.Wrapper):

    def __init__(self, env, skip=4):
        super(SkipFrameWrapper, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        obs, info = None, None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class MarioRewardWrapper(gym.Wrapper):
    """进度型奖励：只奖励"本关新刷新的最远 x"，死亡扣分，举旗加分。

    一局累计奖励 ≈ 0.1 * 本局最远 x - 15 * 死亡次数 + 50 * 举旗次数，
    是"跑了多远"的单调函数。
    死一次后 x 回到关卡开头，但 best_x 不回退 —— 送死刷分拿不到任何分。

    原版奖励是逐帧看「位移」——每一帧只要 x 比上一帧大就加分（跑动 +3）,
    死后 x 重置回关卡开头，同一段路可以重跑一遍再赚一遍，所以「冲上去送死」
    成了最省事的刷分方式（一条命白赚约 +250, 撞死只扣 15）
    """

    def __init__(self, env, progress_scale=0.1, death_penalty=15.0,
                 flag_bonus=50.0, time_penalty=0.0):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.time_penalty = time_penalty
        self._reset_state()

    # 让奖励计算从一局新游戏的干净状态开始，避免上一局的记录串到这一局。
    def _reset_state(self):
        self._best_x = 0
        self._level = None
        self._prev_life = None
        self._flag_awarded = False

    # 新一局开始时，先重置游戏，再清空奖励包装器的记忆，然后把初始观测返回。
    def reset(self):
        obs = self.env.reset()
        self._reset_state()
        return obs

    def step(self, action):
        obs, _reward, done, info = self.env.step(action)

        x = info["x_pos"]
        level = (info["world"], info["stage"])

        # 举旗进入下一关时，进度前沿重新从新关卡开头算
        if level != self._level:
            self._level = level
            self._best_x = x
            self._flag_awarded = False

        # 1) 进度：只算刷新历史最远的正向位移
        progress = max(0, x - self._best_x)
        self._best_x = max(self._best_x, x)
        shaped = self.progress_scale * progress

        # 2) 举旗（上升沿只加一次）
        if info["flag_get"] and not self._flag_awarded:
            shaped += self.flag_bonus
            self._flag_awarded = True

        # 3) 死亡：life 减少；Game Over 时 life 变成 0xFF
        life = info["life"]
        if self._prev_life is not None and (
            life < self._prev_life or (life == 0xFF and self._prev_life != 0xFF)
        ):
            shaped -= self.death_penalty
        self._prev_life = life

        # 4) 可选的小额时间惩罚，抑制原地磨蹭（默认 0，别乱开）
        # shaped -= self.time_penalty

        info["x_best"] = self._best_x  # 方便调试
        return obs, shaped, done, info
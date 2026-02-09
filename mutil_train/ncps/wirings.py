# -*- coding: utf-8 -*-
# train/ncps/wirings.py
# 从原始NCP包迁移的连接结构定义模块

import numpy as np

class Wiring:
    """
    Wiring类定义了神经网络中神经元之间的连接结构
    是模型结构设计的核心组件
    """
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None

    @property
    def num_layers(self):
        return 1

    def get_neurons_of_layer(self, layer_id):
        return list(range(self.units))

    def is_built(self):
        return self.input_dim is not None

    def build(self, input_dim):
        if not self.input_dim is None and self.input_dim != input_dim:
            raise ValueError(
                "输入维度冲突。set_input_dim()设置为{}，但实际输入维度为{}".format(
                    self.input_dim, input_dim
                )
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.adjacency_matrix)

    def sensory_erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.sensory_adjacency_matrix)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32
        )

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    # 可由子类重写
    def get_type_of_neuron(self, neuron_id):
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(
                "无法添加来自{}的突触，细胞只有{}个单元".format(
                    src, self.units
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "无法添加通向{}的突触，细胞只有{}个单元".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "无法添加极性为{}的突触（预期-1或+1）".format(
                    polarity
                )
            )
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError(
                "在调用build()之前无法添加感觉突触！"
            )
        if src < 0 or src >= self.input_dim:
            raise ValueError(
                "无法添加来自{}的感觉突触，输入只有{}个特征".format(
                    src, self.input_dim
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "无法添加通向{}的突触，细胞只有{}个单元".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "无法添加极性为{}的突触（预期-1或+1）".format(
                    polarity
                )
            )
        self.sensory_adjacency_matrix[src, dest] = polarity

    def get_config(self):
        return {
            "units": self.units,
            "adjacency_matrix": self.adjacency_matrix.tolist() if self.adjacency_matrix is not None else None,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix.tolist() if self.sensory_adjacency_matrix is not None else None,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_config(cls, config):
        # 有一个更简洁的解决方案，但这能工作
        wiring = Wiring(config["units"])
        if config["adjacency_matrix"] is not None:
            wiring.adjacency_matrix = np.array(config["adjacency_matrix"])
        if config["sensory_adjacency_matrix"] is not None:
            wiring.sensory_adjacency_matrix = np.array(config["sensory_adjacency_matrix"])
        wiring.input_dim = config["input_dim"]
        wiring.output_dim = config["output_dim"]

        return wiring

    @property
    def synapse_count(self):
        """计算模型内部神经元之间的突触数量"""
        return np.sum(np.abs(self.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        """计算从输入（感觉神经元）到模型内部神经元的突触数量"""
        return np.sum(np.abs(self.sensory_adjacency_matrix))


class FullyConnected(Wiring):
    """
    全连接模式的神经连接结构
    """
    def __init__(
        self, units, output_dim=None, erev_init_seed=1111, self_connections=True
    ):
        super(FullyConnected, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        self._erev_init_seed = erev_init_seed
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                polarity = self._rng.choice([-1, 1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def get_config(self):
        return {
            "units": self.units,
            "output_dim": self.output_dim,
            "erev_init_seed": self._erev_init_seed,
            "self_connections": self.self_connections
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Random(Wiring):
    """
    随机连接结构，可以控制稀疏程度
    """
    def __init__(self, units, output_dim=None, sparsity_level=0.0, random_seed=1111):
        super(Random, self).__init__(units)
        if output_dim is None:
            output_dim = units
        self.set_output_dim(output_dim)
        self.sparsity_level = sparsity_level

        if sparsity_level < 0.0 or sparsity_level >= 1.0:
            raise ValueError(
                "无效的稀疏度级别'{}'，预期值范围为[0,1)".format(
                    sparsity_level
                )
            )
        self._rng = np.random.default_rng(random_seed)
        self._random_seed = random_seed

        number_of_synapses = int(np.round(units * units * (1 - sparsity_level)))
        all_synapses = []
        for src in range(self.units):
            for dest in range(self.units):
                all_synapses.append((src, dest))

        used_synapses = self._rng.choice(
            all_synapses, size=number_of_synapses, replace=False
        )
        for src, dest in used_synapses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        number_of_sensory_synapses = int(
            np.round(self.input_dim * self.units * (1 - self.sparsity_level))
        )
        all_sensory_synapses = []
        for src in range(self.input_dim):
            for dest in range(self.units):
                all_sensory_synapses.append((src, dest))

        used_sensory_synapses = self._rng.choice(
            all_sensory_synapses, size=number_of_sensory_synapses, replace=False
        )
        for src, dest in used_sensory_synapses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)

    def get_config(self):
        return {
            "units": self.units,
            "output_dim": self.output_dim,
            "sparsity_level": self.sparsity_level,
            "random_seed": self._random_seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NCP(Wiring):
    """
    神经回路策略(Neural Circuit Policy)连接结构
    实现了类似生物神经系统的分层连接方式
    """
    def __init__(
        self,
        inter_neurons,
        command_neurons,
        motor_neurons,
        sensory_fanout,
        inter_fanout,
        recurrent_command_synapses,
        motor_fanin,
        seed=22222,
    ):
        """
        创建神经回路策略连接结构
        RNN的总神经元数量（=状态大小）由中间层、命令层和电机层神经元的总和给出
        更简单的NCP连接结构生成方式请参见"AutoNCP"连接类

        Args:
            inter_neurons: 中间神经元数量（第2层）
            command_neurons: 命令神经元数量（第3层）
            motor_neurons: 电机神经元数量（第4层=输出数量）
            sensory_fanout: 感觉到中间神经元的平均输出突触数
            inter_fanout: 中间到命令神经元的平均输出突触数
            recurrent_command_synapses: 命令神经元层中的平均循环连接数
            motor_fanin: 电机神经元从命令神经元的平均输入突触数
            seed: 用于生成连接的随机种子
        """

        super(NCP, self).__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

        # 神经元ID：[0..motor ... command ... inter]
        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(
            range(
                self._num_motor_neurons,
                self._num_motor_neurons + self._num_command_neurons,
            )
        )
        self._inter_neurons = list(
            range(
                self._num_motor_neurons + self._num_command_neurons,
                self._num_motor_neurons
                + self._num_command_neurons
                + self._num_inter_neurons,
            )
        )

        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(
                "错误：电机扇入参数是{}，但命令神经元只有{}个".format(
                    self._motor_fanin, self._num_command_neurons
                )
            )
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(
                "错误：感觉扇出参数是{}，但中间神经元只有{}个".format(
                    self._sensory_fanout, self._num_inter_neurons
                )
            )
        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(
                "错误：中间扇出参数是{}，但命令神经元只有{}个".format(
                    self._inter_fanout, self._num_command_neurons
                )
            )

    @property
    def num_layers(self):
        return 3

    def get_neurons_of_layer(self, layer_id):
        if layer_id == 0:
            return self._inter_neurons
        elif layer_id == 1:
            return self._command_neurons
        elif layer_id == 2:
            return self._motor_neurons
        raise ValueError("未知层{}".format(layer_id))

    def get_type_of_neuron(self, neuron_id):
        if neuron_id < self._num_motor_neurons:
            return "motor"
        if neuron_id < self._num_motor_neurons + self._num_command_neurons:
            return "command"
        return "inter"

    def _build_sensory_to_inter_layer(self):
        unreachable_inter_neurons = [l for l in self._inter_neurons]
        # 随机连接每个感觉神经元到exactly _sensory_fanout数量的中间神经元
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                self._inter_neurons, size=self._sensory_fanout, replace=False
            ):
                if dest in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

        # 如果有些中间神经元没有连接，现在连接它们
        mean_inter_neuron_fanin = int(
            self._num_sensory_neurons * self._sensory_fanout / self._num_inter_neurons
        )
        # 连接"遗忘"的中间神经元，至少1个，至多所有感觉神经元
        mean_inter_neuron_fanin = np.clip(
            mean_inter_neuron_fanin, 1, self._num_sensory_neurons
        )
        for dest in unreachable_inter_neurons:
            for src in self._rng.choice(
                self._sensory_neurons, size=mean_inter_neuron_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def _build_inter_to_command_layer(self):
        # 随机连接中间神经元到命令神经元
        unreachable_command_neurons = [l for l in self._command_neurons]
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                self._command_neurons, size=self._inter_fanout, replace=False
            ):
                if dest in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # 如果有些命令神经元没有连接，现在连接它们
        mean_command_neurons_fanin = int(
            self._num_inter_neurons * self._inter_fanout / self._num_command_neurons
        )
        # 连接"遗忘"的命令神经元，至少1个，至多所有中间神经元
        mean_command_neurons_fanin = np.clip(
            mean_command_neurons_fanin, 1, self._num_command_neurons
        )
        for dest in unreachable_command_neurons:
            for src in self._rng.choice(
                self._inter_neurons, size=mean_command_neurons_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def _build_recurrent_command_layer(self):
        # 在命令神经元中添加循环连接
        for i in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def _build_command__to_motor_layer(self):
        # 随机连接命令神经元到电机神经元
        unreachable_command_neurons = [l for l in self._command_neurons]
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                self._command_neurons, size=self._motor_fanin, replace=False
            ):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # 如果有些命令神经元没有连接，现在连接它们
        mean_command_fanout = int(
            self._num_motor_neurons * self._motor_fanin / self._num_command_neurons
        )
        # 连接"遗忘"的命令神经元，至少1个，至多所有电机神经元
        mean_command_fanout = np.clip(mean_command_fanout, 1, self._num_motor_neurons)
        for src in unreachable_command_neurons:
            for dest in self._rng.choice(
                self._motor_neurons, size=mean_command_fanout, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(0, self._num_sensory_neurons))

        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command__to_motor_layer()

    def get_config(self):
        return {
            "inter_neurons": self._inter_neurons,
            "command_neurons": self._command_neurons,
            "motor_neurons": self._motor_neurons,
            "sensory_fanout": self._sensory_fanout,
            "inter_fanout": self._inter_fanout,
            "recurrent_command_synapses": self._recurrent_command_synapses,
            "motor_fanin": self._motor_fanin,
            "seed": self._rng.seed(),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AutoNCP(NCP):
    """
    自动神经回路策略连接结构构建器
    只需要指定总神经元数量和输出大小即可生成合理的NCP结构
    """
    def __init__(
        self,
        units,
        output_size,
        sparsity_level=0.5,
        seed=22222,
    ):
        """
        只需指定神经元数量和输出数量即可实例化NCP连接结构
        
        Args:
            units: 总神经元数量
            output_size: 电机神经元数量(=输出大小)。该值必须小于units-2
            sparsity_level: 0.0(非常密集)到0.9(非常稀疏)的超参数
            seed: 生成连接的随机种子
        """
        self._output_size = output_size
        self._sparsity_level = sparsity_level
        self._seed = seed
        if output_size >= units - 2:
            raise ValueError(
                f"输出大小必须小于神经元数量-2(给定{units}个神经元，{output_size}个输出)"
            )
        if sparsity_level < 0.1 or sparsity_level > 1.0:
            raise ValueError(
                f"稀疏度级别必须在0.0和0.9之间(给定{sparsity_level})"
            )
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons

        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(int(command_neurons * density_level * 2), 1)
        motor_fanin = max(int(command_neurons * density_level), 1)
        super(AutoNCP, self).__init__(
            inter_neurons,
            command_neurons,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed=seed,
        )

    def get_config(self):
        return {
            "units": self.units,
            "output_size": self._output_size,
            "sparsity_level": self._sparsity_level,
            "seed": self._seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
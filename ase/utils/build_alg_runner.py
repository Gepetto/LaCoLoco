import learning.common_ase.common_ase_models as common_ase_models
import learning.common_ase.common_ase_players as common_ase_players

from learning.ase import ase_agent
from learning.ase import ase_network_builder

from learning.ase_drail import ase_drail_agent
from learning.ase_drail import ase_drail_network_builder

from learning.common_hrl import common_hrl_models
from learning.common_hrl import common_hrl_network_builder

from learning.hrl import hrl_agent
from learning.hrl import hrl_players

from learning.hrl_drail import hrl_drail_agent
from learning.hrl_drail import hrl_drail_players

from rl_games.algos_torch import players
from ase_envs.tasks.utils.rl_games_cat.cat_common import CaTA2CAgent

from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder


def build_alg_runner(algo_observer=None):
    if algo_observer == None:
        runner = Runner()
    else:
        runner = Runner(algo_observer)

    runner.algo_factory.register_builder(
        "ase", lambda **kwargs: ase_agent.ASEAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "ase", lambda **kwargs: common_ase_players.CommonASEPlayer(**kwargs)
    )
    model_builder.register_model(
        "ase",
        lambda network, **kwargs: common_ase_models.CommonModelASEContinuous(network),
    )
    model_builder.register_network(
        "ase", lambda **kwargs: ase_network_builder.ASEBuilder()
    )

    runner.algo_factory.register_builder(
        "ase_drail", lambda **kwargs: ase_drail_agent.ASEDRAILAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "ase_drail", lambda **kwargs: common_ase_players.CommonASEPlayer(**kwargs)
    )
    model_builder.register_model(
        "ase_drail",
        lambda network, **kwargs: common_ase_models.CommonModelASEContinuous(network),
    )
    model_builder.register_network(
        "ase_drail", lambda **kwargs: ase_drail_network_builder.ASEDRAILBuilder()
    )

    runner.algo_factory.register_builder(
        "hrl", lambda **kwargs: hrl_agent.HRLAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "hrl", lambda **kwargs: hrl_players.HRLPlayer(**kwargs)
    )
    model_builder.register_model(
        "hrl", lambda network, **kwargs: common_hrl_models.ModelHRLContinuous(network)
    )
    model_builder.register_network(
        "hrl", lambda **kwargs: common_hrl_network_builder.HRLBuilder()
    )

    runner.algo_factory.register_builder(
        "hrl_drail", lambda **kwargs: hrl_drail_agent.HRLAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "hrl_drail", lambda **kwargs: hrl_drail_players.HRLPlayer(**kwargs)
    )
    model_builder.register_model(
        "hrl_drail",
        lambda network, **kwargs: common_hrl_models.ModelHRLContinuous(network),
    )
    model_builder.register_network(
        "hrl_drail", lambda **kwargs: common_hrl_network_builder.HRLBuilder()
    )

    runner.algo_factory.register_builder(
        "cat_a2c_continuous", lambda **kwargs: CaTA2CAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "cat_a2c_continuous", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )

    return runner

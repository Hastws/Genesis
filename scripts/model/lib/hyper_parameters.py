#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum
from collections import namedtuple as named_tuple


class LabelIndex(enum.IntEnum):
    action_type_encoding = 0
    delay_encoding = 1
    queue_encoding = 2
    select_units_encoding = 3
    target_unit_encoding = 4
    target_location_encoding = 5


LabelSize = named_tuple(
    "LabelSize",
    [
        "action_type_encoding",
        "delay_encoding",
        "queue_encoding",
        "select_units_encoding",
        "target_unit_encoding",
        "target_location_encoding",
    ],
)


class ScalarFeature(enum.IntEnum):
    agent_statistics = 0
    home_race = 1
    away_race = 2
    upgrades = 3
    enemy_upgrades = 4
    time = 5
    available_actions = 6
    unit_counts_bow = 7
    mmr = 8
    units_buildings = 9
    effects = 10
    upgrade = 11
    beginning_build_order = 12
    last_delay = 13
    last_action_type = 14
    last_repeat_queued = 15
    # cumulative_score = 16


ScalarFeatureSize = named_tuple(
    "ScalarFeatureSize",
    [
        "agent_statistics",
        "home_race",
        "away_race",
        "upgrades",
        "enemy_upgrades",
        "time",
        "available_actions",
        "unit_counts_bow",
        "mmr",
        "units_buildings",
        "effects",
        "upgrade",
        "beginning_build_order",
        "last_delay",
        "last_action_type",
        "last_repeat_queued",
        "cumulative_score",
    ],
)

from pysc2.lib.actions import FUNCTIONS, RAW_FUNCTIONS
from pysc2.lib.upgrades import Upgrades
from pysc2.lib.features import Effects
from pysc2.lib.buffs import Buffs
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg

from pysc2.env import sc2_env

import param as PARAMS

TRAINING_RACES = ("Protoss", "Zerg", "Terran")

DatasetSplitRatio = named_tuple("DataSplitRatio", ["training", "val", "test"])

DATASET_SPLIT_RATIO = DatasetSplitRatio(training=0.80, val=0.10, test=0.10)


class ConstSize(object):
    Actions_Size = len(RAW_FUNCTIONS)
    Upgrades_Size = 320
    Effects_Size = len(Effects)
    Neutral_Units_Size = len(Neutral)
    Protoss_Units_Size = len(Protoss)
    Terran_Units_Size = len(Terran)
    Zerg_Units_Size = len(Zerg)
    All_Units_Size = len(Neutral) + len(Protoss) + len(Terran) + len(Zerg)


ArchHyperParameters = named_tuple(
    "ArchHyperParameters",
    [
        "batch_size",
        "sequence_length",
        "max_entities",
        "max_selected",
        "temperature",
        "minimap_size",
        "embedding_size",
        "map_channels",
        "scatter_channels",
        "init_net_params",
        "use_action_type_mask",
        "use_unit_type_entity_mask",
        "use_predict_step_mul",
        "positional_encoding_time",
        "scalar_encoder_fc1_input",
        "scalar_encoder_fc2_input",
        "scalar_feature_size",
        "entity_embedding_size",
        "lstm_hidden_dim",
        "lstm_layers",
        "n_resblocks",
        "original_1024",
        "original_512",
        "original_256",
        "original_128",
        "original_64",
        "original_32",
        "context_size",
        "location_head_max_map_channels",
        "autoregressive_embedding_size",
        "winloss_baseline_input_size",
        "build_order_baseline_input_size",
        "built_units_baseline_input_size",
        "upgrades_baseline_input_size",
        "effects_baseline_input_size",
        "league_learner_num",
        "actorloop_num",
    ],
)

ARCH_HYPER_PARAMETERS = ArchHyperParameters(
    batch_size=int(16 * 8 / PARAMS.BATCH_SCALE),
    sequence_length=int(16 * 8 / PARAMS.SEQ_SCALE),
    max_selected=int(48 / PARAMS.SELECT_SCALE),
    max_entities=int(512),  # this value can not be downscaled, will miss many units.
    temperature=0.8,
    minimap_size=64,
    embedding_size=1856,  # 1545, 1856, 2109
    map_channels=24,
    scatter_channels=1,  # 0 or 1
    init_net_params=1,
    use_action_type_mask=1,
    use_unit_type_entity_mask=0,
    positional_encoding_time=0,
    use_predict_step_mul=0,
    scalar_encoder_fc1_input=864,
    scalar_encoder_fc2_input=448,
    scalar_feature_size=7327,  # Deprecated
    entity_embedding_size=64,
    lstm_hidden_dim=128,
    lstm_layers=1,
    n_resblocks=4,
    original_1024=256,
    original_512=128,
    original_256=64,
    original_128=32,
    original_64=48,
    original_32=16,
    context_size=128,
    location_head_max_map_channels=32,
    autoregressive_embedding_size=256,
    winloss_baseline_input_size=1152,
    build_order_baseline_input_size=1216,
    built_units_baseline_input_size=1152,
    upgrades_baseline_input_size=1152,
    effects_baseline_input_size=1152,
    league_learner_num=4,
    actorloop_num=512,
)

SLTrainingHyperParameters = named_tuple(
    "SLTrainingHyperParameters",
    ["num_epochs", "learning_rate", "weight_decay", "clip", "seed"],
)

SL_TRAINING_HYPER_PARAMETERS = SLTrainingHyperParameters(
    num_epochs=100, learning_rate=1e-3, weight_decay=1e-5, clip=0.5, seed=1
)

RLTrainingHyperParameters = named_tuple(
    "RLTrainingHyperParameters",
    [
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "clip",
        "num_epochs",
        "seed",
    ],
)

RL_TRAINING_HYPER_PARAMETERS = RLTrainingHyperParameters(
    learning_rate=1e-5,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0,
    clip=0.5,
    num_epochs=2,
    seed=1,
)

AgentInterfaceFormatParams = named_tuple(
    "AgentInterfaceFormatParams",
    [
        "feature_dimensions",
        "rgb_dimensions",
        "raw_resolution",
        "action_space",
        "camera_width_world_units",
        "use_feature_units",
        "use_raw_units",
        "use_raw_actions",
        "max_raw_actions",
        "max_selected_units",
        "use_unit_counts",
        "use_camera_position",
        "show_cloaked",
        "show_burrowed_shadows",
        "show_placeholders",
        "hide_specific_actions",
        "action_delay_fn",
        "send_observation_proto",
        "crop_to_playable_area",
        "raw_crop_to_playable_area",
        "allow_cheating_layers",
        "add_cargo_to_units",
    ],
)

AGENT_INTERFACE_FORMAT_PARAMS = AgentInterfaceFormatParams(
    feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
    rgb_dimensions=None,
    raw_resolution=64,
    action_space=None,
    camera_width_world_units=24,
    use_feature_units=True,
    use_raw_units=True,
    use_raw_actions=True,
    max_raw_actions=512,
    max_selected_units=512,
    use_unit_counts=True,
    use_camera_position=False,
    show_cloaked=True,
    show_burrowed_shadows=True,
    show_placeholders=True,
    hide_specific_actions=True,
    action_delay_fn=None,
    send_observation_proto=False,
    crop_to_playable_area=False,
    raw_crop_to_playable_area=False,
    allow_cheating_layers=False,
    add_cargo_to_units=False,
)


StarCraftHyperParameters = named_tuple(
    "StarCraftHyperParameters",
    [
        "screen_size",
        "world_size",
        "max_unit_type",
        "count_beginning_build_order",
        "sc2_default_delay",
        "max_order_ids",
        "max_buffer_ids",
        "max_add_on_type",
        "map_name",
        "game_version",
    ],
)

StarCraft_Hyper_Parameters = StarCraftHyperParameters(
    screen_size=64,  # 128 comsume to much resource, 32 is too small to see
    world_size=AGENT_INTERFACE_FORMAT_PARAMS.raw_resolution,
    # world_size is original 256, a SC2 map has most 256x256 size, but we can use raw_resolution to make it down to 64
    max_unit_type=ConstSize.All_Units_Size,
    count_beginning_build_order=20,
    sc2_default_delay=32,
    max_order_ids=ConstSize.Actions_Size,
    max_buffer_ids=len(Buffs),  # from 0 to 275 TODO: change to value in static_data
    max_add_on_type=50,  # TODO: change to value in static_data
    map_name="Simple64",  # "Simple64" or "AbyssalReef"
    game_version="latest",  # maybe "4.10"
)

SCALAR_FEATURE_SIZE = ScalarFeatureSize(
    agent_statistics=10,
    home_race=5,
    away_race=5,
    upgrades=ConstSize.Upgrades_Size,
    enemy_upgrades=ConstSize.Upgrades_Size,
    time=64,
    available_actions=ConstSize.Actions_Size,
    unit_counts_bow=ConstSize.All_Units_Size,
    mmr=7,
    units_buildings=ConstSize.All_Units_Size,
    effects=ConstSize.Effects_Size,
    upgrade=ConstSize.Upgrades_Size,
    beginning_build_order=StarCraft_Hyper_Parameters.count_beginning_build_order
    * ConstSize.All_Units_Size,
    last_delay=128,
    last_action_type=ConstSize.Actions_Size,
    last_repeat_queued=2,
    cumulative_score=13,
)

LABEL_SIZE = LabelSize(
    action_type_encoding=SCALAR_FEATURE_SIZE.available_actions,
    delay_encoding=SCALAR_FEATURE_SIZE.last_delay,
    queue_encoding=SCALAR_FEATURE_SIZE.last_repeat_queued,
    select_units_encoding=ARCH_HYPER_PARAMETERS.max_entities
    * ARCH_HYPER_PARAMETERS.max_selected,
    target_unit_encoding=ARCH_HYPER_PARAMETERS.max_entities * 1,
    target_location_encoding=StarCraft_Hyper_Parameters.world_size**2,
)

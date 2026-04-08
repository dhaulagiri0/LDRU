"""Constants for the LDRU experiments."""

# Source: Delétang et al. (2023)

import functools

import haiku as hk

from ldru.models import rnn
from ldru.models import transformer
from ldru.models import ldru_v2

s5_loaded = False
try:
    from ldru.models import s5_ssm

    s5_loaded = True
    print("S5 architecture loaded successfully.")
except ImportError:
    print(
        "WARNING: S5 architecture could not be imported. It can be installed from the S5 GitHub repository"
    )
    print(
        "There is one caveat with this: when installing the S5 package.\nYou must update any reference to `jax.numpy.DeviceArray` to `jax.array` in the S5 codebase."
    )
    print("This happens because the S5 codebase uses an older version of JAX.")
    print("If you do not plan to use S5, you can ignore this warning.")

from ldru.models import sdssm

from ldru.tasks.regular import cycle_navigation
from ldru.tasks.regular import even_pairs
from ldru.tasks.regular import modular_arithmetic
from ldru.tasks.regular import parity_check
from ldru.tasks.regular import n_prefix_symbols
from ldru.tasks.regular import tomita_1
from ldru.tasks.regular import tomita_3
from ldru.tasks.regular import tomita_4
from ldru.tasks.regular import tomita_5
from ldru.tasks.regular import tomita_6
from ldru.tasks.regular import tomita_7
from ldru.tasks.regular import d_n

from ldru.training import curriculum as curriculum_lib

from ldru.models.ldru_v2 import BinaryOperator
from ldru.models.ldru import ElementwiseSum, LinearCat, GatedSum

MODEL_BUILDERS = {
    "rnn": functools.partial(rnn.make_rnn, rnn_core=hk.VanillaRNN),
    "lstm": functools.partial(rnn.make_rnn, rnn_core=hk.LSTM),
    # RegularGPT is a transformer_encoder with "0" layers.
    "transformer_encoder": transformer.make_transformer_encoder,
    "ldru": ldru_v2.make_ldru,
    "sdssm": functools.partial(sdssm.make_sdssm, Lp_norm=1.2),
}
if s5_loaded:
    MODEL_BUILDERS["s5_ssm"] = s5_ssm.make_s5_haiku

BINARY_OPERATORS = {
    "elementwise_sum": ElementwiseSum,
    "gated_sum": GatedSum,
    "linear": LinearCat,
    "MLP": BinaryOperator,
}

CURRICULUM_BUILDERS = {
    "fixed": curriculum_lib.FixedCurriculum,
    "regular_increase": curriculum_lib.RegularIncreaseCurriculum,
    "reverse_exponential": curriculum_lib.ReverseExponentialCurriculum,
    "uniform": curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    "modular_arithmetic": modular_arithmetic.ModularArithmetic,
    "parity_check": parity_check.ParityCheck,
    "even_pairs": even_pairs.EvenPairs,
    "cycle_navigation": cycle_navigation.CycleNavigation,
    "n_prefix_symbols": n_prefix_symbols.NPrefixSymbols,
    "tomita_1": tomita_1.Tomita1,
    "tomita_3": tomita_3.Tomita3,
    "tomita_4": tomita_4.Tomita4,
    "tomita_5": tomita_5.Tomita5,
    "tomita_6": tomita_6.Tomita6,
    "tomita_7": tomita_7.Tomita7,
    "d_n": d_n.D_n,
}

TASK_LEVELS = {
    "modular_arithmetic": "regular",
    "parity_check": "regular",
    "even_pairs": "regular",
    "cycle_navigation": "regular",
    "n_prefix_symbols": "regular",
    "tomita_1": "regular",
    "tomita_3": "regular",
    "tomita_4": "regular",
    "tomita_5": "regular",
    "tomita_6": "regular",
    "tomita_7": "regular",
    "d_n": "regular",
}

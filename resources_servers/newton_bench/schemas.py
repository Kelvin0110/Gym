# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional, Dict, Any

from pydantic import BaseModel


class M0GravityRequest(BaseModel):
    mass1: float
    mass2: float
    distance: float

    # for simple_system and complex_system
    initial_velocity: Optional[float] = None
    duration: Optional[float] = None
    time_step: Optional[float] = None


class M1CoulombForceRequest(BaseModel):
    q1: float
    q2: float
    distance: float

    # for simple_system and complex_system
    m1: Optional[float] = None
    m2: Optional[float] = None
    duration: Optional[float] = None
    time_step: Optional[float] = None


class M2MagneticForceRequest(BaseModel):
    current1: float
    current2: float
    distance: float

    # for simple_system and complex_system
    mass_wire: Optional[float] = None
    initial_velocity: Optional[float] = None
    duration: Optional[float] = None
    time_step: Optional[float] = None


class M3FourierLawRequest(BaseModel):
    k: float
    A: float
    delta_T: float
    d: float

    # for simple_system and complex_system
    num_points: Optional[int] = None


class M4SnellLawRequest(BaseModel):
    refractive_index_1: Optional[float] = None
    refractive_index_2: Optional[float] = None
    incidence_angle: float

    # for simple_system and complex_system
    refractive_index_3: Optional[float] = None
    speed_medium1: Optional[float] = None
    speed_medium2: Optional[float] = None


class M5RadioactiveDecayRequest(BaseModel):
    N0: Optional[float] = None
    lambda_decay: Optional[float] = None
    t: float

    # for simple_system and complex_system
    N0a: Optional[float] = None
    N0b: Optional[float] = None
    lambda_a: Optional[float] = None
    lambda_b: Optional[float] = None
    num_points: Optional[int] = None


class M6UnderdampedHarmonicRequest(BaseModel):
    k_constant: float
    mass: float
    b_constant: float

    # for simple_system and complex_system
    initial_amplitude: Optional[float] = None


class M7MalusLawRequest(BaseModel):
    I_0: float
    theta: float


class M8SoundSpeedRequest(BaseModel):
    adiabatic_index: float
    temperature: float
    molar_mass: float

    # for simple_system and complex_system
    distance: Optional[float] = None
    driving_frequency: Optional[float] = None
    tube_diameter: Optional[float] = None


class M9HookeLawRequest(BaseModel):
    x: float

    # for simple_system and complex_system
    m: Optional[float] = None


class M10BEDistributionRequest(BaseModel):
    omega: Optional[float] = None
    temperature: float

    # for simple_system and complex_system
    probe_frequency: Optional[float] = None
    center_frequency: Optional[float] = None
    bandwidth: Optional[float] = None


class M11HeatTransferRequest(BaseModel):
    m: float
    c: float
    delta_T: float


MODULE_REQUEST_CLASSES_MAPPING: Dict[str, Any] = {
    "m0_gravity": M0GravityRequest,
    "m1_coulomb_force": M1CoulombForceRequest,
    "m2_magnetic_force": M2MagneticForceRequest,
    "m3_fourier_law": M3FourierLawRequest,
    "m4_snell_law": M4SnellLawRequest,
    "m5_radioactive_decay": M5RadioactiveDecayRequest,
    "m6_underdamped_harmonic": M6UnderdampedHarmonicRequest,
    "m7_malus_law": M7MalusLawRequest,
    "m8_sound_speed": M8SoundSpeedRequest,
    "m9_hooke_law": M9HookeLawRequest,
    "m10_be_distribution": M10BEDistributionRequest,
    "m11_heat_transfer": M11HeatTransferRequest,
}

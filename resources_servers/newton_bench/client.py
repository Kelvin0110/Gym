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

"""
NewtonBench Client Demo - AI Physicist Workflow

This script demonstrates the full workflow of an AI physicist agent
interacting with the NewtonBench resource server:

1. Initialize a session with a physics module (m0_gravity)
2. Run systematic experiments to gather data
3. Analyze data using execute_python (sandbox)
4. Discover the underlying physical law

Prerequisites:
- NewtonBench server running on localhost:8080
- NewtonBench repo cloned in project root (for run_experiment to work)

Usage:
    python client.py
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def seed_session(session, module_name="m0_gravity", difficulty="easy",
                 system="vanilla_equation", noise_level=0.0, law_version="v0"):
    """
    Initialize a NewtonBench session for a specific physics module.

    Args:
        session: requests.Session object (maintains cookies)
        module_name: Physics module (e.g., "m0_gravity", "m9_hooke_law")
        difficulty: "easy", "medium", or "hard"
        system: "vanilla_equation", "simple_system", or "complex_system"
        noise_level: Measurement noise (0.0 = perfect data)
        law_version: "v0", "v1", or "v2" (different constants)

    Returns:
        Response JSON
    """
    response = session.post(
        f"{BASE_URL}/seed_session",
        json={
            "module_name": module_name,
            "difficulty": difficulty,
            "system": system,
            "noise_level": noise_level,
            "law_version": law_version,
        },
    )
    response.raise_for_status()
    return response.json()


def run_gravity_experiment(session, mass1, mass2, distance):
    """
    Run a gravity experiment and return the measured force.

    Args:
        session: requests.Session with active session
        mass1: Mass of first object (kg)
        mass2: Mass of second object (kg)
        distance: Distance between objects (m)

    Returns:
        dict with 'result' containing the force measurement
    """
    response = session.post(
        f"{BASE_URL}/run_experiment_m0_gravity",
        json={
            "mass1": mass1,
            "mass2": mass2,
            "distance": distance,
        },
    )
    response.raise_for_status()
    return response.json()


def execute_python(session, code):
    """
    Execute Python code in the sandboxed environment.

    The sandbox has numpy, scipy, pandas pre-imported.
    Variables persist across calls within the same session.

    Args:
        session: requests.Session with active session
        code: Python code string to execute

    Returns:
        dict with 'stdout', 'stderr', 'result'
    """
    response = session.post(
        f"{BASE_URL}/execute_python",
        json={"code": code},
    )
    response.raise_for_status()
    return response.json()


def end_session(session):
    """
    End the current session and clean up resources.

    Args:
        session: requests.Session with active session

    Returns:
        Response JSON
    """
    response = session.post(
        f"{BASE_URL}/end_session",
        json={},
    )
    response.raise_for_status()
    return response.json()


def demo_physics_discovery():
    """
    Demonstrate the full AI physicist workflow.

    Simulates an agent discovering Newton's Law of Gravitation:
    F = G * m1 * m2 / r^2
    """
    print("=" * 60)
    print("NewtonBench: AI Physicist Demo")
    print("=" * 60)
    print()

    # Create session (maintains cookies across requests)
    session = requests.Session()

    try:
        # --- Phase 1: Initialize ---
        print("Phase 1: Starting experiment session...")
        print("   Module: m0_gravity (Newton's Law of Gravitation)")
        print("   Difficulty: easy")
        seed_result = seed_session(session, module_name="m0_gravity", difficulty="easy")
        print("   Session initialized")
        print()

        # --- Phase 2: Run Experiments ---
        print("Phase 2: Running experiments...")
        print()

        # Experiment Set 1: Vary distance
        print("   Testing distance dependence (m1=100, m2=100):")
        distances = [10, 20, 40]
        forces_by_distance = []
        for d in distances:
            result = run_gravity_experiment(session, mass1=100, mass2=100, distance=d)
            # Handle different response formats
            res = result.get("result", {})
            if isinstance(res, dict):
                force = res.get("force", res.get("result", 0))
            else:
                force = res
            forces_by_distance.append(force)
            print(f"      distance={d:3} -> force={force:.6e}")
        print()

        # Experiment Set 2: Vary mass1
        print("   Testing mass1 dependence (m2=100, distance=10):")
        masses1 = [50, 100, 200]
        forces_by_mass1 = []
        for m1 in masses1:
            result = run_gravity_experiment(session, mass1=m1, mass2=100, distance=10)
            res = result.get("result", {})
            if isinstance(res, dict):
                force = res.get("force", res.get("result", 0))
            else:
                force = res
            forces_by_mass1.append(force)
            print(f"      mass1={m1:3} -> force={force:.6e}")
        print()

        # --- Phase 3: Analyze with Python ---
        print("Phase 3: Analyzing data with Python...")
        print()

        # Build analysis code with collected data
        analysis_code = f"""
import numpy as np

# Data from experiments
distances = np.array({distances})
forces_d = np.array({forces_by_distance})
masses1 = np.array({masses1})
forces_m = np.array({forces_by_mass1})

# Check inverse-square law: F * r^2 should be constant
f_times_r2 = forces_d * distances**2
print("Checking inverse-square law:")
print(f"   F * r^2 = {{f_times_r2}}")
ratio = f_times_r2.max() / f_times_r2.min() if f_times_r2.min() != 0 else float('inf')
print(f"   Ratio max/min = {{ratio:.6f}}")
if abs(ratio - 1) < 0.01:
    print("   Confirmed: F is proportional to 1/r^2")
print()

# Check mass1 dependence: F / m1 should be constant
f_over_m1 = forces_m / masses1
print("Checking mass1 dependence:")
print(f"   F / m1 = {{f_over_m1}}")
ratio_m = f_over_m1.max() / f_over_m1.min() if f_over_m1.min() != 0 else float('inf')
if abs(ratio_m - 1) < 0.01:
    print("   Confirmed: F is proportional to m1")
print()

# Calculate gravitational constant
# F = C * m1 * m2 / r^2
# C = F * r^2 / (m1 * m2)
m1, m2, r, F = 100, 100, 10, forces_d[0]
C = F * r**2 / (m1 * m2)
print(f"Calculated gravitational constant:")
print(f"   C = F * r^2 / (m1 * m2) = {{C:.6e}}")
"""

        exec_result = execute_python(session, analysis_code)
        print("   Sandbox output:")
        if exec_result.get("stdout"):
            for line in exec_result["stdout"].strip().split("\n"):
                print(f"   {line}")
        if exec_result.get("stderr"):
            print("   Errors:")
            for line in exec_result["stderr"].strip().split("\n"):
                print(f"   {line}")
        print()

        # --- Phase 4: Discovered Law ---
        print("Phase 4: Discovered Law")
        print()
        print("   Based on experiments and analysis:")
        print("   +---------------------------------------------+")
        print("   |  F = C * m1 * m2 / r^2                      |")
        print("   |  (Newton's Law of Universal Gravitation)   |")
        print("   +---------------------------------------------+")
        print()

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server at", BASE_URL)
        print("Make sure the NewtonBench server is running:")
        print('  ng_run "+config_paths=[resources_servers/newton_bench/configs/newton_bench.yaml]"')
        return
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        return
    finally:
        # --- Phase 5: Cleanup ---
        try:
            print("Phase 5: Ending session...")
            end_session(session)
            print("   Session ended")
            print()
        except Exception:
            pass  # Ignore cleanup errors

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demo_execute_python_only():
    """
    Simple demo showing just execute_python functionality.
    
    Useful for testing if the sandbox works without needing
    the full NewtonBench module.
    """
    print("=" * 60)
    print("NewtonBench: Simple Execute Python Demo")
    print("=" * 60)
    print()

    session = requests.Session()

    try:
        # Seed session (required before execute_python)
        print("Seeding session...")
        seed_session(session, module_name="m0_gravity")
        print("Session initialized")
        print()

        # Execute some Python code
        print("Executing Python code in sandbox...")
        code = """
import numpy as np

# Create some data
x = np.linspace(0, 10, 5)
y = x ** 2

print("x values:", x)
print("y = x^2:", y)
print()
print("Sum of y:", np.sum(y))
"""
        result = execute_python(session, code)
        
        print("Output:")
        if result.get("stdout"):
            print(result["stdout"])
        if result.get("result"):
            print("Return value:", result["result"])
        print()

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server at", BASE_URL)
        print("Make sure the NewtonBench server is running.")
        return
    finally:
        try:
            end_session(session)
            print("Session ended")
        except Exception:
            pass

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the full physics discovery demo
    demo_physics_discovery()
    
    # Alternatively, run the simpler demo:
    # demo_execute_python_only()

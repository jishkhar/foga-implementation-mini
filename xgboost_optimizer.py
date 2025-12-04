import subprocess
import time
import random
import os
import sys
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# --- Configuration ---
GCC_FLAGS = [
    "-faggressive-loop-optimizations", "-falign-functions", "-falign-jumps",
    "-falign-labels", "-falign-loops", "-fassociative-math", "-fauto-inc-dec",
    "-fbranch-probabilities", "-fcaller-saves", "-fcode-hoisting",
    "-fcombine-stack-adjustments", "-fcommon", "-fcompare-elim",
    "-fconserve-stack", "-fcprop-registers", "-fcrossjumping",
    "-fcse-follow-jumps", "-fcx-fortran-rules", "-fcx-limited-range",
    "-fdce", "-fdefer-pop", "-fdelayed-branch", "-fdelete-null-pointer-checks",
    "-fdevirtualize", "-fdse", "-fearly-inlining", "-fexceptions",
    "-ffast-math", "-ffinite-math-only", "-ffloat-store", "-fipa-cp",
    "-fipa-cp-clone", "-fipa-icf", "-finline-functions", "-fipa-modref",
    "-fipa-pure-const", "-fipa-sra", "-fjump-tables", "-flive-range-shrinkage",
    "-fmove-loop-invariants", "-fomit-frame-pointer", "-foptimize-sibling-calls",
    "-fpeel-loops", "-free", "-frename-registers", "-frerun-cse-after-loop",
    "-frounding-math", "-fsched-interblock", "-fsched-spec", "-fschedule-insns",
    "-fsection-anchors", "-fshort-enums", "-fshrink-wrap", "-fsplit-paths",
    "-fsplit-wide-types", "-fstrict-aliasing", "-fthread-jumps", "-ftrapv",
    "-ftree-bit-ccp", "-ftree-builtin-call-dce", "-ftree-ccp", "-ftree-ch",
    "-ftree-copy-prop", "-ftree-dce", "-ftree-dominator-opts", "-ftree-dse",
    "-ftree-forwprop", "-ftree-fre", "-ftree-loop-optimize", "-ftree-loop-vectorize",
    "-ftree-pre", "-ftree-pta", "-ftree-reassoc", "-ftree-slp-vectorize",
    "-ftree-sra", "-ftree-ter", "-ftree-vrp", "-funroll-all-loops",
    "-funsafe-math-optimizations", "-funswitch-loops", "-funwind-tables",
    "-fvar-tracking-assignments", "-fweb"
]

# XGBoost Parameters
INITIAL_RANDOM_SAMPLES = 100
BOOSTING_ROUNDS = 50
TOP_FLAGS_COUNT = 20

EXECUTION_TIMEOUT = 10
COMPILATION_TIMEOUT = 30

class XGBoostOptimizer:
    """XGBoost-based optimizer for compiler flags"""

    def __init__(self, source_file_path, test_input=None):
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"Source file not found: {source_file_path}")

        self.source_file_path = source_file_path
        self.test_input = test_input
        self.best_config = None
        self.best_time = float('inf')
        self.evaluation_count = 0

        self.configurations = []
        self.execution_times = []

        if source_file_path.endswith('.c'):
            self.compiler = 'gcc'
        elif source_file_path.endswith('.cpp') or source_file_path.endswith('.cc'):
            self.compiler = 'g++'
        else:
            raise ValueError("Unsupported file type. Please provide a .c or .cpp file.")

        print("=" * 70)
        print("🚀 XGBoost Optimizer")
        print("=" * 70)
        print(f"Compiler: {self.compiler.upper()}")
        print(f"Source: {source_file_path}")
        print(f"Total flags in search space: {len(GCC_FLAGS)}")

    def evaluate_configuration(self, config):
        """Compile and execute with given configuration, return execution time"""
        self.evaluation_count += 1
        flags_str = " ".join([GCC_FLAGS[i] for i, val in enumerate(config) if val == 1])
        output_binary = f"temp_xgb_{os.getpid()}_{self.evaluation_count}"
        compile_command = f"{self.compiler} -O3 {self.source_file_path} {flags_str} -o {output_binary}"

        try:
            result = subprocess.run(
                compile_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=COMPILATION_TIMEOUT
            )

            if result.returncode != 0:
                return float('inf')

            start_time = time.time()
            run_result = subprocess.run(
                f"./{output_binary}",
                shell=True,
                input=self.test_input,
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT
            )

            if run_result.returncode != 0:
                return float('inf')

            exec_time = time.time() - start_time

            if exec_time < self.best_time:
                self.best_time = exec_time
                self.best_config = config.copy()

            return exec_time

        except subprocess.TimeoutExpired:
            return float('inf')
        finally:
            if os.path.exists(output_binary):
                os.remove(output_binary)

    def optimize(self):
        """Run XGBoost-based optimization"""
        start_time = time.time()
        print("\n" + "=" * 70)
        print("📊 PHASE 1: Random Sampling")
        print("=" * 70)

        # Generate initial random samples
        for i in range(INITIAL_RANDOM_SAMPLES):
            config = [random.randint(0, 1) for _ in range(len(GCC_FLAGS))]
            print(f"Sampling {i+1}/{INITIAL_RANDOM_SAMPLES}...", end='\r')
            exec_time = self.evaluate_configuration(config)
            if exec_time != float('inf'):
                self.configurations.append(config)
                self.execution_times.append(exec_time)

        print(f"\nCollected {len(self.configurations)} valid samples")
        print(f"Best time so far: {self.best_time:.6f}s")

        print("\n" + "=" * 70)
        print("🤖 PHASE 2: Training XGBoost Model")
        print("=" * 70)

        # Train XGBoost model
        X = np.array(self.configurations)
        y = np.array(self.execution_times)
        
        if len(self.configurations) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            print("XGBoost model trained successfully.")
        else:
            print("⚠ Not enough samples for split, training on all data.")
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            model.fit(X, y)

        print("\n" + "=" * 70)
        print("🚀 PHASE 3: XGBoost Guided Optimization")
        print("=" * 70)

        # Use the model to predict and refine configurations
        for i in range(BOOSTING_ROUNDS):
            config = [random.randint(0, 1) for _ in range(len(GCC_FLAGS))]
            predicted_time = model.predict([config])[0]
            
            print(f"Iteration {i+1}/{BOOSTING_ROUNDS}: Best: {self.best_time:.6f}s", end='\r')

            if predicted_time < self.best_time:
                exec_time = self.evaluate_configuration(config)
                if exec_time != float('inf'):
                    self.configurations.append(config)
                    self.execution_times.append(exec_time)
                    if exec_time < self.best_time: # Check if it actually improved
                         print(f"\n✨ New Best Time: {exec_time:.6f}s")

        print("\n\n✅ Optimization complete.")
        
        total_time = time.time() - start_time
        self.print_results(total_time)

    def print_results(self, total_time):
        """Print final optimization results"""
        print("\n" + "=" * 70)
        print("✅ XGBOOST OPTIMIZATION COMPLETE")
        print("=" * 70)

        if self.best_config and self.best_time != float('inf'):
            print(f"\n🏆 Best Execution Time: {self.best_time:.6f} seconds")
            print(f"⏱️  Total Optimization Time: {total_time:.2f} seconds")
            print(f"📊 Total Evaluations: {self.evaluation_count}")
            
            enabled_flags = [GCC_FLAGS[i] for i, val in enumerate(self.best_config) if val == 1]
            
            print(f"\n📋 Enabled Flags ({len(enabled_flags)} total):")
            for i, flag in enumerate(enabled_flags, 1):
                print(f"  {i:2d}. {flag}")
            
            flags_str = " ".join(enabled_flags)
            print(f"\n💻 Optimal Compilation Command:")
            print(f"{self.compiler} -O3 {self.source_file_path} {flags_str} -o optimized_xgb")
            
            # Create optimized binary
            print("\n🔨 Creating optimized binary...")
            compile_cmd = f"{self.compiler} -O3 {self.source_file_path} {flags_str} -o optimized_xgb"
            result = subprocess.run(compile_cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                print("✅ Optimized binary created: ./optimized_xgb")
            else:
                print("⚠ Failed to create optimized binary")
        else:
            print("\n❌ Could not find a valid solution.")

    def save_results(self):
        """Save optimization results to a JSON file"""
        results = {
            'best_time': self.best_time,
            'total_evaluations': self.evaluation_count,
            'enabled_flags': [GCC_FLAGS[i] for i, val in enumerate(self.best_config) if val == 1]
        }

        with open('xgboost_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("Results saved to xgboost_results.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python xgboost_optimizer.py <source_file> [test_input_file]")
        sys.exit(1)

    source_file = sys.argv[1]
    test_input = None

    if len(sys.argv) > 2:
        input_file = sys.argv[2]
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                test_input = f.read()

    optimizer = XGBoostOptimizer(source_file, test_input)
    optimizer.optimize()
    optimizer.save_results()
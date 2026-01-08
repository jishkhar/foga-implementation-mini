import subprocess
import time
import random
import os
import sys
import numpy as np
from collections import defaultdict
import json

# Bayesian Optimization imports
try:
    from skopt import gp_minimize
    from skopt.space import Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Install with: pip install scikit-optimize")

# Random Forest imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

# --- Configuration (same as FOGA) ---
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
    "-fvar-tracking-assignments", "-fweb", "-fwrapv"
]

EXECUTION_TIMEOUT = 10
COMPILATION_TIMEOUT = 30

# HBRF Parameters
INITIAL_RANDOM_SAMPLES = 100  # Phase 1: Random sampling
TOP_FLAGS_COUNT = 20          # Phase 2: Top important flags
BAYESIAN_ITERATIONS = 50      # Phase 3: Bayesian optimization iterations
FINAL_GREEDY_ADDITIONS = 10   # Phase 4: Greedy refinement


class HBRFOptimizer:
    """Hybrid Bayesian-Random Forest Optimizer for compiler flags"""
    
    def __init__(self, source_file_path, test_input=None):
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"Source file not found: {source_file_path}")
        
        if not SKOPT_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("Required libraries not installed. Run: pip install scikit-optimize scikit-learn")
        
        self.source_file_path = source_file_path
        self.test_input = test_input
        self.best_config = None
        self.best_time = float('inf')
        self.evaluation_count = 0
        
        # Data storage
        self.configurations = []  # List of binary arrays
        self.execution_times = []  # Corresponding execution times
        self.flag_importance = {}  # Flag importance scores
        
        # Detect compiler
        if source_file_path.endswith('.c'):
            self.compiler = 'gcc'
        elif source_file_path.endswith('.cpp') or source_file_path.endswith('.cc'):
            self.compiler = 'g++'
        else:
            raise ValueError("Unsupported file type. Please provide a .c or .cpp file.")
        
        print("="*70)
        print("🔬 HYBRID BAYESIAN-RANDOM FOREST OPTIMIZER (HBRF)")
        print("="*70)
        print(f"Compiler: {self.compiler.upper()}")
        print(f"Source: {source_file_path}")
        print(f"Total flags in search space: {len(GCC_FLAGS)}")
        print(f"\nOptimization Strategy:")
        print(f"  Phase 1: Random sampling ({INITIAL_RANDOM_SAMPLES} configurations)")
        print(f"  Phase 2: Feature importance analysis (top {TOP_FLAGS_COUNT} flags)")
        print(f"  Phase 3: Bayesian optimization ({BAYESIAN_ITERATIONS} iterations)")
        print(f"  Phase 4: Greedy refinement")
        print("-"*70)
    
    def _get_flags_string(self, config):
        """Convert binary config to flag string"""
        return " ".join([GCC_FLAGS[i] for i, val in enumerate(config) if val == 1])
    
    def evaluate_configuration(self, config):
        """Compile and execute with given configuration, return execution time"""
        self.evaluation_count += 1
        flags_str = self._get_flags_string(config)
        output_binary = f"temp_hbrf_{os.getpid()}_{self.evaluation_count}"
        compile_command = f"{self.compiler} -O3 {self.source_file_path} {flags_str} -o {output_binary} 2>&1"
        
        try:
            # Compilation
            result = subprocess.run(
                compile_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=COMPILATION_TIMEOUT
            )
            
            if result.returncode != 0:
                return float('inf')
            
            # Execution
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
            
            # Update best
            if exec_time < self.best_time:
                self.best_time = exec_time
                self.best_config = config.copy()
            
            return exec_time
            
        except subprocess.TimeoutExpired:
            return float('inf')
        except Exception as e:
            return float('inf')
        finally:
            if os.path.exists(output_binary):
                try:
                    os.remove(output_binary)
                except:
                    pass
    
    def phase1_random_sampling(self):
        """Phase 1: Collect random samples to train RF model"""
        print("\n" + "="*70)
        print("📊 PHASE 1: Random Sampling")
        print("="*70)
        
        for i in range(INITIAL_RANDOM_SAMPLES):
            # Generate random configuration
            config = [random.randint(0, 1) for _ in range(len(GCC_FLAGS))]
            
            print(f"Sampling {i+1}/{INITIAL_RANDOM_SAMPLES}...", end='\r')
            
            exec_time = self.evaluate_configuration(config)
            
            if exec_time != float('inf'):
                self.configurations.append(config)
                self.execution_times.append(exec_time)
        
        print(f"\nCollected {len(self.configurations)} valid samples")
        print(f"Best time so far: {self.best_time:.6f}s")
        print(f"Average time: {np.mean(self.execution_times):.6f}s")
    
    def phase2_feature_importance(self):
        """Phase 2: Train RF and identify most important flags"""
        print("\n" + "="*70)
        print("🌲 PHASE 2: Random Forest Feature Importance Analysis")
        print("="*70)
        
        if len(self.configurations) < 10:
            print("⚠ Insufficient data for RF training, using all flags")
            return list(range(len(GCC_FLAGS)))
        
        X = np.array(self.configurations)
        y = np.array(self.execution_times)
        
        # Train Random Forest
        print("Training Random Forest model...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importance
        importances = rf.feature_importances_
        
        # Rank flags by importance
        flag_rankings = sorted(
            enumerate(importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n📈 Top {TOP_FLAGS_COUNT} Most Important Flags:")
        print("-"*70)
        for rank, (idx, importance) in enumerate(flag_rankings[:TOP_FLAGS_COUNT], 1):
            self.flag_importance[idx] = importance
            print(f"  {rank:2d}. {GCC_FLAGS[idx]:40s} | Importance: {importance:.4f}")
        
        # Return indices of top flags
        top_indices = [idx for idx, _ in flag_rankings[:TOP_FLAGS_COUNT]]
        
        print(f"\n✓ Reduced search space from 2^{len(GCC_FLAGS)} to 2^{TOP_FLAGS_COUNT}")
        print(f"  Reduction factor: {2**len(GCC_FLAGS) / 2**TOP_FLAGS_COUNT:.2e}")
        
        return top_indices
    
    def phase3_bayesian_optimization(self, top_flag_indices):
        """Phase 3: Bayesian Optimization on reduced space"""
        print("\n" + "="*70)
        print("🎯 PHASE 3: Bayesian Optimization on Reduced Space")
        print("="*70)
        
        # Define search space (only top flags)
        space = [Integer(0, 1, name=f'flag_{i}') for i in range(len(top_flag_indices))]
        
        # Objective function for Bayesian Optimization
        @use_named_args(space)
        def objective(**params):
            # Create full configuration
            config = [0] * len(GCC_FLAGS)
            for i, flag_idx in enumerate(top_flag_indices):
                config[flag_idx] = params[f'flag_{i}']
            
            exec_time = self.evaluate_configuration(config)
            
            print(f"BO Iteration {self.evaluation_count - INITIAL_RANDOM_SAMPLES}: {exec_time:.6f}s (Best: {self.best_time:.6f}s)", end='\r')
            
            return exec_time
        
        # Run Bayesian Optimization
        print(f"Running Bayesian Optimization for {BAYESIAN_ITERATIONS} iterations...")
        result = gp_minimize(
            objective,
            space,
            n_calls=BAYESIAN_ITERATIONS,
            random_state=42,
            verbose=False,
            n_jobs=1
        )
        
        print(f"\n✓ Bayesian Optimization complete")
        print(f"  Best time found: {self.best_time:.6f}s")
        print(f"  Total evaluations: {self.evaluation_count}")
    
    def phase4_greedy_refinement(self, top_flag_indices):
        """Phase 4: Greedy addition of remaining flags"""
        print("\n" + "="*70)
        print("🔍 PHASE 4: Greedy Refinement")
        print("="*70)
        
        # Get flags not in top set
        remaining_flags = [i for i in range(len(GCC_FLAGS)) if i not in top_flag_indices]
        random.shuffle(remaining_flags)
        
        improvements = 0
        for flag_idx in remaining_flags[:FINAL_GREEDY_ADDITIONS]:
            # Try adding this flag
            test_config = self.best_config.copy()
            test_config[flag_idx] = 1
            
            exec_time = self.evaluate_configuration(test_config)
            
            if exec_time < self.best_time:
                print(f"  ✓ Adding {GCC_FLAGS[flag_idx]}: {self.best_time:.6f}s → {exec_time:.6f}s")
                improvements += 1
            else:
                # Also try removing if it was already there
                test_config[flag_idx] = 0
                exec_time = self.evaluate_configuration(test_config)
                if exec_time < self.best_time:
                    print(f"  ✓ Removing {GCC_FLAGS[flag_idx]}: improved to {exec_time:.6f}s")
                    improvements += 1
        
        print(f"\n✓ Greedy refinement complete: {improvements} improvements found")
    
    def run(self):
        """Execute the complete HBRF optimization pipeline"""
        start_time = time.time()
        
        try:
            # Phase 1: Random Sampling
            self.phase1_random_sampling()
            
            # Phase 2: Feature Importance
            top_flag_indices = self.phase2_feature_importance()
            
            # Phase 3: Bayesian Optimization
            self.phase3_bayesian_optimization(top_flag_indices)
            
            # Phase 4: Greedy Refinement
            self.phase4_greedy_refinement(top_flag_indices)
            
        except KeyboardInterrupt:
            print("\n\n⚠ Optimization interrupted by user!")
        
        total_time = time.time() - start_time
        
        # Print final results
        self.print_results(total_time)
    
    def print_results(self, total_time):
        """Print final optimization results"""
        print("\n" + "="*70)
        print("✅ HBRF OPTIMIZATION COMPLETE")
        print("="*70)
        
        if self.best_config and self.best_time != float('inf'):
            print(f"\n🏆 Best Execution Time: {self.best_time:.6f} seconds")
            print(f"⏱️  Total Optimization Time: {total_time:.2f} seconds")
            print(f"📊 Total Evaluations: {self.evaluation_count}")
            print(f"📈 Evaluations per second: {self.evaluation_count/total_time:.2f}")
            
            enabled_flags = [GCC_FLAGS[i] for i, val in enumerate(self.best_config) if val == 1]
            
            print(f"\n📋 Enabled Flags ({len(enabled_flags)} total):")
            for i, flag in enumerate(enabled_flags, 1):
                importance = self.flag_importance.get(GCC_FLAGS.index(flag), 0.0)
                if importance > 0:
                    print(f"  {i:2d}. {flag:45s} | Importance: {importance:.4f}")
                else:
                    print(f"  {i:2d}. {flag}")
            
            best_flags = self._get_flags_string(self.best_config)
            print(f"\n💻 Optimal Compilation Command:")
            print(f"{self.compiler} -O3 {self.source_file_path} {best_flags} -o optimized_hbrf")
            
            # Create optimized binary
            print("\n🔨 Creating optimized binary...")
            compile_cmd = f"{self.compiler} -O3 {self.source_file_path} {best_flags} -o optimized_hbrf"
            result = subprocess.run(compile_cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                print("✅ Optimized binary created: ./optimized_hbrf")
            else:
                print("⚠ Failed to create optimized binary")
            
            # Save results to JSON
            results = {
                'best_time': self.best_time,
                'total_optimization_time': total_time,
                'total_evaluations': self.evaluation_count,
                'enabled_flags': enabled_flags,
                'flag_importance': {GCC_FLAGS[k]: v for k, v in self.flag_importance.items()}
            }
            
            with open('hbrf_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("📄 Results saved to: hbrf_results.json")
            
        else:
            print("\n❌ Could not find a valid solution.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hbrf_optimizer.py <path_to_c_or_cpp_file> [test_input_file]")
        print("Example: python hbrf_optimizer.py matrix_multiply.c")
        sys.exit(1)
    
    source_file = sys.argv[1]
    test_input = None
    
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                test_input = f.read()
            print(f"Using test input from: {input_file}")
    
    try:
        optimizer = HBRFOptimizer(source_file, test_input)
        optimizer.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
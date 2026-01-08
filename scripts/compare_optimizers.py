#!/usr/bin/env python3
"""
Comparison script for AutoFlag vs HBRF optimization approaches
Runs both optimizers and provides detailed comparative analysis
"""

import subprocess
import time
import json
import sys
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class OptimizerComparison:
    def __init__(self, source_file, test_input_file=None):
        self.source_file = source_file
        self.test_input_file = test_input_file
        self.results = {
            'AutoFlag': {},
            'HBRF': {},
            'baseline': {}
        }

    def run_baseline_benchmarks(self):
        """Run baseline -O1, -O2, -O3 benchmarks"""
        print("=" * 80)
        print("🎯 RUNNING BASELINE BENCHMARKS (-O1, -O2, -O3)")
        print("=" * 80)

        compiler = 'gcc' if self.source_file.endswith('.c') else 'g++'
        baseline_times = {}

        for opt_level in ['-O1', '-O2', '-O3']:
            binary = f"baseline{opt_level}"
            compile_cmd = f"{compiler} {opt_level} {self.source_file} -o {binary}"

            try:
                result = subprocess.run(compile_cmd, shell=True, capture_output=True, timeout=30)
                if result.returncode != 0:
                    print(f"  {opt_level}: Compilation failed")
                    baseline_times[opt_level] = float('inf')
                    continue

                test_input = None
                if self.test_input_file and os.path.exists(self.test_input_file):
                    with open(self.test_input_file, 'r') as f:
                        test_input = f.read()

                start = time.time()
                exec_result = subprocess.run(
                    f"./{binary}",
                    shell=True,
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                exec_time = time.time() - start

                if exec_result.returncode == 0:
                    baseline_times[opt_level] = exec_time
                    print(f"  {opt_level}: {exec_time:.6f}s")
                else:
                    baseline_times[opt_level] = float('inf')
                    print(f"  {opt_level}: Execution failed")

                if os.path.exists(binary):
                    os.remove(binary)

            except Exception as e:
                print(f"  {opt_level}: Error - {e}")
                baseline_times[opt_level] = float('inf')

        self.results['baseline'] = baseline_times
        return baseline_times

    def run_autoflag(self):
        """Run AutoFlag optimizer (streaming output live)"""
        print("\n" + "=" * 80)
        print("🧬 RUNNING AutoFlag (Genetic Algorithm)")
        print("=" * 80)

        cmd = ['python3', '-u', 'autoflag.py', self.source_file]
        if self.test_input_file:
            cmd.append(self.test_input_file)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            best_time = float('inf')
            evaluations = 0

            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)
                output_lines.append(line)

                if 'Best Execution Time:' in line:
                    try:
                        best_time = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Gen' in line and 'Best:' in line:
                    evaluations += 1

            process.wait(timeout=3600)
            total_time = time.time() - start_time
            output = "\n".join(output_lines)

            self.results['AutoFlag'] = {
                'best_time': best_time,
                'total_time': total_time,
                'evaluations': evaluations,
                'output': output
            }

            print(f"\n✅ AutoFlag Complete: {best_time:.6f}s in {total_time:.2f}s")
            
            # Load detailed results to display requested metrics
            if os.path.exists('autoflag_results.json'):
                try:
                    with open('autoflag_results.json', 'r') as f:
                        autoflag_data = json.load(f)
                        print(f"BEST EXECUTION TIME: {autoflag_data.get('best_time', 'N/A')}")
                        print(f"TOTAL EVALUATIONS: {autoflag_data.get('total_evaluations', 'N/A')}")
                        print(f"ENABLED FLAGS: {json.dumps(autoflag_data.get('enabled_flags', []), indent=2)}")
                except Exception as e:
                    print(f"Could not read detailed AutoFlag results: {e}")

            return best_time, total_time

        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ AutoFlag timed out after 1 hour")
            return float('inf'), 3600
        except Exception as e:
            print(f"❌ AutoFlag failed: {e}")
            return float('inf'), 0

    def run_hbrf(self):
        """Run HBRF optimizer (streaming output live)"""
        print("\n" + "=" * 80)
        print("🔬 RUNNING HBRF (Hybrid Bayesian-RF)")
        print("=" * 80)

        cmd = ['python3', '-u', 'hbrf_optimizer.py', self.source_file]
        if self.test_input_file:
            cmd.append(self.test_input_file)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            best_time = float('inf')
            evaluations = 0

            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)
                output_lines.append(line)

                if 'Best Execution Time:' in line:
                    try:
                        best_time = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Total Evaluations:' in line:
                    try:
                        evaluations = int(line.split(':')[1].strip())
                    except:
                        pass

            process.wait(timeout=3600)
            total_time = time.time() - start_time
            output = "\n".join(output_lines)

            if os.path.exists('hbrf_results.json'):
                with open('hbrf_results.json', 'r') as f:
                    hbrf_data = json.load(f)
                    best_time = hbrf_data.get('best_time', best_time)
                    evaluations = hbrf_data.get('total_evaluations', evaluations)

            self.results['HBRF'] = {
                'best_time': best_time,
                'total_time': total_time,
                'evaluations': evaluations,
                'output': output
            }

            print(f"\n✅ HBRF Complete: {best_time:.6f}s in {total_time:.2f}s")
            
            # Display requested metrics
            if os.path.exists('hbrf_results.json'):
                try:
                    # We already loaded hbrf_data above if it existed, but let's reload or use what we have if we kept it.
                    # The code above loaded it into local vars but didn't keep the dict. Let's reload to be safe and clean.
                    with open('hbrf_results.json', 'r') as f:
                        hbrf_data = json.load(f)
                        print(f"BEST EXECUTION TIME: {hbrf_data.get('best_time', 'N/A')}")
                        print(f"TOTAL EVALUATIONS: {hbrf_data.get('total_evaluations', 'N/A')}")
                        print(f"ENABLED FLAGS: {json.dumps(hbrf_data.get('enabled_flags', []), indent=2)}")
                except Exception as e:
                    print(f"Could not read detailed HBRF results: {e}")
            
            return best_time, total_time

        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ HBRF timed out after 1 hour")
            return float('inf'), 3600
        except Exception as e:
            print(f"❌ HBRF failed: {e}")
            return float('inf'), 0

    def run_xgboost(self):
        """Run XGBoost-based optimizer"""
        print("\n" + "=" * 80)
        print("⚡ RUNNING XGBOOST OPTIMIZER")
        print("=" * 80)

        cmd = ['python3', '-u', 'xgboost_optimizer.py', self.source_file]
        if self.test_input_file:
            cmd.append(self.test_input_file)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)
                output_lines.append(line)

            process.wait(timeout=3600)
            total_time = time.time() - start_time
            output = "\n".join(output_lines)

            best_time = float('inf')
            evaluations = 0

            if os.path.exists('xgboost_results.json'):
                try:
                    with open('xgboost_results.json', 'r') as f:
                        xgb_data = json.load(f)
                        best_time = xgb_data.get('best_time', best_time)
                        evaluations = xgb_data.get('total_evaluations', evaluations)
                except Exception as e:
                    print(f"Warning: could not read xgboost_results.json: {e}")

            self.results['XGBOOST'] = {
                'best_time': best_time,
                'total_time': total_time,
                'evaluations': evaluations,
                'output': output
            }

            print(f"\n✅ XGBOOST Complete: {best_time:.6f}s in {total_time:.2f}s")
            
            # Display requested metrics
            if os.path.exists('xgboost_results.json'):
                try:
                    with open('xgboost_results.json', 'r') as f:
                        xgb_data = json.load(f)
                        print(f"BEST EXECUTION TIME: {xgb_data.get('best_time', 'N/A')}")
                        print(f"TOTAL EVALUATIONS: {xgb_data.get('total_evaluations', 'N/A')}")
                        print(f"ENABLED FLAGS: {json.dumps(xgb_data.get('enabled_flags', []), indent=2)}")
                except Exception as e:
                    print(f"Could not read detailed XGBoost results: {e}")
            
            return best_time, total_time

        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ XGBOOST timed out after 1 hour")
            return float('inf'), 3600
        except Exception as e:
            print(f"❌ XGBOOST failed: {e}")
            return float('inf'), 0

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE COMPARISON REPORT")
        print("=" * 80)

        baseline = self.results['baseline']
        autoflag = self.results['AutoFlag']
        hbrf = self.results['HBRF']
        xgboost = self.results['XGBOOST']

        print("\n1️⃣  EXECUTION TIME COMPARISON")
        print("-" * 80)
        print(f"{'Method':<20} | {'Time (s)':<15} | {'Speedup vs -O3':<20} | {'Rank':<10}")
        print("-" * 80)

        o3_time = baseline.get('-O3', float('inf'))

        times_dict = {
            '-O1': baseline.get('-O1', float('inf')),
            '-O2': baseline.get('-O2', float('inf')),
            '-O3': o3_time,
            'AutoFlag': autoflag.get('best_time', float('inf')),
            'HBRF': hbrf.get('best_time', float('inf')),
            'XGBOOST': xgboost.get('best_time', float('inf'))
        }

        ranked = sorted(times_dict.items(), key=lambda x: x[1])

        for rank, (method, exec_time) in enumerate(ranked, 1):
            if exec_time == float('inf'):
                speedup_str = "Failed"
            elif o3_time == float('inf') or o3_time == 0:
                speedup_str = "N/A"
            else:
                speedup = ((o3_time - exec_time) / o3_time) * 100
                speedup_str = f"{speedup:+.2f}%"

            time_str = f"{exec_time:.6f}" if exec_time != float('inf') else "Failed"
            print(f"{method:<20} | {time_str:<15} | {speedup_str:<20} | #{rank}")

        print("\n2️⃣  OPTIMIZATION TIME COMPARISON")
        print("-" * 80)
        print(f"{'Method':<20} | {'Opt. Time (s)':<15} | {'Evaluations':<15} | {'Eval/sec':<10}")
        print("-" * 80)

        autoflag_time = autoflag.get('total_time', 0)
        hbrf_time = hbrf.get('total_time', 0)
        xgb_time = xgboost.get('total_time', 0)
        autoflag_evals = autoflag.get('evaluations', 0)
        hbrf_evals = hbrf.get('evaluations', 0)
        xgb_evals = xgboost.get('evaluations', 0)

        autoflag_eps = autoflag_evals / autoflag_time if autoflag_time > 0 else 0
        hbrf_eps = hbrf_evals / hbrf_time if hbrf_time > 0 else 0
        xgb_eps = xgb_evals / xgb_time if xgb_time > 0 else 0

        print(f"{'AutoFlag':<20} | {autoflag_time:<15.2f} | {autoflag_evals:<15} | {autoflag_eps:<10.2f}")
        print(f"{'HBRF':<20} | {hbrf_time:<15.2f} | {hbrf_evals:<15} | {hbrf_eps:<10.2f}")
        print(f"{'XGBOOST':<20} | {xgb_time:<15.2f} | {xgb_evals:<15} | {xgb_eps:<10.2f}")

        print("\n3️⃣  EFFICIENCY METRICS")
        print("-" * 80)

        if autoflag.get('best_time', float('inf')) != float('inf') and o3_time != float('inf'):
            autoflag_improvement = ((o3_time - autoflag['best_time']) / o3_time) * 100
        else:
            autoflag_improvement = 0

        if hbrf.get('best_time', float('inf')) != float('inf') and o3_time != float('inf'):
            hbrf_improvement = ((o3_time - hbrf['best_time']) / o3_time) * 100
        else:
            hbrf_improvement = 0

        if xgboost.get('best_time', float('inf')) != float('inf') and o3_time != float('inf'):
            xgb_improvement = ((o3_time - xgboost['best_time']) / o3_time) * 100
        else:
            xgb_improvement = 0

        print(f"AutoFlag Improvement over -O3: {autoflag_improvement:+.2f}%")
        print(f"HBRF Improvement over -O3: {hbrf_improvement:+.2f}%")
        print(f"XGBOOST Improvement over -O3: {xgb_improvement:+.2f}%")

        print("\n4️⃣  WINNER ANALYSIS")
        print("-" * 80)

        autoflag_best = autoflag.get('best_time', float('inf'))
        hbrf_best = hbrf.get('best_time', float('inf'))
        xgb_best = xgboost.get('best_time', float('inf'))

        # determine winner among available results
        bests = {'AutoFlag': autoflag_best, 'HBRF': hbrf_best, 'XGBOOST': xgb_best}
        sorted_bests = sorted(bests.items(), key=lambda x: x[1])
        winner_name, winner_time = sorted_bests[0]

        if winner_time == float('inf'):
            winner = "NONE"
            print("No successful optimizations found")
        else:
            winner = winner_name
            print(f"🏆 WINNER: {winner}")
            # compute margin against second best if available
            if len(sorted_bests) > 1 and sorted_bests[1][1] != float('inf') and sorted_bests[1][1] > 0:
                margin = ((sorted_bests[1][1] - winner_time) / sorted_bests[1][1]) * 100
                print(f"   {winner} achieved {margin:.2f}% better execution time vs second best")

        # Load detailed results from JSON files
        detailed_results = {}
        
        for method, filename in [('AutoFlag', 'autoflag_results.json'), ('HBRF', 'hbrf_results.json'), ('XGBOOST', 'xgboost_results.json')]:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        # Ensure the format matches the user requirement
                        detailed_results[method] = {
                            "best_time": data.get('best_time', float('inf')),
                            "total_evaluations": data.get('total_evaluations', 0),
                            "enabled_flags": data.get('enabled_flags', [])
                        }
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")
                    detailed_results[method] = {
                        "best_time": float('inf'),
                        "total_evaluations": 0,
                        "enabled_flags": []
                    }
            else:
                detailed_results[method] = {
                    "best_time": float('inf'),
                    "total_evaluations": 0,
                    "enabled_flags": []
                }

        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'source_file': self.source_file,
            'baseline': baseline,
            'AutoFlag': detailed_results['AutoFlag'],
            'HBRF': detailed_results['HBRF'],
            'XGBOOST': detailed_results['XGBOOST'],
            'winner': winner,
            'improvements': {
                'AutoFlag_vs_O3': autoflag_improvement,
                'HBRF_vs_O3': hbrf_improvement,
                'XGBOOST_vs_O3': xgb_improvement
            }
        }

        with open('comparison_results.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print("\n📄 Comparison results saved to: comparison_results.json")
        
        print("\n" + "=" * 80)
        print("📝 DETAILED METRICS")
        print("=" * 80)

        for method in ['AutoFlag', 'HBRF', 'XGBOOST']:
            print(f"\n--- {method} ---")
            
            # Get data from detailed_results (loaded from JSON)
            d_res = detailed_results.get(method, {})
            best_time = d_res.get('best_time', 'N/A')
            evals = d_res.get('total_evaluations', 'N/A')
            flags = d_res.get('enabled_flags', [])
            
            # Get optimization time from self.results (measured by this script)
            opt_time = self.results.get(method, {}).get('total_time', 'N/A')
            if isinstance(opt_time, (int, float)):
                opt_time = f"{opt_time:.4f}s"

            print(f"BEST EXECUTION TIME: {best_time}")
            print(f"OPTIMIZATION TIME: {opt_time}")
            print(f"TOTAL EVALUATIONS: {evals}")
            print(f"ENABLED FLAGS: {json.dumps(flags, indent=2)}")

        print("\n" + "=" * 80)
        
        # Also print the detailed results in the requested format for verification/parsing
        print("\n📋 Detailed Results (JSON Format):")
        print(json.dumps(detailed_results, indent=2))

        self.generate_visualizations(times_dict, autoflag_time, hbrf_time, xgb_time)

    def generate_visualizations(self, times_dict, autoflag_opt_time, hbrf_opt_time, xgb_opt_time):
        """Generate comparison charts"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Execution Time Comparison
            methods = list(times_dict.keys())
            times = [times_dict[m] if times_dict[m] != float('inf') else 0 for m in methods]
            # extend color palette to accommodate extra method
            base_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
            colors = base_colors[:len(methods)]

            axes[0, 0].bar(methods, times, color=colors)
            axes[0, 0].set_ylabel('Execution Time (seconds)')
            axes[0, 0].set_title('Execution Time Comparison')
            axes[0, 0].grid(axis='y', alpha=0.3)

            # Optimization Time Comparison
            opt_methods = ['AutoFlag', 'HBRF', 'XGBOOST']
            opt_times = [autoflag_opt_time, hbrf_opt_time, xgb_opt_time]

            axes[0, 1].bar(opt_methods, opt_times, color=['#e74c3c', '#9b59b6', '#1abc9c'])
            axes[0, 1].set_ylabel('Optimization Time (seconds)')
            axes[0, 1].set_title('Optimization Time Comparison')
            axes[0, 1].grid(axis='y', alpha=0.3)

            # Percentage Improvements Over -O3
            o3_time = times_dict.get('-O3', float('inf'))
            improvements = {}
            for key in ['AutoFlag', 'HBRF', 'XGBOOST']:
                val = times_dict.get(key, float('inf'))
                improvements[key] = ((o3_time - val) / o3_time) * 100 if o3_time != float('inf') and val != float('inf') else 0

            axes[1, 0].bar(list(improvements.keys()), list(improvements.values()), color=['#e74c3c', '#9b59b6', '#1abc9c'])
            axes[1, 0].set_ylabel('Improvement (%)')
            axes[1, 0].set_title('Percentage Improvement Over -O3')
            axes[1, 0].grid(axis='y', alpha=0.3)

            # Evaluations Per Second
            autoflag_evals = self.results['AutoFlag'].get('evaluations', 0)
            hbrf_evals = self.results['HBRF'].get('evaluations', 0)
            xgb_evals = self.results['XGBOOST'].get('evaluations', 0)
            autoflag_eps = autoflag_evals / autoflag_opt_time if autoflag_opt_time > 0 else 0
            hbrf_eps = hbrf_evals / hbrf_opt_time if hbrf_opt_time > 0 else 0
            xgb_eps = xgb_evals / xgb_opt_time if xgb_opt_time > 0 else 0

            axes[1, 1].bar(['AutoFlag', 'HBRF', 'XGBOOST'], [autoflag_eps, hbrf_eps, xgb_eps], color=['#e74c3c', '#9b59b6', '#1abc9c'])
            axes[1, 1].set_ylabel('Evaluations Per Second')
            axes[1, 1].set_title('Evaluation Efficiency')
            axes[1, 1].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig('enhanced_comparison_chart.png', dpi=150, bbox_inches='tight')
            print("📊 Enhanced visualization saved to: enhanced_comparison_chart.png")

        except Exception as e:
            print(f"⚠ Could not generate visualizations: {e}")

    def run_full_comparison(self):
        """Run complete comparison pipeline"""
        print("=" * 80)
        print("🚀 STARTING COMPREHENSIVE OPTIMIZATION COMPARISON")
        print("=" * 80)
        print(f"Source file: {self.source_file}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        self.run_baseline_benchmarks()
        self.run_autoflag()
        self.run_hbrf()
        self.run_xgboost()
        self.generate_comparison_report()

        print("\n" + "=" * 80)
        print("✅ COMPARISON COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_optimizers.py <source_file> [test_input_file]")
        sys.exit(1)

    source_file = sys.argv[1]
    test_input = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(source_file):
        print(f"❌ Error: Source file '{source_file}' not found")
        sys.exit(1)

    comparison = OptimizerComparison(source_file, test_input)
    comparison.run_full_comparison()

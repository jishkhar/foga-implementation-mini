import subprocess
import time
import random
import os
import sys
import signal
import json
from contextlib import contextmanager

# --- Configuration based on FOGA Paper (Table II) ---

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

# Tuned Genetic Algorithm Parameters from the paper
POPULATION_SIZE = 277
MAX_GENERATIONS = 10
ELITISM_RATIO = 0.147
MUTATION_PROB = 0.287
CROSSOVER_PROB = 0.120
MAX_ITER_NO_IMPROVEMENT = 45

# Execution timeout in seconds (adjust based on your program)
EXECUTION_TIMEOUT = 10
COMPILATION_TIMEOUT = 30

@contextmanager
def timeout(duration):
    """Context manager for timeout operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class Individual:
    """Represents a single solution (a set of compiler flags)."""
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = float('inf')
        self.compilation_success = False

    @classmethod
    def create_random(cls):
        """Creates an individual with a random set of flags."""
        chromosome = [random.randint(0, 1) for _ in range(len(GCC_FLAGS))]
        return cls(chromosome)

class FOGA:
    """The main class to run the Flag Optimization with Genetic Algorithm."""

    def __init__(self, source_file_path, test_input=None):
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"Source file not found: {source_file_path}")
        
        self.source_file_path = source_file_path
        self.test_input = test_input  # Optional input for the program
        self.population = [Individual.create_random() for _ in range(POPULATION_SIZE)]
        self.best_individual = None
        self.compilation_errors_shown = 0
        self.generation_stats = []
        
        # Detect compiler
        if source_file_path.endswith('.c'):
            self.compiler = 'gcc'
        elif source_file_path.endswith('.cpp') or source_file_path.endswith('.cc'):
            self.compiler = 'g++'
        else:
            raise ValueError("Unsupported file type. Please provide a .c or .cpp file.")
        
        print(f"Detected {self.compiler.upper()} source file. Using '{self.compiler}' compiler.")
        print(f"Population size: {POPULATION_SIZE}")
        print(f"Max generations: {MAX_GENERATIONS}")
        print(f"Execution timeout: {EXECUTION_TIMEOUT}s per test")
        print("-" * 60)

    def _get_flags_from_chromosome(self, chromosome):
        """Converts a binary chromosome to a string of GCC flags."""
        return " ".join([GCC_FLAGS[i] for i, gene in enumerate(chromosome) if gene == 1])

    def calculate_fitness(self, individual):
        """
        Fitness Function: Compiles and runs the code, returning the execution time.
        Includes timeout handling for both compilation and execution.
        """
        flags_str = self._get_flags_from_chromosome(individual.chromosome)
        output_binary = f"temp_binary_{os.getpid()}_{id(individual)}"
        compile_command = f"{self.compiler} -O3 {self.source_file_path} {flags_str} -o {output_binary} 2>&1"

        try:
            # Compilation with timeout
            result = subprocess.run(
                compile_command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=COMPILATION_TIMEOUT
            )
            
            if result.returncode != 0:
                individual.fitness = float('inf')
                individual.compilation_success = False
                if self.compilation_errors_shown < 3:
                    print(f"\n⚠ Compilation failed with flags: {flags_str[:50]}...")
                    self.compilation_errors_shown += 1
                return
            
            individual.compilation_success = True
            
            # Execution with timeout
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
                individual.fitness = float('inf')
            else:
                individual.fitness = time.time() - start_time
                
        except subprocess.TimeoutExpired:
            individual.fitness = float('inf')
            individual.compilation_success = False
            
        except Exception as e:
            individual.fitness = float('inf')
            individual.compilation_success = False
            
        finally:
            # Cleanup
            if os.path.exists(output_binary):
                try:
                    os.remove(output_binary)
                except:
                    pass

    def _selection(self):
        """Selects parents using Linear Ranking Selection."""
        # Filter only successfully compiled individuals
        valid_population = [ind for ind in self.population if ind.fitness != float('inf')]
        
        if len(valid_population) < 2:
            # If we don't have enough valid individuals, use the whole population
            valid_population = self.population
        
        valid_population.sort(key=lambda ind: ind.fitness)
        ranks = list(range(len(valid_population)))
        inverted_ranks = [(len(ranks) - 1 - r) for r in ranks]
        total_inverted_ranks = sum(inverted_ranks)
        
        if total_inverted_ranks == 0:
            probabilities = [1/len(valid_population)] * len(valid_population)
        else:
            probabilities = [r / total_inverted_ranks for r in inverted_ranks]
            
        return random.choices(valid_population, weights=probabilities, k=2)

    def _crossover(self, parent1, parent2):
        """Performs Segment-Based Crossover."""
        if random.random() >= CROSSOVER_PROB:
            return Individual(parent1.chromosome[:]), Individual(parent2.chromosome[:])

        child1_chromo, child2_chromo = parent1.chromosome[:], parent2.chromosome[:]
        size = len(child1_chromo)
        
        if size > 1:
            start, end = sorted([random.randrange(size) for _ in range(2)])
            child1_chromo[start:end] = parent2.chromosome[start:end]
            child2_chromo[start:end] = parent1.chromosome[start:end]
            
        return Individual(child1_chromo), Individual(child2_chromo)

    def _mutate(self, individual):
        """Performs bit-flip mutation."""
        for i in range(len(individual.chromosome)):
            if random.random() < MUTATION_PROB:
                individual.chromosome[i] = 1 - individual.chromosome[i]

    def print_statistics(self, generation):
        """Prints generation statistics."""
        valid_count = sum(1 for ind in self.population if ind.fitness != float('inf'))
        avg_fitness = sum(ind.fitness for ind in self.population if ind.fitness != float('inf'))
        
        if valid_count > 0:
            avg_fitness /= valid_count
        else:
            avg_fitness = float('inf')
            
        print(f"Gen {generation+1:3d} | Valid: {valid_count:3d}/{POPULATION_SIZE} | "
              f"Best: {self.best_individual.fitness:.8f}s | Avg: {avg_fitness:.4f}s")

    def run(self):
        """Executes the main GA loop."""
        print("🚀 Starting FOGA optimization...")
        print("Press Ctrl+C to stop early and get the best result so far\n")
        
        generations_without_improvement = 0
        
        try:
            for generation in range(MAX_GENERATIONS):
                # Evaluate fitness for all individuals
                print(f"Evaluating generation {generation+1}...", end='\r')
                
                for i, ind in enumerate(self.population):
                    if ind.fitness == float('inf'):  # Only evaluate if not already evaluated
                        self.calculate_fitness(ind)
                    
                    # Show progress
                    if i % 10 == 0:
                        print(f"Evaluating generation {generation+1}... [{i}/{POPULATION_SIZE}]", end='\r')
                
                # Sort by fitness
                self.population.sort(key=lambda ind: ind.fitness)
                current_best = self.population[0]
                
                # Update best individual
                if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
                    self.best_individual = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Print statistics
                self.print_statistics(generation)
                
                # Check for early stopping
                if generations_without_improvement >= MAX_ITER_NO_IMPROVEMENT:
                    print(f"\n📊 Terminating early: No improvement for {MAX_ITER_NO_IMPROVEMENT} generations.")
                    break
                
                # Generate next generation
                next_generation = []
                
                # Elitism
                elitism_count = int(POPULATION_SIZE * ELITISM_RATIO)
                next_generation.extend(self.population[:elitism_count])
                
                # Create new individuals through crossover and mutation
                while len(next_generation) < POPULATION_SIZE:
                    parents = self._selection()
                    child1, child2 = self._crossover(parents[0], parents[1])
                    self._mutate(child1)
                    self._mutate(child2)
                    next_generation.append(child1)
                    if len(next_generation) < POPULATION_SIZE:
                        next_generation.append(child2)
                
                self.population = next_generation
                
        except KeyboardInterrupt:
            print("\n\n⚠ Optimization interrupted by user!")
        
        self.print_results()

    def _get_optimization_flags(self, level):
        """Gets the flags enabled by a specific optimization level using compiler query."""
        temp_file = f"temp_query_{os.getpid()}.c"
        
        # Create a minimal source file
        with open(temp_file, 'w') as f:
            f.write("int main() { return 0; }")
        
        try:
            # Use -Q --help=optimizers to get enabled flags
            result = subprocess.run(
                f"{self.compiler} {level} -Q --help=optimizers",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            enabled_flags = []
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('-f') and '[enabled]' in line:
                        flag = line.split()[0]
                        enabled_flags.append(flag)
            
            return enabled_flags
            
        except Exception as e:
            return []
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
    def _benchmark_and_compare(self):
        """Benchmarks standard GCC/G++ optimization levels and compares with FOGA's result."""
        print("\n" + "-"*60)
        print("📊 BENCHMARKING COMPARISON")
        print("-"*60)
        
        benchmark_results = {}
        base_levels = ['-O1', '-O2', '-O3']
        
        print("Running benchmarks for standard optimization levels...")

        for level in base_levels:
            output_binary = f"temp_binary_benchmark_{level[1:]}"
            compile_command = f"{self.compiler} {level} {self.source_file_path} -o {output_binary}"
            
            try:
                # Compile
                compile_proc = subprocess.run(compile_command, shell=True, capture_output=True, timeout=COMPILATION_TIMEOUT)
                if compile_proc.returncode != 0:
                    print(f"  - {level}: Compilation Failed")
                    benchmark_results[level] = float('inf')
                    continue

                # Execute and Time
                start_time = time.time()
                # Give a bit more time for non-optimized versions
                exec_timeout = EXECUTION_TIMEOUT * 2 if level in ['-O0', '-O1'] else EXECUTION_TIMEOUT
                exec_proc = subprocess.run(
                    f"./{output_binary}",
                    shell=True,
                    input=self.test_input,
                    capture_output=True,
                    text=True,
                    timeout=exec_timeout
                )
                
                if exec_proc.returncode != 0:
                     print(f"  - {level}: Execution Failed")
                     benchmark_results[level] = float('inf')
                else:
                    exec_time = time.time() - start_time
                    benchmark_results[level] = exec_time
                    print(f"  - {level}: {exec_time:.6f}s")

            except subprocess.TimeoutExpired:
                print(f"  - {level}: Timed out")
                benchmark_results[level] = float('inf')
            finally:
                if os.path.exists(output_binary):
                    os.remove(output_binary)

        # Prepare data for the table
        foga_time = self.best_individual.fitness
        o3_time = benchmark_results.get('-O3', float('inf'))

        # Helper for calculating improvement
        def calc_improvement(baseline, new_time):
            if baseline == float('inf') or new_time == float('inf') or baseline == 0:
                return "N/A"
            improvement = ((baseline - new_time) / baseline) * 100
            return f"{improvement:+.2f}%"

        # Print the final table
        print("\n" + "="*60)
        print(f"| {'Optimization':<15} | {'Execution Time (s)':<20} | {'Improvement vs -O3':<22} |")
        print(f"|{'-'*17}|{'-'*22}|{'-'*24}|")
        
        for level in base_levels:
            time_val = benchmark_results.get(level, float('inf'))
            time_str = f"{time_val:.6f}" if time_val != float('inf') else "Failed"
            improvement_str = calc_improvement(o3_time, time_val)
            print(f"| {level:<15} | {time_str:<20} | {improvement_str:<22} |")

        foga_time_str = f"{foga_time:.6f}" if foga_time != float('inf') else "Failed"
        foga_improvement_str = calc_improvement(o3_time, foga_time)
        print(f"| {'FOGA (Best)':<15} | {foga_time_str:<20} | {foga_improvement_str:<22} |")
        print("="*60)
        
        # Display flags comparison
        self._print_flags_comparison()

    def _print_flags_comparison(self):
        """Prints a detailed comparison of flags used in each optimization level."""
        print("\n" + "="*80)
        print("🚩 OPTIMIZATION FLAGS COMPARISON")
        print("="*80)
        
        print("\nQuerying compiler for optimization flags...")
        
        # Get flags for each level
        o1_flags = set(self._get_optimization_flags('-O1'))
        o2_flags = set(self._get_optimization_flags('-O2'))
        o3_flags = set(self._get_optimization_flags('-O3'))
        foga_flags = set([GCC_FLAGS[i] for i, gene in enumerate(self.best_individual.chromosome) if gene == 1])
        
        print(f"\n📊 Flag Count Summary:")
        print(f"  -O1:        {len(o1_flags)} flags enabled")
        print(f"  -O2:        {len(o2_flags)} flags enabled")
        print(f"  -O3:        {len(o3_flags)} flags enabled")
        print(f"  FOGA:       {len(foga_flags)} flags enabled")
        
        # Find unique flags
        all_flags = o1_flags | o2_flags | o3_flags | foga_flags
        
        print(f"\n📋 Detailed Flag Breakdown:")
        print("-"*80)
        print(f"{'Flag':<45} | {'O1':<5} | {'O2':<5} | {'O3':<5} | {'FOGA':<5}")
        print("-"*80)
        
        # Sort flags alphabetically for easier reading
        for flag in sorted(all_flags):
            o1_mark = "✓" if flag in o1_flags else "✗"
            o2_mark = "✓" if flag in o2_flags else "✗"
            o3_mark = "✓" if flag in o3_flags else "✗"
            foga_mark = "✓" if flag in foga_flags else "✗"
            
            print(f"{flag:<45} | {o1_mark:^5} | {o2_mark:^5} | {o3_mark:^5} | {foga_mark:^5}")
        
        print("-"*80)
        
        # Show FOGA-unique flags
        foga_unique = foga_flags - o3_flags
        o3_missing = o3_flags - foga_flags
        
        if foga_unique:
            print(f"\n✨ FOGA-Specific Flags (not in -O3): {len(foga_unique)}")
            for i, flag in enumerate(sorted(foga_unique), 1):
                print(f"  {i:2d}. {flag}")
        else:
            print(f"\n✨ FOGA-Specific Flags: None (FOGA uses a subset of -O3)")
        
        if o3_missing:
            print(f"\n🔍 -O3 Flags Excluded by FOGA: {len(o3_missing)}")
            for i, flag in enumerate(sorted(o3_missing), 1):
                print(f"  {i:2d}. {flag}")
        
        print("\n" + "="*80)

    def print_results(self):
        """Prints the final optimization results."""
        print("\n" + "="*60)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*60)
        
        if self.best_individual and self.best_individual.fitness != float('inf'):
            print(f"\n🏆 Best Execution Time: {self.best_individual.fitness:.6f} seconds")
            
            best_flags = self._get_flags_from_chromosome(self.best_individual.chromosome)
            enabled_flags = [GCC_FLAGS[i] for i, gene in enumerate(self.best_individual.chromosome) if gene == 1]
            
            print(f"\n📋 Enabled Flags ({len(enabled_flags)} total):")
            for i, flag in enumerate(enabled_flags, 1):
                print(f"  {i:2d}. {flag}")
            
            print(f"\n💻 Optimal Compilation Command:")
            print(f"{self.compiler} -O3 {self.source_file_path} {best_flags} -o optimized_binary")
            
            # Create optimized binary
            print("\n🔨 Creating optimized binary...")
            compile_cmd = f"{self.compiler} -O3 {self.source_file_path} {best_flags} -o optimized_binary"
            result = subprocess.run(compile_cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                print("✅ Optimized binary created successfully: ./optimized_binary")
            else:
                print("⚠ Failed to create optimized binary")

            # Save results to JSON
            results = {
                'best_time': self.best_individual.fitness,
                'total_evaluations': POPULATION_SIZE * MAX_GENERATIONS, # Approximate or track actual
                'enabled_flags': enabled_flags
            }
            
            with open('foga_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("📄 Results saved to: foga_results.json")

            # Add the comparison table at the end
            self._benchmark_and_compare()
                
        else:
            print("\n❌ Could not find a valid compilable solution.")
            print("     Please check your source code or try with different parameters.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python foga.py <path_to_c_or_cpp_file> [test_input_file]")
        print("Example: python foga.py matrix_multiply.c")
        print("         python foga.py program.c input.txt")
        sys.exit(1)
    
    source_file = sys.argv[1]
    test_input = None
    
    # Load test input if provided
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                test_input = f.read()
            print(f"Using test input from: {input_file}")
    
    try:
        foga_optimizer = FOGA(source_file, test_input)
        foga_optimizer.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
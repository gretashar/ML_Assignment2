# import mlrose_hiive as mlrose
from cmath import inf
from telnetlib import GA
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import time
import matplotlib.pyplot as plt

#Source: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html#what-is-an-optimization-problem
# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

   # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):

                # Check for horizontal, diagonal-up and diagonal-down attacks
                if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):

                    # If no attacks, then increment counter
                    fitness_cnt += 1

    return fitness_cnt

# Initialize custom fitness function object
fitness_queen = mlrose.CustomFitness(queens_max)
num_queens = 100
problem = mlrose.DiscreteOpt(length = num_queens, fitness_fn = fitness_queen, maximize = True, max_val = num_queens)


# Define decay schedule for SA
schedule = mlrose.ExpDecay()

# Define initial state
# init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
init_state = np.random.randint(0, high=num_queens, size=num_queens, dtype=int)

start_SA = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = inf,
                                                      init_state = init_state, random_state = 1, curve=True)
end_SA = time.time()
SA_time = end_SA-start_SA

print("\n*******************************************************\n")

print('Timing for Simulated Annealing for NQueens: {:.6f}'. format(SA_time))
print('Best State for NQueens: ')
print(best_state)
print('Best Fitness for NQueens:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)

# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0, color="r")
plt.grid()
plt.title("Simulated Annealing for N-Queens Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/n_queens_SA.png')  
plt.clf()


fitness_queen = mlrose.CustomFitness(queens_max)
num_queens = 100
problem = mlrose.DiscreteOpt(length = num_queens, fitness_fn = fitness_queen, maximize = True, max_val = num_queens)

init_state = np.random.randint(0, high=num_queens, size=num_queens, dtype=int)

start_HC = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                      max_attempts = 100, max_iters = inf,
                                                      init_state = init_state, random_state = 1, curve=True)
end_HC = time.time()
HC_time = end_HC-start_HC

print("\n*******************************************************\n")

print('Timing for Randomized Hill Climbing for NQueens: {:.6f}'. format(HC_time))
print('Best State for NQueens: ')
print(best_state)
print('Best Fitness for NQueens:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)

# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0, color="orange")
plt.grid()
plt.title("Randomized Hill Climbing for N-Queens Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/n_queens_HC.png')  
plt.clf()





fitness_queen = mlrose.CustomFitness(queens_max)
num_queens = 100
problem = mlrose.DiscreteOpt(length = num_queens, fitness_fn = fitness_queen, maximize = True, max_val = num_queens)

init_state = np.random.randint(0, high=num_queens, size=num_queens, dtype=int)

start_GA = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,
                                                      max_attempts = 100, max_iters = inf,
                                                      random_state = 1, curve=True)
end_GA = time.time()
GA_time = end_GA-start_GA

print("\n*******************************************************\n")

print('Timing for Genetic Algorithm for NQueens: {:.6f}'. format(GA_time))
print('Best State for NQueens: ')
print(best_state)
print('Best Fitness for NQueens:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)

# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0, color="green")
plt.grid()
plt.title("Genetic Algorithm for N-Queens Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/n_queens_GA.png')  
plt.clf()







fitness_queen = mlrose.CustomFitness(queens_max)
num_queens = 100
problem = mlrose.DiscreteOpt(length = num_queens, fitness_fn = fitness_queen, maximize = True, max_val = num_queens)

init_state = np.random.randint(0, high=num_queens, size=num_queens, dtype=int)

start_MIMIC = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                      max_attempts = 100, max_iters = 10000,
                                                      random_state = 1, curve=True, fast_mimic=True)
end_MIMIC = time.time()
MIMIC_time = end_MIMIC-start_MIMIC

print("\n*******************************************************\n")

print('Timing for MIMIC for NQueens: {:.6f}'. format(MIMIC_time))
print('Best State for NQueens: ')
print(best_state)
print('Best Fitness for NQueens:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)

# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0)
plt.grid()
plt.title("MIMIC for N-Queens Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/n_queens_MIMIC.png')  
plt.clf()



def different_inputs():
    num = [8,16,32,64,100]
    
    SA_Fitness = []
    SA_time_list = []
    RHC_Fitness = []
    RHC_time_list = []
    GA_Fitness = []
    GA_time_list = []
    MIMIC_Fitness = []
    MIMIC_time_list = []
    
    for n in num:   
        fitness_queen = mlrose.CustomFitness(queens_max)

        problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness_queen, maximize = True, max_val = n)
                        

        #Simulated Annealing
        schedule = mlrose.ExpDecay()
        start_SA = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 1, curve=True)
        end_SA = time.time()
        SA_time = end_SA-start_SA
        SA_time_list.append(SA_time)
        SA_Fitness.append(best_fitness)


        start_HC = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 1, curve=True)
        end_HC = time.time()
        HC_time = end_HC-start_HC
        RHC_time_list.append(HC_time)
        RHC_Fitness.append(best_fitness)
     
     
        start_GA = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 1, curve=True)
        end_GA = time.time()
        GA_time = end_GA-start_GA    
    
        GA_time_list.append(GA_time)
        GA_Fitness.append(best_fitness)
        
        
        start_MIMIC = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 1, curve=True, fast_mimic=True)
        end_MIMIC = time.time()
        MIMIC_time = end_MIMIC-start_MIMIC      
                
        MIMIC_time_list.append(MIMIC_time)
        MIMIC_Fitness.append(best_fitness)
        
    # Plotting the Graph
    plt.plot(num, SA_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("Simulated Annealing Input Size vs Fitness for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/n_queens_SA_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, SA_time_list,linewidth=3.0)
    plt.grid()
    plt.title("Simulated Annealing Input Size vs Time for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/n_queens_SA_Time.png')  
    plt.clf()


    # Plotting the Graph
    plt.plot(num, RHC_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("Randomized Hill Climbing Input Size vs Fitness for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/n_queens_RHC_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, RHC_time_list,linewidth=3.0)
    plt.grid()
    plt.title("Randomized Hill Climbing  Input Size vs Time for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/n_queens_RHC_Time.png')  
    plt.clf()
    
    
    # Plotting the Graph
    plt.plot(num, GA_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("Genetic Algorithm Input Size vs Fitness for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/n_queens_GA_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, GA_time_list,linewidth=3.0)
    plt.grid()
    plt.title("Genetic Algorithm Input Size vs Time for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/n_queens_GA_Time.png')  
    plt.clf()
        
    
    # Plotting the Graph
    plt.plot(num, MIMIC_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("MIMIC Input Size vs Fitness for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/n_queens_MIMIC_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, MIMIC_time_list,linewidth=3.0)
    plt.grid()
    plt.title("MIMIC Input Size vs Time for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/n_queens_MIMIC_Time.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, SA_Fitness,linewidth=1.0)
    plt.plot(num, RHC_Fitness,linewidth=1.0)
    plt.plot(num, GA_Fitness,linewidth=1.0)
    plt.plot(num, MIMIC_Fitness,linewidth=1.0)
    plt.grid()
    plt.legend(["SA", "RHC", "GA", "MIMIC"])
    # plt.legend(["SA", "RHC", "GA"])
    plt.title("Input Size vs Fitness for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/n_queens_ALL_Input.png')  
    plt.clf()
    print(SA_Fitness)
    print(RHC_Fitness)
    print(GA_Fitness)
    
    # Plotting the Graph
    plt.plot(num, SA_time_list,linewidth=1.0)
    plt.plot(num, RHC_time_list,linewidth=1.0)
    plt.plot(num, GA_time_list,linewidth=1.0)
    plt.plot(num, MIMIC_time_list,linewidth=1.0)
    plt.grid()
    plt.legend(["SA", "RHC", "GA", "MIMIC"])
    # plt.legend(["SA", "RHC", "GA"])
    plt.title("Input Size vs Time for N-Queens Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/n_queens_ALL_Time.png')  
    plt.clf()








if __name__ == "__main__":
    different_inputs()
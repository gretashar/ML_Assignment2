# import mlrose_hiive as mlrose
from cmath import inf
from turtle import color
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import time
import matplotlib.pyplot as plt

#Source: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html#what-is-an-optimization-problem
# Define alternative Four Peaks fitness function for maximization problem


# Initialize custom fitness function object
fitness = mlrose.FourPeaks()
num = 110
problem = mlrose.DiscreteOpt(length = num, fitness_fn = fitness, maximize = True,max_val=2)


# Define decay schedule for SA
schedule = mlrose.ExpDecay()

start_SA = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 10, max_iters = inf,
                                                      random_state = 3, curve=True)
end_SA = time.time()
SA_time = end_SA-start_SA

print("\n*******************************************************\n")

print('Timing for Simulated Annealing for Four Peaks: {:.6f}'. format(SA_time))
print('Best State for Four Peaks: ')
print(best_state)
print('Best Fitness for Four Peaks:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)

# Plotting the Graph
SA_fitness= fitness_curve
SA_x=x
plt.plot(x, fitness_curve,linewidth=3.0, color="r")
plt.grid()
plt.title("Simulated Annealing for Four Peaks Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/four_peaks_SA.png')  
plt.clf()



start_HC = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                      max_attempts = 10, max_iters = inf,
                                                      random_state = 4, curve=True)
end_HC = time.time()
HC_time = end_HC-start_HC

print("\n*******************************************************\n")

print('Timing for Randomized Hill Climbing for Four Peaks: {:.6f}'. format(HC_time))
print('Best State for Four Peaks: ')
print(best_state)
print('Best Fitness for Four Peaks:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)
HC_fitness= fitness_curve
HC_x=x
# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0)
plt.grid()
plt.title("Randomized Hill Climbing for Four Peaks Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/four_peaks_HC.png')  
plt.clf()




start_GA = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,
                                                      max_attempts = 10, max_iters = inf,
                                                      random_state = 3, curve=True, pop_size=1000, mutation_prob=.1)
end_GA = time.time()
GA_time = end_GA-start_GA

print("\n*******************************************************\n")

print('Timing for Genetic Algorithm for Four Peaks: {:.6f}'. format(GA_time))
print('Best State for Four Peaks: ')
print(best_state)
print('Best Fitness for Four Peaks:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)
GA_fitness= fitness_curve
GA_x=x
# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0, color="g")
plt.grid()
plt.title("Genetic Algorithm for Four Peaks Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/four_peaks_GA.png')  
plt.clf()






start_MIMIC = time.time()
# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                      max_attempts = 10, max_iters = 10000,
                                                      random_state = 3, curve=True, fast_mimic=True, pop_size=1000)
end_MIMIC = time.time()
MIMIC_time = end_MIMIC-start_MIMIC

print("\n*******************************************************\n")

print('Timing for MIMIC for Four Peaks: {:.6f}'. format(MIMIC_time))
print('Best State for Four Peaks: ')
print(best_state)
print('Best Fitness for Four Peaks:: {:.2f}'. format(best_fitness))
print("\n*******************************************************\n")
        
# print(fitness_curve)
print(len(fitness_curve))

x = np.arange(0, len(fitness_curve), dtype=int)
MIMIC_fitness= fitness_curve
MIMIC_x=x
# Plotting the Graph
plt.plot(x, fitness_curve,linewidth=3.0, color="orange")
plt.grid()
plt.title("MIMIC for Four Peaks Problem")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/four_peaks_MIMIC.png')  
plt.clf()



# Plotting the Graph
plt.plot(SA_x, SA_fitness,linewidth=1.0)
plt.plot(HC_x, HC_fitness,linewidth=1.0)
plt.plot(GA_x, GA_fitness,linewidth=1.0)
plt.plot(MIMIC_x, MIMIC_fitness,linewidth=1.0)
plt.grid()
plt.legend(["SA", "RHC", "GA", "MIMIC"])
plt.title("Four Peaks Problem Fitness Curves")
plt.xlabel("Iterations")
plt.ylabel("Fitness Values")
plt.savefig('./plots/four_peaks_ALL_fitness.png')  
plt.clf()




def different_inputs():
    num = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # num = [10,20,30,40,50,60,70,80,90, 100,110] 
    SA_Fitness = []
    SA_time_list = []
    RHC_Fitness = []
    RHC_time_list = []
    GA_Fitness = []
    GA_time_list = []
    MIMIC_Fitness = []
    MIMIC_time_list = []
    
    for i,n in enumerate(num):   
        fitness = mlrose.FourPeaks()
        pop=1000

        problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True,max_val=2)

        #Simulated Annealing
        schedule = mlrose.ExpDecay()
        start_SA = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 3, curve=True)
        end_SA = time.time()
        SA_time = end_SA-start_SA
        SA_time_list.append(SA_time)
        SA_Fitness.append(best_fitness)


        start_HC = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 3, curve=True)
        end_HC = time.time()
        HC_time = end_HC-start_HC
        RHC_time_list.append(HC_time)
        RHC_Fitness.append(best_fitness)
     
     
        start_GA = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 3, curve=True, pop_size=1000, mutation_prob=.1)
        end_GA = time.time()
        GA_time = end_GA-start_GA    
    
        GA_time_list.append(GA_time)
        GA_Fitness.append(best_fitness)
        
        
        start_MIMIC = time.time()
        # Solve problem using simulated annealing
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                            max_attempts = 10, max_iters = inf,
                                                            random_state = 3, curve=True, fast_mimic=True, pop_size=1000)
        end_MIMIC = time.time()
        MIMIC_time = end_MIMIC-start_MIMIC      
                
        MIMIC_time_list.append(MIMIC_time)
        MIMIC_Fitness.append(best_fitness)
        
    # Plotting the Graph
    plt.plot(num, SA_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("Simulated Annealing Input Size vs Fitness for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/four_peaks_SA_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, SA_time_list,linewidth=3.0)
    plt.grid()
    plt.title("Simulated Annealing Input Size vs Time for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time (s)")
    plt.savefig('./plots/four_peaks_SA_Time.png')  
    plt.clf()


    # Plotting the Graph
    plt.plot(num, RHC_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("Randomized Hill Climbing Input Size vs Fitness for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/four_peaks_RHC_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, RHC_time_list,linewidth=3.0)
    plt.grid()
    plt.title("Randomized Hill Climbing  Input Size vs Time for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/four_peaks_RHC_Time.png')  
    plt.clf()
    
    
    # Plotting the Graph
    plt.plot(num, GA_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("Genetic Algorithm Input Size vs Fitness for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/four_peaks_GA_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, GA_time_list,linewidth=3.0)
    plt.grid()
    plt.title("Genetic Algorithm Input Size vs Time for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/four_peaks_GA_Time.png')  
    plt.clf()
        
    
    # Plotting the Graph
    plt.plot(num, MIMIC_Fitness,linewidth=3.0)
    plt.grid()
    plt.title("MIMIC Input Size vs Fitness for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/four_peaks_MIMIC_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, MIMIC_time_list,linewidth=3.0)
    plt.grid()
    plt.title("MIMIC Input Size vs Time for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/four_peaks_MIMIC_Time.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, SA_Fitness,linewidth=1.0)
    plt.plot(num, RHC_Fitness,linewidth=1.0)
    plt.plot(num, GA_Fitness,linewidth=1.0)
    plt.plot(num, MIMIC_Fitness,linewidth=1.0)
    plt.grid()
    plt.legend(["SA", "RHC", "GA", "MIMIC"])
    plt.title("Input Size vs Fitness for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/four_peaks_ALL_Input.png')  
    plt.clf()
    
    # Plotting the Graph
    plt.plot(num, SA_time_list,linewidth=1.0)
    plt.plot(num, RHC_time_list,linewidth=1.0)
    plt.plot(num, GA_time_list,linewidth=1.0)
    plt.plot(num, MIMIC_time_list,linewidth=1.0)
    plt.grid()
    plt.legend(["SA", "RHC", "GA", "MIMIC"])
    plt.title("Input Size vs Time for Four Peaks Problem")
    plt.xlabel("Input Size")
    plt.ylabel("Time")
    plt.savefig('./plots/four_peaks_ALL_Time.png')  
    plt.clf()


if __name__ == "__main__":
    different_inputs()
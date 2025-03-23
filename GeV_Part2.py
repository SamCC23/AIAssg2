import numpy as np
import random
import time
import matplotlib.pyplot as plt
from itertools import product
#PART 2 CODE BELOW, clearly labelled memory 2 functions ----------------------------------------------
def always_cooperate(_, __):
    return 1  # cooperate

def always_defect(_, __):
    return 0  # defect

def tit_for_tat(my_hist, o_hist):
    if not o_hist:
        return 1  # cooperate first
    return o_hist[-1]  # mirror opponents last move

def tournament_selection(population, fitnesses, tournament_size=7):
    chosen_idxs = random.sample(range(len(population)), tournament_size)  # pick random spots from the pop
    curr_best_idx = chosen_idxs[0]  # first one is the best for now
    curr_best_fit = fitnesses[curr_best_idx]  # keep their fitness for now

    # go through the rest and see if there's a better one
    for i in chosen_idxs:  
        fit = fitnesses[i]  # get fitness for this one
        if fit > curr_best_fit:  
            curr_best_idx = i  # if better, update best index & best fitness
            curr_best_fit = fit  

    best_indiv = population[curr_best_idx] 

    return best_indiv  # send back the winner

def uniform_crossover(parent_1, parent_2): 
    child = []
    index = 0  
    while index < len(parent_1):  
        gene_from_1 = parent_1[index]  # get gene from parents
        gene_from_2 = parent_2[index]    
        if random.random() < 0.5:     # pick randomly which gene to take
            picked_gene = gene_from_1    
        else:
            picked_gene = gene_from_2    
        
        child.append(picked_gene)  # add the picked gene to the child  
        index = index + 1  
    return child

def mutation(strategy, m_rate):
    for i in range(len(strategy)):
        chance = random.random()
        if chance < m_rate:
            if strategy[i] == 1: #flip the bits 
                strategy[i] = 0
            else:
                strategy[i] = 1

    return strategy

#-----------------------------------------------------
#Part 2 functions and code below here 

def tit_for_tat_noise(my_hist, opp_hist):
    noise=0.1
    if not opp_hist:
        return 1
    if random.random() < noise:
        return 1 - opp_hist[-1]  # flip the expected move
    return opp_hist[-1]

def strategy_decision_memory2(strategy, my_hist, opp_hist):
    # do this when no history
    if len(my_hist) == 0:
        return strategy[0]

    # for when only one round of history
    if len(my_hist) == 1:
        my_prev = my_hist[-1]
        opp_prev = opp_hist[-1]

        if my_prev == 1 and opp_prev == 1:
            return strategy[1] 
        elif my_prev == 1 and opp_prev == 0:
            return strategy[2] 
        elif my_prev == 0 and opp_prev == 1:
            return strategy[3]  
        else:
            return strategy[4]  

    
    my_prev2 = my_hist[-2]
    my_prev1 = my_hist[-1]
    opp_prev2 = opp_hist[-2]
    opp_prev1 = opp_hist[-1]

    
    case_number = 0  # start counter from zero

    # adding points for different move combinations 16 total
    if my_prev2 == 1:
        case_number = case_number + 8  # two rounds ago, we cooperated
    if my_prev1 == 1:
        case_number = case_number + 4  # last round, we cooperated
    if opp_prev2 == 1:
        case_number = case_number + 2  
    if opp_prev1 == 1:
        case_number = case_number + 1  

    # add 1 because [0] is the first move
    return strategy[1 + case_number]

def get_population_memory2(pop_size, strategy_len=17): #strat len 17 here, same otherwise
    population = []
    for i in range(pop_size):
        strat = []
        for j in range(strategy_len):
            rand_bit = int(np.random.choice([0, 1]))
            strat.append(rand_bit)

        if i < pop_size / 2:
            strat[0] = 1  # first move is cooperate for first half
        population.append(strat)
    return population

def fitness_memory2(strategy, opponents, rounds=100): #calls pd game v2
    score = 0
    for o in opponents:
        result = run_pd_game_memory2(strategy, o, rounds) 
        s = result[0] 
        score = score + s
    return score

def run_pd_game_memory2(my_strat, opp_behaviour, total_rounds=100):#uses strat decision memory 2
    my_points = 0
    their_points = 0
    my_moves_history = []
    their_moves_history = []

    for round_num in range(total_rounds):
        this_move = strategy_decision_memory2(my_strat, my_moves_history, their_moves_history)
        their_move = opp_behaviour(their_moves_history, my_moves_history)

        my_moves_history.append(this_move)
        their_moves_history.append(their_move)

        if (this_move, their_move) == (1, 1):
            my_points += 3
            their_points += 3
        elif (this_move, their_move) == (1, 0):
            my_points += 0
            their_points += 5
        elif (this_move, their_move) == (0, 1):
            my_points += 5
            their_points += 0
        else:
            my_points += 1
            their_points += 1

    return my_points, their_points

#only change is it calls v2 functions and accepts a list of opponents as an argument
def genetic_algo_memory2(pop_size=100, generations=200, crossover_rate=0.85, mutation_rate=0.1, elite_size=5, opponents=None):
    if opponents is None:
        opponents = [always_cooperate, always_defect, tit_for_tat]

    population = get_population_memory2(pop_size)
    fitness_history = []
    best_strategy = None
    best_fitness = 0
    start_time = time.time()

    for gen in range(generations):
        fitnesses = [fitness_memory2(ind, opponents) for ind in population]

        idxs = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        sorted_fit = [fitnesses[k] for k in idxs]
        sorted_pop = [population[k] for k in idxs]

        new_population = sorted_pop[:elite_size]  # elitism

        if sorted_fit[0] > best_fitness:
            best_fitness = sorted_fit[0]
            best_strategy = sorted_pop[0]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            if random.random() < crossover_rate:
                child = uniform_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population
        fitness_history.append(best_fitness)

        if gen % 20 == 0 or gen == generations - 1:
            print(f"Generation {gen} - Best Fitness: {best_fitness}")

    end_time = time.time()

    print("\nBest Memory-2 Strategy:", best_strategy)
    print("Best fitness:", best_fitness)
    print("Time:", end_time - start_time, "seconds")

    plt.plot(fitness_history)
    plt.title(f"Fitness over Generations (Memory-2)\nPop: {pop_size}, Gen: {generations}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Score)")
    plt.show()

    return best_strategy, best_fitness, fitness_history


if __name__ == "__main__":
    opponents_with_noisy_tft = [
        always_cooperate,
        always_defect,
        tit_for_tat_noise
    ]

    best_strategy_noisy, best_fitness_noisy, fitness_history_noisy = genetic_algo_memory2(
        pop_size=100,
        generations=2000,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elite_size=2,
        opponents=opponents_with_noisy_tft
    ) 
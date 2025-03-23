import numpy as np
import random
import time
import matplotlib.pyplot as plt
from itertools import product

def always_cooperate(_, __):
    return 1  # cooperate

def always_defect(_, __):
    return 0  # defect

def tit_for_tat(my_hist, o_hist):
    if not o_hist:
        return 1  # cooperate first
    return o_hist[-1]  # mirror opponents last move

# player strategy 
# memory-one index mapping (C, C) = 0, (C, D) = 1, (D, C) = 2, (D, D) = 3
def strategy_decision(strategy, my_hist, opp_hist):
    if len(my_hist) == 0:
        return strategy[0]  # strategy[0] if its the first move 
    my_prev = my_hist[-1]
    opp_prev = opp_hist[-1]

    if my_prev == 1 and opp_prev == 1:
        index = 0  # both cooperated
    elif my_prev == 1 and opp_prev == 0:
        index = 1  # we cooperated, they defected
    elif my_prev == 0 and opp_prev == 1:
        index = 2  # we defected, they cooperated
    elif my_prev == 0 and opp_prev == 0:
        index = 3  # both defected

    return strategy[index]

def run_pd_game(my_strat, opp_behaviour, total_rounds=100):
    my_points = 0 # keep track of points for both sides
    their_points = 0
    my_moves_history = []
    their_moves_history = []

    for round_num in range(total_rounds):
        # decide what I do this round
        this_move = strategy_decision(my_strat, my_moves_history, their_moves_history)

        # ask opponent for their move, give them histories
        their_move = opp_behaviour(their_moves_history, my_moves_history)
        my_moves_history.append(this_move) # record both moves
        their_moves_history.append(their_move)
        situation = (this_move, their_move)

        if situation == (1, 1):  # figure out points for both based on moves
            my_points += 3
            their_points += 3
        elif situation == (1, 0):
            my_points += 0  
            their_points += 5
        elif situation == (0, 1):
            my_points += 5
            their_points += 0
        elif situation == (0, 0):
            my_points += 1
            their_points += 1

    return my_points, their_points # return back both total scores

# fitness function
def fitness(strategy, opponents, rounds=100):
    score = 0
    for o in opponents:
        result = run_pd_game(strategy, o, rounds)
        s = result[0] #take player score
        score = score + s
    return score

def get_population(pop_size, strategy_len=4):
    population = []
    i = 0  # starting index
    while i < pop_size:
        strat = []  # make a random strategy
        j = 0
        while j < strategy_len:
            rand_bit = int(np.random.choice([0, 1]))
            strat.append(rand_bit)
            j = j + 1

        if i < pop_size / 2:  # make sure the first half always cooperates first
            strat[0] = 1

        population.append(strat)
        i = i + 1  

    return population

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

def genetic_algo(pop_size=100, generations=200, crossover_rate=0.85, mutation_rate=0.1, elite_size=5):
    opponents = [always_cooperate, always_defect, tit_for_tat]
    population = get_population(pop_size) # gets population of [0,1,0,1]'s
    fitness_history = [] #list of best fitnesses over gens to plot
    best_strategy = None
    best_fitness = 0
    start_time = time.time() # just timing how long it takes to run the GA

    for gen in range(generations):
        fitnesses = []  
        for individual in population:  # go through the population
            score = fitness(individual, opponents)
            fitnesses.append(score)  

    idxs = list(range(len(fitnesses)))   # make a list of indexes from 0 to how many fitnesses we have

    for i in range(len(idxs)):  # sort the indexes by comparing their fitness values
        for j in range(i + 1, len(idxs)):
            if fitnesses[idxs[j]] > fitnesses[idxs[i]]:
                tmp = idxs[i]               
                idxs[i] = idxs[j]           
                idxs[j] = tmp               

    sorted_fit = []  # fitnesses in order (big to small)
    sorted_pop = [] 

    for k in idxs:
        sorted_fit.append(fitnesses[k])   # add fitness to sorted list
        sorted_pop.append(population[k])  # matching population to sorted list
        new_population = sorted_pop[:elite_size] # elitism

        if sorted_fit[0] > best_fitness:     # set best solution
            best_fitness = sorted_fit[0]
            best_strategy = sorted_pop[0]

        # make offspring, same logic from my first assignment
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            if random.random() < crossover_rate:
                child = uniform_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutation(child, mutation_rate)
            new_population.append(child) # add to new pop

        population = new_population
        fitness_history.append(best_fitness)

        if gen % 20 == 0 or gen == generations - 1:
            print(f"Generation {gen} - Best Fitness: {best_fitness}")

    end_time = time.time()

    print("\nBest Strategy:", best_strategy)
    print("Best fitness:", best_fitness)
    print("Time:", end_time - start_time, "seconds")
    print("\nPerformance vs every opponent:")
    for opp in opponents:
        score, opp_score = run_pd_game(best_strategy, opp)
        print(f"{opp.__name__}: {score} (Opponent score: {opp_score})")

    # plot fitness
    plt.plot(fitness_history)
    plt.title(f"Fitness over Generations\nPop: {pop_size}, Gen: {generations}, Cross: {crossover_rate}, Mut: {mutation_rate}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()

    return best_strategy, best_fitness, fitness_history

if __name__ == "__main__":
    best_strategy, best_fitness, fitness_history = genetic_algo(
        pop_size=100,
        generations=200,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elite_size=2
    )

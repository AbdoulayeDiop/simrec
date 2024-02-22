import numpy as np
import pygad
from ranking import ALL_MODELS, scorer, scorer_func
from sklearn.model_selection import cross_val_predict, KFold
import torch
import torch.nn as nn
from meta_model import AEKNN


def mfs_plus_hpo_knn(X, Y, n_neighbors_values=None, metrics=None, weights=None, num_generations=100,
                 pop_size=8, parent_selection_type="sss", crossover_probability=0.95,
                 mutation_probability=0.05, crossover_type="uniform", mutation_type="random",
                 n_splits=10, scorer=scorer):
    if n_neighbors_values is None:
        n_neighbors_values = np.arange(1, 32, 2)
    if metrics is None:
        metrics = ["euclidean", "manhattan", "cosine"]
    if weights is None:
        weights = ["uniform", "distance"]

    def on_generation(ga_instance):
        if ga_instance.generations_completed % 50 == 0:
            _, fitness, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness)
            print(
                f"GEN: {ga_instance.generations_completed}, Best fitness: {fitness}")

    def fitness_func(ga_instance, solution, solution_idx):
        selected_feats = np.array(solution[:X.shape[1]]) > 0
        n_neighbors = solution[X.shape[1]]
        metric = metrics[solution[X.shape[1]+1]]
        w = weights[solution[X.shape[1]+2]]
        knn = ALL_MODELS["KNN"](n_neighbors=n_neighbors,
                                metric=metric, weights=w)
        Y_pred = cross_val_predict(
            knn, X[:, selected_feats], Y, cv=n_splits, n_jobs=-1)
        fitness = scorer_func(Y, Y_pred)
        return fitness #- 3e-4*sum(selected_feats)

    fitness_function = fitness_func
    num_parents_mating = pop_size//2
    sol_per_pop = pop_size
    num_genes = X.shape[1] + 3
    gene_type = int
    gene_space = [[0, 1] for _ in range(
        X.shape[1])] + [n_neighbors_values, range(len(metrics)), range(len(weights))]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_type=gene_type,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           crossover_probability=crossover_probability,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           on_generation=on_generation
                           )
    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()
    selected_feats = np.array(solution[:X.shape[1]]) > 0
    n_neighbors = solution[X.shape[1]]
    metric = metrics[solution[X.shape[1]+1]]
    w = weights[solution[X.shape[1]+2]]
    knn = ALL_MODELS["KNN"](n_neighbors=n_neighbors, metric=metric, weights=w)
    return knn, selected_feats, n_neighbors, metric, w, ga_instance


def mfs_plus_hpo_dtree(X, Y, min_samples_leaf_values=None, max_features=None, num_generations=100,
                 pop_size=8, parent_selection_type="sss", crossover_probability=0.9,
                 mutation_probability=0.1, crossover_type="uniform", mutation_type="random",
                 n_splits=10, scorer=scorer):
    if min_samples_leaf_values is None:
        min_samples_leaf_values = np.arange(1, 32, 2)
    if max_features is None:
        max_features = [None, "sqrt", "log2"]

    def on_generation(ga_instance):
        if ga_instance.generations_completed % 10 == 0:
            _, fitness, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness)
            print(
                f"GEN: {ga_instance.generations_completed}, Best fitness: {fitness}")

    def fitness_func(ga_instance, solution, solution_idx):
        selected_feats = np.array(solution[:X.shape[1]]) > 0
        min_samples_leaf = solution[X.shape[1]]
        max_f = max_features[solution[X.shape[1]+1]]
        model = ALL_MODELS["DTree"](min_samples_leaf=min_samples_leaf, max_features=max_f)
        Y_pred = cross_val_predict(model, X[:, selected_feats], Y, cv=n_splits, n_jobs=-1)
        fitness = scorer_func(Y, Y_pred)
        return fitness #- 3e-4*sum(selected_feats)

    fitness_function = fitness_func
    num_parents_mating = pop_size//2
    sol_per_pop = pop_size
    num_genes = X.shape[1] + 2
    gene_type = int
    gene_space = [[0, 1] for _ in range(
        X.shape[1])] + [min_samples_leaf_values, range(len(max_features))]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_type=gene_type,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           crossover_probability=crossover_probability,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           on_generation=on_generation
                           )
    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()
    selected_feats = np.array(solution[:X.shape[1]]) > 0
    min_samples_leaf = solution[X.shape[1]]
    max_f = max_features[solution[X.shape[1]+1]]
    model = ALL_MODELS["DTree"](min_samples_leaf=min_samples_leaf, max_features=max_f)
    return model, selected_feats, min_samples_leaf, max_f, ga_instance

def create_encoder_decoder(input_dim, n_neurons):
    input_dim, output_dim = input_dim, input_dim
    encoder = nn.Sequential()
    for i, n in enumerate(n_neurons):
        if len(encoder) > 0:
            encoder.append(nn.ReLU())
        encoder.append(nn.Linear(input_dim, n))
        input_dim = n

    decoder = nn.Sequential()
    for i in range(len(n_neurons)):
        if n_neurons[len(n_neurons)-i-1] > input_dim:
            decoder.append(nn.Linear(input_dim, n_neurons[len(n_neurons)-i-1]))
            decoder.append(nn.ReLU())
            input_dim = n_neurons[len(n_neurons)-i-1]
    decoder.append(nn.Linear(input_dim, output_dim))
    return encoder, decoder


def aeknn_hpo(X, Y, n_neurons, ae_fit_params, n_neighbors_values=None, metrics=None,
              weights=None, device='cpu', num_generations=100, pop_size=8, parent_selection_type="sss",
              crossover_probability=0.75, mutation_probability=0.25, crossover_type="uniform",
              mutation_type="random", n_splits=10, scorer_func=scorer_func):

    if n_neighbors_values is None:
        n_neighbors_values = np.arange(1, 26)
    if metrics is None:
        metrics = ["euclidean", "manhattan", "cosine"]
    if weights is None:
        weights = ["uniform", "distance"]

    def on_generation(ga_instance):
        _, fitness, _ = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness)
        print(
            f"GEN: {ga_instance.generations_completed}, Best fitness: {fitness}")

    def fitness_func(ga_instance, solution, solution_idx):
        selected_layers = np.array(solution[:len(n_neurons)]) > 0
        n_neighbors = solution[len(n_neurons)]
        metric = metrics[solution[len(n_neurons)+1]]
        w = weights[solution[len(n_neurons)+2]]
        fitness = 0
        for train, test in KFold(n_splits=n_splits).split(X):
            encoder, decoder = create_encoder_decoder(
                X.shape[1], n_neurons[selected_layers])
            aeknn = AEKNN(encoder, decoder, n_neighbors,
                          metric, w, device=device)
            aeknn.fit(X[train], Y[train], ae_fit_params)
            Y_pred = aeknn.predict(X[test])
            fitness += scorer_func(Y[test], Y_pred)/n_splits
        return fitness

    fitness_function = fitness_func
    num_parents_mating = pop_size//2
    sol_per_pop = pop_size
    num_genes = len(n_neurons) + 3
    gene_type = int
    gene_space = [[0, 1] for _ in range(
        len(n_neurons))] + [n_neighbors_values, range(len(metrics)), range(len(weights))]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_type=gene_type,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           crossover_probability=crossover_probability,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           on_generation=on_generation,
                           #    parallel_processing=["process", 10]
                           )
    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()
    selected_layers = np.array(solution[:len(n_neurons)]) > 0
    n_neighbors = solution[len(n_neurons)]
    metric = metrics[solution[len(n_neurons)+1]]
    w = weights[solution[len(n_neurons)+2]]
    return n_neurons[selected_layers], n_neighbors, metric, w, ga_instance


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    print(X.shape)
    X = np.c_[X, np.random.rand(*X.shape)]
    # knn, selected_feats, n_neighbors, metric, w, ga_instance = mfs_plus_hpo(X, y, num_generations=30, scorer="neg_mean_absolute_error")
    # print(np.where(selected_feats>0)[0])
    # print(n_neighbors, metric, w)
    ae_fit_params = {
        "lr": 1e-3,
        "weight_decay": 0,
        "epochs": 400,
        "batch_size_train": 16,
        "batch_size_val": 16,
    }
    n_neurons = np.array([32, 16, 8, 4])
    n_neurons, n_neighbors, metric, w, ga_instance = aeknn_hpo(
        X[:100], y[:100], n_neurons, ae_fit_params, num_generations=30, pop_size=10, scorer_func=lambda yt, y: -np.mean((y - yt)**2))
    print(n_neurons)
    print(n_neighbors, metric, w)

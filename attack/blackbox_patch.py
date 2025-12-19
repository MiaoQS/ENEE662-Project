import torch
import numpy as np
from numpy.linalg import norm
import math
from torch.utils.data.dataloader import DataLoader
from skimage.transform import resize
from scipy.special import softmax
from attack.patch_attack import PatchAttack
from config import Config
from my_utils import softmax_parent_selection, find_neighbor



class Blackbox_patch(PatchAttack):
    def __init__(self, model, model_name, patch_area, n_batch=1, batch_size=5, tb_logger=None, log_dir=None, tracker=None, targeted_attack=False):
        super().__init__(model, model_name, patch_area, n_batch, batch_size, tb_logger, log_dir, tracker, targeted_attack)


    def create_patch(self, params, p_h, p_w, mode):
        if mode == 'square':
            params_np = np.array(params).reshape((-1, 6))
            params_np = params_np[params_np[:, 0].argsort()[::-1]]
            patch = np.zeros([3, p_h, p_w])
            max_edge_r = 0.3
            for i in range(params_np.shape[0]):
                s = max(1, math.ceil(max_edge_r * min(p_h, p_w) * params_np[i, 0]))
                h_start = int((p_h-s) * params_np[i, 1])
                w_start = int((p_w-s) * params_np[i, 2])
                patch[:, h_start:h_start + s, w_start:w_start + s] = np.reshape(params_np[i, 3:6], (3, 1, 1))
        elif mode == 'pixel':
            patch = np.array(params).reshape((3, p_h, p_w))
        return patch

    def GA_attack(self, num_generations = 10000, num_parents_mating = 5, sol_per_pop = 20, mode='square', patch_only=True): # mode: 'pixel' or 'square',
        import pygad
        # Note: x, x_best, x_new are numpy.ndarray in the range of [0, 1]
        C = 3
        p_t, p_l, p_h, p_w = self.patch_area
        if mode == 'pixel':
            i_h, i_w = 10, 10 # p_h, p_w
            n_patch_features = C * i_h * i_w
        elif mode == 'square':
            i_h, i_w = p_h, p_w
            squares = 500
            n_patch_features = 6 * squares
        
        def fitness_function(ga, solution, solution_idx):
            patch = self.create_patch(solution, i_h, i_w, mode)
            patch = resize(patch, (C, p_h, p_w))
            patch = np.clip(patch, 0., 1.)
            with torch.no_grad():
                curr_loss = self.Score.score(patch, patch_only=True)
            fitness = -curr_loss
            return fitness

        def on_generation_func(ga):
            solution, solution_fitness, solution_idx = ga.best_solution()
            patch = self.create_patch(solution, i_h, i_w, mode)
            patch = resize(patch, (C, p_h, p_w))
            patch = np.clip(patch, 0., 1.)
            i_iter = ga.generations_completed
            # log
            self.log(patch, -solution_fitness, None, None, i_iter, 'GA_attack', log_gap=1 ,img_gap=100, patch_only=True)
            # evaluation
            self.evaluate(patch, i_iter, 'GA_attack', log_gap=5, img_gap=100, patch_only=True)

        gene_space = {'low': 0., 'high': 1.} 
        ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=n_patch_features,
                       init_range_low=0.,
                       init_range_high=1.,
                       parent_selection_type=softmax_parent_selection, #"sss", # rank
                    #    K_tournament=5,
                       keep_elitism=1,
                    #    keep_parents=-1,
                       crossover_type="single_point", # prob_uniform_crossover, # "uniform",
                    #    crossover_probability=1,
                       mutation_type="random",
                    #    mutation_probability=0.3,
                       mutation_percent_genes=50,
                       random_mutation_min_val=-0.3,
                       random_mutation_max_val=0.3,
                       gene_space=gene_space,
                       random_seed=17,
                       on_generation=on_generation_func
                       )
        ga_instance.run()


    def zoo(self, total_steps=10000, K=4, num_pos=100, num_init=500, trail=30, patch_only=True):
        print(f'zoo optimization, attack model: {self.model_name}, k={K}, n_init={num_init}, trail={trail}, num_pos={num_pos}, patch_only: {patch_only}')
        # scene_loader
        scene_loader = DataLoader(self.train_dataset, batch_size=Config.train_scenes, shuffle=False,\
                                    num_workers=2, pin_memory=True, drop_last=True)
        scene_loader_iter = iter(scene_loader)
        scenes, _ = next(scene_loader_iter)
        c = scenes.shape[1]
        p_t, p_l, p_h, p_w = self.patch_area
        # Initialize pattern
        patch_curr = np.random.rand(c, p_h, p_w)
        patch_best = patch_curr.copy()
        score_best = self.Score.score(patch_curr, patch_only=patch_only) # lower score is better
        
        # Initialize pattern
        for i in range(num_init):
            patch_curr = np.random.rand(c, p_h, p_w)
            score = self.Score.score(patch_curr, patch_only=patch_only)
            if score < score_best:
                score_best = score
                patch_best[:] = patch_curr
                print(f"Initialize: step: {i} best score: {score_best}")

        # Choose pattern:
        hist_patch = [patch_best]
        hist_score = np.array([1 / score_best])
        sim_graph = np.eye(100)
        last_mean_y = np.array([0] * num_pos)
        # last_avg_mean = 0
        beta1 = Config.beta1
        beta2 = Config.beta1
        eps = Config.eps
        lr = Config.lr
        for i in range(1, total_steps):
            if not Config.hardbeat_oneway:
                if len(hist_score) > K:
                    topk_idx = np.argpartition(hist_score, -K)[-K:]
                else:
                    topk_idx = np.arange(len(hist_score))
                topk_prob = softmax(hist_score[topk_idx])
                curr_idx = np.random.choice(topk_idx, size=1, p=topk_prob)[0]
                u = np.random.rand(1)[0]
                if u <= min(1, hist_score[curr_idx] / hist_score[-1]):
                    patch_curr = hist_patch[curr_idx]
                    if u <= 0.5 and i > 2:
                        neighbor_idx = find_neighbor(sim_graph, curr_idx, hist_score)
                        neighbor_patch = hist_patch[neighbor_idx]
                        alpha = np.random.rand(1)[0]
                        patch_curr = alpha * patch_curr + (1-alpha) * neighbor_patch
                else:
                    patch_curr = hist_patch[-1]
                    curr_idx = len(hist_patch) - 1

            ## gradient estimate:
            deltas = []
            avg_mean = []
            for j in range(num_pos): # for each position
                noise = np.random.rand(trail, *patch_curr.shape)
                noise = 2.0 * (noise - 0.5) * 0.5
                patches_candi = np.clip(noise + patch_curr, 0, 1)
                noise = patches_candi - patch_curr

                indice = torch.randperm(scenes.shape[0])[:1]
                scene = scenes[indice]
                # scene = scenes[0:1,:,:,:]
                scores = self.Score.gredient_sample(
                                        patches_candi, #numpy[trail,3,h,w]
                                        patch_curr, #numpy[1,3,h,w]
                                        scene, #tensor[1,3,h,w]
                                        patch_only)

                candi_y = scores.cpu().numpy()  # scores(-1-1) of all sample points
                candi_y = np.sign(candi_y)
                # candi_y = 1.0/ (1+ np.exp(np.negative(candi_y)))
                mean_y = np.mean(candi_y)   # mean of scores
                avg_mean.append(mean_y)
                diff_y = mean_y - last_mean_y[j]
                # diff_y = np.exp(diff_y) if diff_y > 0 else np.log(diff_y + 3)
                if diff_y > 0:
                    diff_y = np.exp(diff_y)
                    if mean_y >= 1:
                        diff_y /= 5
                else:
                    diff_y = np.log(diff_y + 3)
                if mean_y == -1 or mean_y == 1:
                    delta = mean_y * np.mean(noise, axis=0)
                else:
                    delta = np.mean(noise * (candi_y - mean_y).reshape((trail, 1, 1, 1)), axis=0)
                # delta = diff_y * delta / norm(delta)
                delta = delta / norm(delta)
                last_mean_y[j] = mean_y
                deltas.append(delta)
            gradf = torch.from_numpy(np.mean(deltas, axis=0))
            avg_mean = np.mean(avg_mean)

            ## optimizer
            gradf_flat = gradf.flatten()
            if i == 1:
                grad_momentum = gradf
                full_matrix   = torch.outer(gradf_flat, gradf_flat)
            else:
                grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
                full_matrix   = beta2 * full_matrix\
                                + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)
            grad_momentum /= (1 - beta1 ** (i + 1))
            full_matrix   /= (1 - beta2 ** (i + 1))
            factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
            gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)
            
            patch_curr = patch_curr + lr * gradf.numpy()
            patch_curr = np.clip(patch_curr, 0, 1)
            # update history
            
            score_curr = self.Score.score(patch_curr, patch_only=patch_only)
            print('-' * 30)
            print(f"score_curr: {score_curr}")
            if not Config.hardbeat_oneway:
                hist_patch.append(patch_curr)
                hist_score = np.append(hist_score, 1 / score_curr)

                if len(hist_patch) == 100:
                    hist_patch.pop(0)
                    hist_score = np.delete(hist_score, 0)
                    sim_graph = np.delete(sim_graph, 0, axis=0)
                    sim_graph = np.delete(sim_graph, 0, axis=1)
                    new_row = np.zeros((1, sim_graph.shape[1]))
                    sim_graph = np.vstack((sim_graph, new_row))
                    new_colomn = np.zeros((sim_graph.shape[0], 1))
                    sim_graph = np.hstack((sim_graph, new_colomn))
                    sim_graph[-1][-1] = 1

            # store best
            if score_curr < score_best:
                score_best = score_curr
                patch_best[:] = patch_curr

            # log
            self.log(patch_best, score_best, None, None, i, 'zoo', log_gap=1 ,img_gap=1000, patch_only=patch_only)
            # evaluation
            self.evaluate(patch_best, i, 'zoo', log_gap=50, img_gap=1000, patch_only=patch_only)
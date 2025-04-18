#!/usr/bin/env python
# filename: main_illuminate_particle_lenia.py

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default="particle_lenia",
                   help="name of the substrate (assume 'particle_lenia' is implemented)")
group.add_argument("--rollout_steps", type=int, default=512,
                   help="number of rollout timesteps for the simulation")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", 
                   help="the foundation model to use (clip, dino, etc.)")
# Adding time-series collection parameters for causal blanket analysis
group.add_argument("--save_time_series", action="store_true", 
                   help="whether to save time-series data for causal blanket analysis")
group.add_argument("--time_sampling_rate", type=int, default=8,
                   help="sampling rate for time-series data (1=every step, 8=every 8th step)")
group.add_argument("--cb_eval_subset", type=int, default=16, 
                   help="evaluate causal blanket metrics on a subset of the population")

group = parser.add_argument_group("optimization")
group.add_argument("--k_nbrs", type=int, default=2, help="k_neighbors for nearest neighbor calculation")
group.add_argument("--n_child", type=int, default=32, help="number of children to generate")
group.add_argument("--pop_size", type=int, default=256, help="population size for the illumination library")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")


def parse_args():
    args = parser.parse_args()
    # 文字列"none"などをPythonのNoneに置き換える例
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args


def main(args):
    print("ARGS:", args)

    # 1) Foundation Model (CLIP / DINO など)
    fm = foundation_models.create_foundation_model(args.foundation_model)
    
    # 2) Particle Lenia のサブストレート生成
    substrate = substrates.create_substrate(args.substrate)
    # Wrap with FlattenSubstrateParameters to access n_params attribute
    substrate = substrates.FlattenSubstrateParameters(substrate)
    # ここでは FlattenSubstrateParameters は使わず、別途 substrate.n_params など想定
    # 必要に応じて substrates.FlattenSubstrateParameters などを挟んでください
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps  # substrateのデフォルト
    
    # 3) rollout関数を定義：パラメータ p からシミュレーションを行い、FM埋め込みを返す
    def rollout_fn_(rng, p):
        # Configure time sampling based on whether we're collecting time series data
        time_sampling = args.time_sampling_rate if args.save_time_series else 'final'
        
        # シミュレーションを実行 → 画像化 → FM で埋め込み
        # return_state=True to get the full state history when needed
        rollout_result = rollout_simulation(
            rng=rng, params=p,
            substrate=substrate, 
            fm=fm, 
            rollout_steps=args.rollout_steps,
            time_sampling=time_sampling,
            img_size=224,
            return_state=args.save_time_series  # Return state history for causal blanket analysis
        )
        # rollout_resultには z (埋め込み) や生成画像等を含む想定
        # dict(params=p, z=..., image=..., ...)
        return dict(params=p, **rollout_result)
    
    # JAX jitをかける
    rollout_fn = jax.jit(lambda rng, p: rollout_fn_(rng, p))
    
    # 4) 初期母集団の作成
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = split(rng)
    # パラメータ初期化：とりあえず 0付近で pop_size 個
    # substrate.n_params を使う想定
    param_init = 0.0 * jax.random.normal(_rng, (args.pop_size, substrate.n_params))
    
    # まず全員rolloutして埋め込み z を得る
    raw_pop_results = [rollout_fn(split(_rng, 1)[0], p) for p in tqdm(param_init)]
    
    # If we're saving time series for causal blanket analysis, extract states before stacking
    if args.save_time_series:
        # Save state histories separately before stacking
        state_histories = []
        for p_result in raw_pop_results:
            if 'state' in p_result:
                state_histories.append(p_result['state'])
                # Remove state to avoid issues with stacking
                del p_result['state']
    
    # Now stack the results (without state histories if we're saving them)
    pop = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *raw_pop_results)

    @jax.jit
    def do_iter(pop, rng):
        """1反復（子生成→rollout→類似度計算→母集団更新）を行う。"""
        rng, _rng = split(rng)
        # ランダムに親を n_child 個サンプリング
        idx_p = jax.random.randint(_rng, (args.n_child,), minval=0, maxval=args.pop_size)
        params_parent = pop['params'][idx_p]  # shape: (n_child, D)
        
        rng, _rng1, _rng2 = split(rng, 3)
        noise = jax.random.normal(_rng1, (args.n_child, substrate.n_params))
        params_children = params_parent + args.sigma * noise
        
        # 子のrollout
        children = jax.vmap(rollout_fn)(split(_rng2, args.n_child), params_children)
        
        # Combine parent and children populations
        pop_new = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *[pop, children])
        
        # pop_newのサイズは pop_size + n_child
        # pop_new['z'] が (pop_size + n_child, z_dim) と仮定
        X = pop_new['z']
        # cosine類似度を計算 = -X @ X.T で負の類似度を使う例
        # ここではembeddingが正規化されている想定 or 
        # あるいは L2 norm してから内積をとるなど
        D = -X @ X.T  # shape: (pop_size+n_child, pop_size+n_child)
        
        # 対角は無限大にして、自分自身を最近傍にしない
        n_tot = pop['z'].shape[0] + children['z'].shape[0]
        D = D.at[jnp.arange(n_tot), jnp.arange(n_tot)].set(jnp.inf)
        
        to_kill = jnp.zeros(args.n_child, dtype=int)
        
        def kill_least(carry, _):
            D_, tk_, i_ = carry
            # k近傍(k_nbrs)との類似度が最小（=距離が最大）を落とす など 
            # ここでは "最も類似(=最も -X@X.T が小さい）" を killする例
            # k_nbrs=2個の近傍の平均をとり、最小の個体をkill
            # つまり多様性が低い個体から消す
            # ただし実装例はASALのコードによって微妙に異なるかもしれません
            sort_vals = D_.sort(axis=-1)[:, :args.k_nbrs].mean(axis=-1)
            tki = sort_vals.argmin()
            # tki行列をinf埋め
            D_ = D_.at[:, tki].set(jnp.inf)
            D_ = D_.at[tki, :].set(jnp.inf)
            tk_ = tk_.at[i_].set(tki)
            return (D_, tk_, i_+1), None
        
        (D_final, to_kill, _), _ = jax.lax.scan(kill_least, (D, to_kill, 0), None, length=args.n_child)
        
        # 残すインデックス
        all_idx = jnp.arange(n_tot)
        to_keep = jnp.setdiff1d(all_idx, to_kill, assume_unique=True, size=args.pop_size)
        
        pop_survived = jax.tree_util.tree_map(lambda x: x[to_keep], pop_new)
        
        # 多様性指標などを計算してログに格納
        # 例: illumination_score = ?
        # あるいは適宜やりたい計算に入れ替え
        # ここでは asal_metrics.calc_illumination_score を呼ぶ場合の例:
        ill_score = asal_metrics.calc_illumination_score(pop_survived['z'])
        
        return pop_survived, dict(illumination=ill_score, keep_indices=to_keep)

    # 5) 探索ループ
    data = []
    pbar = tqdm(range(args.n_iters))
    
    # Initialize list to store state histories if we're saving them
    all_state_histories = []
    if args.save_time_series:
        all_state_histories = state_histories.copy()
    
    for i_iter in pbar:
        rng, _rng = split(rng)
        
        # Handle state histories separately when doing iterations
        if args.save_time_series:
            # Rollout children to get their states (outside of jit)
            idx_p = jax.random.randint(_rng, (args.n_child,), minval=0, maxval=args.pop_size)
            params_parent = pop['params'][idx_p]
            
            rng, _rng1, _rng2 = split(rng, 3)
            noise = jax.random.normal(_rng1, (args.n_child, substrate.n_params))
            params_children = params_parent + args.sigma * noise
            
            # Get children states
            children_results = [rollout_fn(r, p) for r, p in zip(split(_rng2, args.n_child), params_children)]
            children_states = []
            for r in children_results:
                if 'state' in r:
                    children_states.append(r['state'])
                    del r['state']
            
            # Stack children results
            children = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *children_results)
            
            # Do iteration
            pop, di = do_iter(pop, rng)
            
            # Update state histories based on keep_indices
            keep_indices = di['keep_indices']
            updated_state_histories = []
            
            # Combine parent and child state histories
            combined_states = all_state_histories + children_states
            
            # Keep only the states that survived
            for idx in keep_indices:
                updated_state_histories.append(combined_states[idx])
            
            all_state_histories = updated_state_histories
        else:
            # Normal iteration without state tracking
            pop, di = do_iter(pop, rng)
        
        data.append(di)
        pbar.set_postfix(ill_score=di['illumination'].item())
        
        # 10%ごと or 最終イテレーションで保存
        if (args.save_dir is not None) and ((i_iter % (args.n_iters//10)==0) or (i_iter==args.n_iters-1)):
            data_save = jax.tree_util.tree_map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            
            # Save population
            pop_save = jax.tree_util.tree_map(lambda x: np.array(x), pop)
            
            # If we're saving time series data, include state histories
            if args.save_time_series:
                # If we have a large population, only save state histories for a subset
                if len(all_state_histories) > args.cb_eval_subset:
                    # Calculate diversity scores to select representative individuals
                    embeddings = np.array(pop['z'])
                    D = -embeddings @ embeddings.T
                    np.fill_diagonal(D, np.inf)
                    
                    # Select the cb_eval_subset most diverse individuals
                    selected_indices = []
                    remaining_indices = list(range(len(embeddings)))
                    
                    # Greedy selection of diverse individuals
                    while len(selected_indices) < args.cb_eval_subset and remaining_indices:
                        if not selected_indices:
                            # Start with a random individual
                            idx = np.random.choice(remaining_indices)
                        else:
                            # Find individual with maximum distance to already selected
                            sel_embeddings = embeddings[selected_indices]
                            rem_embeddings = embeddings[remaining_indices]
                            distances = -rem_embeddings @ sel_embeddings.T
                            min_distances = distances.min(axis=1)
                            idx = remaining_indices[min_distances.argmax()]
                        
                        selected_indices.append(idx)
                        remaining_indices.remove(idx)
                    
                    # Only save state histories for selected individuals
                    selected_state_histories = [all_state_histories[idx] for idx in selected_indices]
                    selected_params = np.array(pop_save['params'])[selected_indices].tolist()
                    
                    # Create separate file for causal blanket analysis data
                    cb_data = {
                        'states': selected_state_histories,
                        'params': selected_params,
                        'indices': selected_indices
                    }
                    util.save_pkl(args.save_dir, "causal_blanket_data", cb_data)
                else:
                    # Save all state histories if population is small enough
                    cb_data = {
                        'states': all_state_histories,
                        'params': np.array(pop_save['params']).tolist(),
                        'indices': list(range(len(all_state_histories)))
                    }
                    util.save_pkl(args.save_dir, "causal_blanket_data", cb_data)
            
            util.save_pkl(args.save_dir, "pop", pop_save)
    
    print("Illumination finished!")


if __name__ == "__main__":
    args = parse_args()
    main(args)

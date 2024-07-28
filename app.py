from shiny import render, ui
from shiny.express import input

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as betadist
import pandas as pd

sns.set_style("white")

ui.panel_title("A/B sample size calculator")
ui.panel_title("By Paul Anzel (anzelpwj@gmail.com)")

ui.input_numeric("open_rate_avg", "Average open rate (percent)", 20, min=1, max=99, step=0.1)  
ui.input_numeric("open_rate_stg", "Open rate standard deviation (percent)", 5, min=1, max=20, step=0.1)  
ui.input_numeric("list_size", "Email list size", 1000, min=201, max=10000, step=1)

def compute_alpha_beta(μ, σ):
    """Method of moments calculation of Beta distribution parameters"""
    μ = float(μ)
    σ = float(σ)
    α = -μ + ((1 - μ)*μ**2)/(σ**2)
    β = -α + (α/μ)
    return α, β

@render.text
def display_alpha_beta():
    α, β = compute_alpha_beta(input.open_rate_avg()/100, input.open_rate_stg()/100)
    return f"α = {α:.2f}, β = {β:.2f}"

@render.plot(alt="Plot of beta distribution")
def display_beta_plot():
    α, β = compute_alpha_beta(input.open_rate_avg()/100, input.open_rate_stg()/100)
    x = np.linspace(0, 1, 100)
    y = betadist.pdf(x, α, β)
    fig, ax = plt.subplots()
    ax.plot(100*x, y)
    ax.set_title("Estimated distribution of rates")
    ax.tick_params(labelleft=False)
    ax.set_xlabel("Percent")
    sns.despine()
    return fig

def see_if_we_pick_higher_rate_parallel(size, a, b, n):
    # If we pick the A B sample that did the best, did we pick the one with
    # the higher open rate?
    λ = betadist.rvs(a, b, size=(2, n))
    for ii in range(n): # Flip to have the first be largest
        if λ[0, ii] < λ[1, ii]:
            λ[0, ii], λ[1, ii] = λ[1, ii], λ[0, ii]
    sample_0 = np.random.rand(size, n)
    sample_1 = np.random.rand(size, n)
    
    λ_0 = λ[0, :]
    λ_0.shape = (1, n)
    λ_1 = λ[1, :]
    λ_1.shape = (1, n)

    sample_0 = sample_0 < λ_0
    sample_1 = sample_1 < λ_1

    sums_0 = sample_0.sum(axis=0)
    sums_1 = sample_1.sum(axis=0)
    
    total = sum(sums_0 > sums_1)/n
    return total

@render.plot(alt="Chance that better performing email is best")
def simple_mc_best_performer():
    n_runs = 10000
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000]

    α, β = compute_alpha_beta(input.open_rate_avg()/100, input.open_rate_stg()/100)
    correct_pick_rate = [0] * len(sample_sizes)

    for ii, size in enumerate(sample_sizes):
        pick_rate = see_if_we_pick_higher_rate_parallel(size, α, β, n_runs)
        correct_pick_rate[ii] = pick_rate*100
    
    fig, ax = plt.subplots()

    ax.semilogx(sample_sizes, correct_pick_rate, marker='o')
    ax.set_xlabel("Sample size (A or B)", fontsize=14)
    ax.set_ylabel("Chance of correct pick (%)", fontsize=14)
    ax.set_title("Chance making right pick from small sample", fontsize=12)
    ax.tick_params(labelsize=14)
    ax.grid()
    sns.despine()

    return fig


def loss_versus_oracle_parallel(mc_runs, m, n, a, b):
    λ = betadist.rvs(a, b, size=(2, mc_runs))

    sample_0 = np.random.rand(n, mc_runs)
    sample_1 = np.random.rand(n, mc_runs)

    λ_0 = λ[0, :]
    λ_0.shape = (1, mc_runs)
    λ_1 = λ[1, :]
    λ_1.shape = (1, mc_runs)

    sample_0 = sample_0 < λ_0
    sample_1 = sample_1 < λ_1

    sample_0_score = sample_0.sum(axis=0)
    sample_1_score = sample_1.sum(axis=0)

    best_score = np.maximum(sample_0_score, sample_1_score)
    worst_score = np.minimum(sample_0_score, sample_1_score)
    average_score = (best_score + worst_score)/2

    a_test = sample_0[:m, :]
    b_test = sample_1[m:2*m, :]
    a_score = a_test.sum(axis=0)
    b_score = b_test.sum(axis=0)

    winner = a_score >= b_score
    winner_choice = np.tile(winner, (n - 2*m, 1))

    picked = np.where(winner_choice, sample_0[2*m:, :], sample_1[2*m:, :])

    final_data = np.vstack((a_test, b_test, picked))
    final_score = final_data.sum(axis=0)
    regret = (best_score - final_score)/n
    # gain = (final_score - worst_score)/n
    # avg_gain = (final_score - average_score)/n
    return regret #, gain, avg_gain


@render.plot(alt="Loss for different sample sizes")
def mc_loss_estimator():
    n_runs = 10000
    total_pop = input.list_size()
    sample_sizes = np.array([10, 20, 50, 100, 200, 500, 1000])
    sample_sizes = sample_sizes[sample_sizes < total_pop/2]
    dfs = []
    α, β = compute_alpha_beta(input.open_rate_avg()/100, input.open_rate_stg()/100)

    for size in sample_sizes:
        regrets = loss_versus_oracle_parallel(n_runs, size, total_pop, α, β)
        df = pd.DataFrame({"size": [size] * n_runs, "regret": regrets})
        dfs.append(df)

    final_data2 = pd.concat(dfs)
    # Convert to percentage
    final_data2["regret"] *= 100

    avg_regret = final_data2.groupby("size").mean()
    avg_regret.reset_index(inplace=True)

    fig, ax = plt.subplots(1, 2)

    sns.boxplot(x="size", y="regret", data=final_data2, showfliers=False, ax=ax[0])
    ax[0].set_xlabel("Size", fontsize=14)
    ax[0].set_ylabel("Regret (%)", fontsize=14)
    ax[0].set_title(f"Regret range", fontsize=12)
    ax[0].tick_params(labelsize=12)

    ax[1].semilogx(avg_regret["size"], avg_regret["regret"], marker="o", label="Mean", color="k")
    ax[1].set_xlabel("Size", fontsize=14)
    ax[1].set_title(f"Average regret", fontsize=12)
    ax[1].tick_params(labelsize=12)

    return fig

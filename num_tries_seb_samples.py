# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:28:44 2022

@author: Aakash
"""

from random import shuffle, randint
import matplotlib.pyplot as plt
from statistics import stdev, median, mean


def run_game(num_tries):
    """
    Runs a single game of finding the random value from the sequence

    Appends the guess to num_tries
    """

    elements = [x for x in range(1, 101)]
    to_find = randint(1, 100)

    for n in range(100):
        shuffle(elements)
        guess = elements.pop()
        if guess == to_find:
            num_tries.append(n)
            return num_tries


def loop_tries(num_tries, runs=1000):
    """
    Loops the game several times to get a good estimate of the average number
    of guesses needed
    """

    for i in range(runs):
        run_game(num_tries)

    return num_tries


def plot_results(num_tries):
    plt.hist(num_tries)
    plt.title('Number of draws needed to get correct value')
    plt.xlabel('Num Draws')
    plt.ylabel('Frequency')


def print_stats(num_tries):
    avg_guesses = mean(num_tries)
    stdev_guess = stdev(num_tries)
    median_guess = median(num_tries)

    print(f'The mean number of draws was {avg_guesses:.3f} for success.')
    print(f'This had a standard deviation of {stdev_guess:.3f}.')
    print(f'Similarly, the median number of guesses was {median_guess}.')


def main():
    num_tries = []
    loop_tries(num_tries, runs=10000)

    plot_results(num_tries)
    print_stats(num_tries)


if __name__ == '__main__':
    main()

from ga import GeneticAlgorithm
from config import AlgorithmConfig, ProblemDefinition
from constants import *

# Types of selection, mutation and crossovers -> https://deap.readthedocs.io/en/master/api/tools.html#operators

# Opis
# 1)	Dokonaj optymalizacji twojej funkcji z projektu nr 1 oraz 2.
# 2)	Wykorzystaj algorytmy wskazane przez prowadzącego. Poeksperymentuj z mniej znanymi technikami.
#       a) mutacje mutShuffleIndexes, mutFlipBit i jeszcze coś
#       b) krzyżowania cxUniform, krzyżowanie arytmetyczne, krzyżowanie heurystyczne
#       c) selekcje selRandom, selBest, selWorst, selRoulette
# 3)	Uzupełnij program o przedstawianie najlepszych wyników (wartość funkcji celu, średniej, odchylenia standardowego na wykresie).
# 4)	Program wykonaj w wersji console application. Wykresy przygotuj z wykorzystaniem biblioteki matplotlib bądź wygeneruj w Excelu(lub dowolnym innym programie z którego korzystasz).
# 5)	Przygotuj sprawozdanie podobne do sprawozdań z projektów nr 1 oraz nr 2 (porównanie różnych konfiguracji algorytmu genetycznego).
# 6)	Wykonaj eksperyment sprawdzający jakość zrównoleglenia twojego projektu.
# Skonfiguruj twój algorytm na 1000 epok oraz 1000 osobników.
# Uruchom go dla 1,2,4,8,16 procesów.
# Zmierz czas obliczeń.
# Wykonaj wykres zależności czasu od liczby procesów.
# Celem lepszej obserwacji rezultatów można oczywiście jeszcze zwiększyć liczbę osobników oraz epok.

definition = ProblemDefinition()
config = AlgorithmConfig(
    begin=-10,
    end=10,
    population_size=1000,
    epoch_number=1000,
    selection=SEL_TOURNAMENT,
    crossover_func=CR_ARITHMETIC_CROSSOVER,
    mutation_func=MUT_ES_LOG_NORMAL,
    crossover_prob=0.8,
    mutation_prop=0.2,
    elite_amount=1
)
ga = GeneticAlgorithm(definition, config)
ga.solve()

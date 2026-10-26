#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Лабораторная работа № 1 по дисциплине «Генетические алгоритмы». Вариант 4.

Решение задачи о коммивояжере генетическим алгоритмом.

Требования варианта 4:
  * хромосома — замкнутый маршрут, начинающийся и заканчивающийся
    в стартовом городе (значение 1 стоит в начале и в конце маршрута);
  * отбор родителей — турнирная селекция (k = 3, выбран самостоятельно);
  * скрещивание — двухточечное: обмен средними частями, устранение
    конфликтов нулями и последующая корректировка (восстановление)
    заполнением нулей ближайшими допустимыми городами «от середины
    к концам» маршрута;
  * мутация — выбранный город переносится в позицию рядом с ближайшим
    к нему городом (j-1 или j+1);
  * учёт ограничений — корректировка недопустимой хромосомы (repair).

Исходные данные задачи читаются из JSON-файла задачи, параметры алгоритма —
из отдельного JSON-файла (по условию работы они не смешиваются).

Примеры запуска:
    python ga_tsp.py
    python ga_tsp.py problem.json --verbose
    python ga_tsp.py problem.json --runs 10 --bruteforce
    python ga_tsp.py problem2.json --runs 10
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from itertools import permutations

TOURNAMENT_K = 3  # размер турнира

DEFAULT_PARAMS = {
    "population_size": 60,          # размер популяции
    "crossover_probability": 0.9,   # вероятность скрещивания пары
    "mutation_probability": 0.3,    # вероятность мутации хромосомы
    "max_generations": 200,         # максимальное число поколений
    "no_improve_limit": 40,         # остановка после стольких поколений без улучшений
}


def _is_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value))


def load_problem(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Ошибка: файл задачи не найден: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Ошибка: файл {path} не является корректным JSON: {e}")

    errors = []
    if not isinstance(data, dict):
        raise SystemExit("Ошибка: содержимое файла задачи должно быть JSON-объектом")

    matrix = data.get("distance_matrix")
    n = None
    if not isinstance(matrix, list) or len(matrix) < 3:
        errors.append('поле "distance_matrix" должно быть квадратной матрицей '
                      '(массив массивов) размера не меньше 3')
        matrix = None
    else:
        n = len(matrix)
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != n:
                errors.append(f"distance_matrix[{i}]: ожидается строка длины {n} "
                              f"(матрица должна быть квадратной)")
                continue
            for j, d in enumerate(row):
                if not _is_number(d):
                    errors.append(f"distance_matrix[{i}][{j}] = {d!r}: ожидается число")
                elif i == j and d != 0:
                    errors.append(f"distance_matrix[{i}][{j}] = {d!r}: диагональный "
                                  f"элемент должен быть равен 0")
                elif i != j and d < 0:
                    errors.append(f"distance_matrix[{i}][{j}] = {d!r}: расстояние "
                                  f"должно быть неотрицательным")

    start = data.get("start_city")
    if not isinstance(start, int) or isinstance(start, bool):
        errors.append('поле "start_city": ожидается целое число')
    elif n is not None and not 1 <= start <= n:
        errors.append(f"start_city = {start}: ожидается значение от 1 до {n}")

    if errors:
        print("Файл задачи не прошёл проверку корректности:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    return matrix, start


def load_params(path):
    """Читает JSON-файл параметров алгоритма, недостающие берёт из DEFAULT_PARAMS."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Файл параметров {path} не найден — используются параметры по умолчанию")
        return dict(DEFAULT_PARAMS)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Ошибка: файл {path} не является корректным JSON: {e}")

    if not isinstance(data, dict):
        raise SystemExit("Ошибка: файл параметров должен содержать JSON-объект")

    unknown = set(data) - set(DEFAULT_PARAMS)
    if unknown:
        raise SystemExit(
            "Ошибка: неизвестные параметры: " + ", ".join(sorted(unknown))
            + "\nДопустимые имена: " + ", ".join(sorted(DEFAULT_PARAMS)))

    errors = []
    params = dict(DEFAULT_PARAMS)

    int_limits = {
        "population_size": 2,
        "max_generations": 1,
        "no_improve_limit": 1,
    }
    for key, low in int_limits.items():
        if key in data:
            v = data[key]
            if not isinstance(v, int) or isinstance(v, bool) or v < low:
                errors.append(f"{key}: ожидается целое число не меньше {low}")
            else:
                params[key] = v

    for key in ("crossover_probability", "mutation_probability"):
        if key in data:
            v = data[key]
            if not _is_number(v) or not 0.0 <= v <= 1.0:
                errors.append(f"{key}: ожидается число от 0 до 1")
            else:
                params[key] = v

    if errors:
        print("Файл параметров не прошёл проверку корректности:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    return params


# --------------------------------------------------------------------------
# Генетический алгоритм
# --------------------------------------------------------------------------

class GeneticTSP:
    """Генетический алгоритм для задачи о коммивояжере (вариант 4)."""

    def __init__(self, matrix, start_city, params, rng):
        self.n = len(matrix)          # число городов
        self.start = start_city       # стартовый город (1..N)
        self.cities = list(range(1, self.n + 1))
        # D[u][v] — расстояние из города u в город v (города нумеруются с 1);
        # нулевые строка/столбец — технические, для индексации с единицы.
        self.D = [None] + [[None] + list(row) for row in matrix]

        self.pop_size = params["population_size"]
        self.p_crossover = params["crossover_probability"]
        self.p_mutation = params["mutation_probability"]
        self.max_generations = params["max_generations"]
        self.no_improve_limit = params["no_improve_limit"]

        self.rng = rng

    # --- оценка особи ------------------------------------------------------

    def route_length(self, route):
        """Длина замкнутого маршрута (значение целевой функции)."""
        total = 0
        for k in range(self.n):
            total += self.D[route[k]][route[k + 1]]
        return total

    # --- вспомогательное ---------------------------------------------------

    def _nearest(self, ref, candidates):
        """Ближайший к городу ref город из списка candidates.

        Расстояние смотрится по строке ref (D[ref][c]); при равенстве
        расстояний выбирается город с меньшим номером.
        """
        best = None
        best_d = None
        for c in candidates:
            d = self.D[ref][c]
            if best is None or d < best_d or (d == best_d and c < best):
                best, best_d = c, d
        return best

    def random_genome(self):
        """Случайная хромосома: стартовый город + перестановка остальных + старт."""
        interior = [c for c in self.cities if c != self.start]
        self.rng.shuffle(interior)
        return [self.start] + interior + [self.start]

    def tournament(self, population, fits):
        """Турнирная селекция: побеждает лучший из k случайных особей."""
        best_idx = self.rng.randrange(len(population))
        for _ in range(TOURNAMENT_K - 1):
            idx = self.rng.randrange(len(population))
            if fits[idx] > fits[best_idx]:
                best_idx = idx
        return list(population[best_idx])

    # --- скрещивание (по варианту 4) ----------------------------------------

    def crossover(self, parent1, parent2):
        """Двухточечный кроссовер с обменом средних частей и корректировкой.

        1) У потомков фиксируется стартовый город в начале и в конце.
        2) Маршруты родителей делятся двумя секущими на три части.
        3) Средние части обмениваются.
        4) Боковые части заполняются городами «своего» родителя, начиная от
           средней части к концам; при конфликте (город уже есть в средней
           части) ставится 0, и остальные позиции этой стороны тоже 0.
        5) Нули заполняются ближайшими допустимыми городами от середины
           к концам маршрута (корректировка, repair).
        """
        if self.n < 3 or self.rng.random() >= self.p_crossover:
            return list(parent1), list(parent2)
        n = self.n
        a = self.rng.randint(1, n - 2)        # левая секущая (средняя часть [a, b])
        b = self.rng.randint(a, n - 2)        # правая секущая

        child1 = [0] * (n + 1)
        child2 = [0] * (n + 1)
        child1[0] = child1[n] = child2[0] = child2[n] = self.start

        for i in range(a, b + 1):             # обмен средними частями
            child1[i] = parent2[i]
            child2[i] = parent1[i]

        self._fill_sides(child1, parent1, a, b)
        self._fill_sides(child2, parent2, a, b)
        self._repair(child1, a, b)
        self._repair(child2, a, b)
        return child1, child2

    def _fill_sides(self, child, parent, a, b):
        """Шаг 4 кроссовера: заполнение боковых частей городами родителя.

        Заполнение идёт от средней части к концам. Конфликт — город уже
        присутствует в средней части потомка (или совпадает со стартовым):
        в конфликтную позицию и во все более далёкие от середины позиции
        этой стороны записывается 0.
        """
        middle = set(child[a:b + 1])
        broken = False
        for pos in range(a - 1, 0, -1):               # левая часть: к началу
            if broken:
                child[pos] = 0
                continue
            city = parent[pos]
            if city in middle or city == self.start:
                child[pos] = 0
                broken = True
            else:
                child[pos] = city
        broken = False
        for pos in range(b + 1, self.n):              # правая часть: к концу
            if broken:
                child[pos] = 0
                continue
            city = parent[pos]
            if city in middle or city == self.start:
                child[pos] = 0
                broken = True
            else:
                child[pos] = city

    def _repair(self, child, a, b):
        """Шаг 5 кроссовера: корректировка (восстановление) потомка.

        Нули заменяются ближайшими допустимыми (ещё не использованными)
        городами; заполнение — «из середины к концам» маршрута: для очередной
        нулевой позиции ближайший город ищется к уже размещённому соседу,
        примыкающему к средней части.
        """
        present = set(child)
        missing = [c for c in self.cities if c not in present]
        for pos in range(a - 1, 0, -1):               # левая часть: к началу
            if child[pos] == 0:
                city = self._nearest(child[pos + 1], missing)
                child[pos] = city
                missing.remove(city)
        for pos in range(b + 1, self.n):              # правая часть: к концу
            if child[pos] == 0:
                city = self._nearest(child[pos - 1], missing)
                child[pos] = city
                missing.remove(city)
        return child


    def mutate(self, genome):
        if self.rng.random() >= self.p_mutation:
            return genome
        pos_i = self.rng.randint(1, self.n - 1)       # позиция внутреннего города
        ti = genome[pos_i]
        tj = self._nearest(ti, [c for c in self.cities if c != ti])
        j = genome.index(tj)                          # позиция города tj (0-based)
        target = j if self.rng.random() < 0.5 else j + 2   # 1-based позиция вставки
        target = max(2, min(target, self.n))          # только внутренние позиции
        genome.pop(pos_i)
        genome.insert(target - 1, ti)
        return genome

    # --- основной цикл -----------------------------------------------------

    def run(self, verbose=False):
        population = [self.random_genome() for _ in range(self.pop_size)]
        fits = [-self.route_length(g) for g in population]

        best_route = None
        best_length = None
        best_found_gen = 0
        generations_done = 0
        stall = 0

        for gen in range(1, self.max_generations + 1):
            generations_done = gen

            # Размножение: турнирный выбор родителей, кроссовер, мутация.
            offspring = []
            while len(offspring) < self.pop_size:
                parent1 = self.tournament(population, fits)
                parent2 = self.tournament(population, fits)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(child1))
                if len(offspring) < self.pop_size:
                    offspring.append(self.mutate(child2))

            # Элитный отбор: новое поколение — лучшие особи из объединения
            # родителей и потомков (приспособленность = -длина маршрута).
            candidates = population + offspring
            cand_fits = fits + [-self.route_length(g) for g in offspring]
            order = sorted(range(len(candidates)), key=lambda i: cand_fits[i], reverse=True)
            top = order[:self.pop_size]
            population = [candidates[i] for i in top]
            fits = [cand_fits[i] for i in top]

            # Запоминаем эталонное (коротчайшее) решение.
            cur_length = -fits[0]
            if best_length is None or cur_length < best_length - 1e-9:
                best_length = cur_length
                best_route = list(population[0])
                best_found_gen = gen
                stall = 0
            else:
                stall += 1

            if verbose and (gen == 1 or gen % 10 == 0):
                print(f"  поколение {gen:3d}: длина лучшего маршрута = {-fits[0]:.10g}")

            if stall >= self.no_improve_limit:
                break

        return {
            "route": best_route,
            "length": best_length,
            "fitness": -best_length,
            "generations": generations_done,
            "best_found_gen": best_found_gen,
        }


def brute_force(ga):
    if ga.n > 10:
        return None
    others = [c for c in ga.cities if c != ga.start]
    best_length = None
    best_route = None
    for perm in permutations(others):
        route = [ga.start] + list(perm) + [ga.start]
        length = ga.route_length(route)
        if best_length is None or length < best_length:
            best_length = length
            best_route = route
    return {"length": best_length, "route": best_route}


def route_to_str(route):
    return " -> ".join(str(c) for c in route)


def print_solution(result, title="Лучший найденный маршрут"):
    print()
    print(title)
    print("  Маршрут: " + route_to_str(result["route"]))
    print(f"  Значение целевой функции (длина маршрута): {result['length']:g}")
    print(f"  Значение функции приспособленности:        {-result['length']:g}")
    print("  Каждый город посещается ровно один раз — маршрут допустим")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Генетический алгоритм решения задачи о коммивояжере (вариант 4)")
    parser.add_argument("problem", nargs="?", default="problem.json",
                        help="JSON-файл с исходными данными задачи (по умолчанию problem.json)")
    parser.add_argument("--params", default="ga_params.json",
                        help="JSON-файл с параметрами алгоритма (по умолчанию ga_params.json)")
    parser.add_argument("--runs", type=int, default=1,
                        help="число запусков для оценки устойчивости (по умолчанию 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="начальное зерно генератора случайных чисел")
    parser.add_argument("--verbose", action="store_true",
                        help="печатать динамику по поколениям (при одном запуске)")
    parser.add_argument("--bruteforce", action="store_true",
                        help="сравнить результат с точным перебором всех маршрутов (N <= 10)")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    if args.runs < 1:
        raise SystemExit("Ошибка: число запусков должно быть не меньше 1")

    matrix, start = load_problem(args.problem)
    params = load_params(args.params)

    n = len(matrix)
    symmetric = all(matrix[i][j] == matrix[j][i]
                    for i in range(n) for j in range(n) if i != j)

    print("=" * 68)
    print("Лабораторная работа № 1. Вариант 4. Задача о коммивояжере")
    print("=" * 68)
    print(f"Файл задачи: {args.problem}")
    print(f"Число городов N = {n}, стартовый город = {start}")
    print(f"Матрица расстояний: {'симметричная' if symmetric else 'асимметричная'}")
    print("Проверка входных данных: OK")
    print("Способ учёта ограничений: корректировка (восстановление) потомка "
          "при скрещивании; мутация допустимость не нарушает")
    print(f"Параметры: популяция = {params['population_size']}, "
          f"p_скрещ = {params['crossover_probability']:g}, "
          f"p_мут = {params['mutation_probability']:g}, "
          f"турнир k = {TOURNAMENT_K}, "
          f"поколений до {params['max_generations']}, "
          f"стагнация {params['no_improve_limit']}")
    print()

    results = []
    for run_no in range(1, args.runs + 1):
        seed = None if args.seed is None else args.seed + run_no - 1
        ga = GeneticTSP(matrix, start, params, random.Random(seed))
        start_time = time.perf_counter()
        result = ga.run(verbose=(args.verbose and args.runs == 1))
        result["time"] = time.perf_counter() - start_time
        results.append(result)
        print(f"Запуск {run_no:2d}: длина маршрута = {result['length']:g}, "
              f"поколений = {result['generations']}, "
              f"лучшее найдено в поколении {result['best_found_gen']} "
              f"({result['time']:.2f} c)")

    if args.runs > 1:
        lengths = [r["length"] for r in results]
        best_length = min(lengths)
        print()
        print(f"Устойчивость по {args.runs} запускам:")
        print(f"  лучшая длина маршрута  = {best_length:g}")
        print(f"  худшая длина маршрута  = {max(lengths):g}")
        print(f"  средняя длина маршрута = {statistics.mean(lengths):g}")
        print(f"  СКО длин маршрутов     = {statistics.pstdev(lengths):g}")
        hits = sum(1 for length in lengths if length == best_length)
        print(f"  лучший результат достигнут в {hits} из {args.runs} запусков")

    best = min(results, key=lambda r: r["length"])
    print_solution(best)

    if args.bruteforce:
        ga = GeneticTSP(matrix, start, params, random.Random(0))
        exact = brute_force(ga)
        if exact is None:
            print(f"\nПолный перебор пропущен: N = {n} > 10")
        else:
            print(f"\nТочный оптимум (полный перебор): длина = {exact['length']:g}, "
                  f"маршрут: {route_to_str(exact['route'])}")
            gap = best["length"] - exact["length"]
            if exact["length"] > 0:
                percent = gap / exact["length"] * 100.0
                print(f"Отклонение результата ГА от оптимума: {gap:g} ({percent:.2f}%)")
            else:
                print(f"Отклонение результата ГА от оптимума: {gap:g}")


if __name__ == "__main__":
    main()

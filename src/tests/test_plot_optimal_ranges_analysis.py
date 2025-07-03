from araras.optuna.analyze import plot_optimal_ranges_analysis
def main():
    print("Testing plot_optimal_ranges_analysis from araras.optuna.analyze")
    try:
        res = plot_optimal_ranges_analysis("dummy", "dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

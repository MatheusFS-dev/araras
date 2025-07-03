from araras.optuna.analyze import plot_trend_analysis
def main():
    print("Testing plot_trend_analysis from araras.optuna.analyze")
    try:
        res = plot_trend_analysis("dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
